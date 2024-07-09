import torch 
import metrics
import matplotlib
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from model import Restormer
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.backends import cudnn
from preprocessing import DeBlurDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torchvision.transforms.functional as tv_func
from torch.optim.lr_scheduler import CosineAnnealingLR

matplotlib.use('TkAgg')

def test_model(model , model_file , test_image_path):
    model.eval()
    with torch.no_grad():
        blur_image = (Image.open("{}/blur/1.png".format(test_image_path)))
        model_inp = Resize((400 , 400))(tv_func.to_tensor(blur_image))
        model_inp_cuda = model_inp.cuda().to(torch.bfloat16).unsqueeze(0)
        print(model_inp.size())
        sharp_image = Image.open("{}/sharp/0.png".format(test_image_path))
        model.load_state_dict(torch.load(model_file))
        out = model(model_inp_cuda)
        
        denorm_out = denormalize_image(out , 720 , 1280)
    
    cpu_out = np.asarray(denorm_out.cpu().squeeze(0)).transpose(1,2,0)
    plt.imshow(cpu_out)
    plt.show()
    

def denormalize_image(img , height , width , range = 255):
    img = torch.clamp(img[: , : , :height , :width] , 0 , 1)
    img = torch.clamp(img.mul(range) , 0 , range)
    return img.byte()

def eval_model(model , test_data_loader , iter , total_iter , real_dims):
    model.eval()
    length = 1
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        test_bar = tqdm(test_data_loader , initial = 1 , dynamic_ncols = True)
        for blur_img , sharp_img in test_bar:
            blur_img = blur_img.cuda().to(torch.bfloat16)
            sharp_img = sharp_img.cuda()
            out = model(blur_img)
            denorm_out = denormalize_image(out , real_dims[0] , real_dims[1])
            denorm_gt = denormalize_image(sharp_img , real_dims[0] , real_dims[1])
            current_psnr = metrics.calc_psnr(denorm_out , denorm_gt)
            current_ssim = metrics.calc_ssim(denorm_out , denorm_gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            length += 1
            test_bar.set_description("Test Iter : [{} / {}] PSNR : {:.3f}  SSIM : {:.3f}".format(iter , total_iter , total_psnr / length , total_ssim / length))
            
    return total_psnr / length , total_ssim / length

def save_model(model , save_path , dataset_name):
    torch.save(model.state_dict() , "{}/{}.pth".format(save_path , dataset_name))
    print("Pytorch model has been saved")
    
def train(model_file = None):
    torch.cuda.manual_seed_all(-1)
    cudnn.deterministic = True
    cudnn.benchmark = False
    datapath = "/home/yogeesh/workspace/datasets/DBlur"
    dataset = "Gopro"
    phase = "train"
    num_iter = 4500
    progressive_timeline = [0 , 400 , 700 , 1200 , 1800 , 2000 , 2200]
    progressive_patch_size = [16 , 32 , 48 , 64 , 96 , 128]
    progressive_batch_size = [128 , 54 , 24 , 14 , 6 , 3]
    total_num , total_loss = 0 , 0.0
    val_freq = 500
    
    model = Restormer()
    train_bar = tqdm(range(1, num_iter + 1), initial=1, dynamic_ncols=True)
    test_dataset = DeBlurDataset(datapath , dataset , "test" , progressive_patch_size[-1] , data_aug = False , length = 500)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003 , weight_decay = 1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer , T_max = num_iter , eta_min = 1e-6)
    test_loader = DataLoader(test_dataset , batch_size = 1 , shuffle = False , num_workers = 4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    supports_bfloat16 = torch.cuda.is_bf16_supported()
    model.to(torch.bfloat16)
    print("using bfloat DType")
    
    if(model_file is not None):
        test_model(model , model_file , "{}/{}/test/".format(datapath , dataset))
        return
    
    for n_iter in train_bar:
        if n_iter == 1 or n_iter - 1 in progressive_timeline:
            idx = progressive_timeline.index(n_iter - 1)
            start = n_iter - 1
            end = progressive_timeline[idx + 1] if idx != len(progressive_timeline)-1 else num_iter
            batch_size = progressive_batch_size[idx] if idx < len(progressive_timeline)-1 else progressive_batch_size[-1]
            patch_size = progressive_patch_size[idx] if idx < len(progressive_timeline)-1 else progressive_patch_size[-1]
            length = batch_size * (end - start)
            # print("Training for timeline {} with :  {} , {}".format(n_iter-1 , patch_size , batch_size))
            train_dataset = DeBlurDataset(datapath , dataset , phase , patch_size , data_aug = False , length = length)
            train_loader = iter(DataLoader(train_dataset , batch_size , shuffle = True , num_workers = 2))
    

        model.train()
        blur_images , sharp_images = next(train_loader)
        blur_images , sharp_images = blur_images.to(device) , sharp_images.to(device)
        blur_images , sharp_images = blur_images.to(torch.bfloat16) , sharp_images.to(torch.bfloat16)
        model_out = model(blur_images)
        loss = nn.functional.l1_loss(model_out , sharp_images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += blur_images.size(0)
        total_loss += loss.item() * blur_images.size(0)
        train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                      .format(n_iter, num_iter, total_loss / total_num))
        
        lr_scheduler.step()
        if n_iter % val_freq == 0:
            val_psnr , val_ssim = eval_model(model , test_loader , n_iter , num_iter , test_dataset.get_actual_dim())
            save_model(model , "/home/yogeesh/workspace/restormer/saved_model" , "gopro")
            
        
if __name__ == "__main__":
    train()
    # train("/home/yogeesh/workspace/restormer/saved_model/gopro.pth")
    
        
        
        
    
    
            