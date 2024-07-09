import glob
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as tv_func

class DeBlurDataset(Dataset):
    def __init__(self , datapath , dataset , phase , patch_size , data_aug , length = None):
        self.datapath = datapath
        self.dataset = dataset
        self.phase = phase
        self.blur_images = sorted(glob.glob("{}/{}/{}/blur/*.png".format(datapath , dataset , phase)))
        self.sharp_images = sorted(glob.glob("{}/{}/{}/sharp/*.png".format(datapath , dataset , phase)))
        self.num_images = len(self.blur_images)
        self.patch_size = patch_size
        self.data_aug = data_aug
        self.aug_threshold = 0.3
        self.curr_batch_length = self.num_images
        if length is not None:
            self.curr_batch_length = length
            
        
    def __len__(self):
        return self.curr_batch_length
    
    def get_actual_dim(self):
        dummy_img = tv_func.to_tensor(Image.open(self.blur_images[0]))
        self.real_width , self.real_height = tv_func.get_image_size(dummy_img)
        return [self.real_height , self.real_width]
    
    def pad_image(self, image , patch_width , patch_height):
        width , height = tv_func.get_image_size(image)
        pad_height = max(patch_height - height , 0)
        pad_width = max(patch_width - width , 0)
        return tv_func.pad(image , [pad_width , pad_height] , padding_mode = "reflect")
    
    def __getitem__(self, idx):
        blur_image = tv_func.to_tensor(Image.open(self.blur_images[idx % self.num_images]))
        sharp_image = tv_func.to_tensor(Image.open(self.sharp_images[idx % self.num_images]))
        padded_blur_image = self.pad_image(blur_image , self.patch_size , self.patch_size)
        padded_sharp_image = self.pad_image(sharp_image , self.patch_size , self.patch_size)
        
        top_left_x , top_left_y , height , width = RandomCrop.get_params(padded_blur_image , (self.patch_size , self.patch_size))
        cropped_blur_image = tv_func.crop(padded_blur_image , top_left_x , top_left_y , height , width)
        cropped_sharp_image = tv_func.crop(padded_sharp_image , top_left_x , top_left_y , height , width)
        if self.data_aug:
            if torch.rand(1) < self.aug_threshold:
                cropped_blur_image = tv_func.hflip(cropped_blur_image)
                cropped_sharp_image = tv_func.hflip(cropped_sharp_image)
            
        return cropped_blur_image , cropped_sharp_image
    
#Functionality Check
# if __name__ == "__main__":
#     datapath = "/home/yogeesh/workspace/datasets/DBlur"
#     dataset = "Gopro"
#     phase = "train"
#     patch_size = 64
#     deblur_ds = DeBlurDataset(datapath , dataset , phase , patch_size , False)
#     bi , si = deblur_ds.__getitem__(0)
#     print(tv_func.get_image_size(bi))
#     print(deblur_ds.get_actual_dim())
        
        
        