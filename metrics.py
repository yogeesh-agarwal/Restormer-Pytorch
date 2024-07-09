import math
import torch 
import matplotlib
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tv_func

matplotlib.use('TkAgg')

def normalise(image , range):
    return image / range

def calc_psnr(image1 , image2 , range = 255.0):
    image1 = normalise(image1 , range)
    image2 = normalise(image2 , range)
    mean_squared_error = torch.mean(torch.square(image1 - image2))
    psnr_score = 20 * torch.log10(255.0 / torch.sqrt(mean_squared_error))
    return psnr_score
        
def calc_ssim(image1 , image2 , kernel_size = 11,
              kernel_sigma = 1.5 , range = 255.0,
              k1 = 0.01 ,
              k2 = 0.03):
    image1 = normalise(image1 , range)
    image2 = normalise(image2 , range)
    
    functional_size = max(1 , round(min(image1.size()[-2:]) / 256))
    if functional_size > 1:
        image1 = nn.functional.avg_pool2d(image1 , kernel_size = functional_size)
        image2 = nn.functional.avg_pool2d(image2 , kernel_size = functional_size)
    
    pad = 0
    _ , channels , height , width = image1.size()
    
    # generate Gaussian Filter
    coords = torch.arange(kernel_size , dtype = image1.dtype , device = image1.device)
    coords -= (kernel_size - 1.0) // 2
    gaussian_filter_1d = torch.square(coords)
    gaussian_filter_1d = (- (gaussian_filter_1d.unsqueeze(0) + gaussian_filter_1d.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    gaussian_filter_1d = gaussian_filter_1d / gaussian_filter_1d.sum()
    kernel = gaussian_filter_1d.unsqueeze(0).repeat(image1.size(1) , 1 , 1 , 1)
    
    #calculate SSIM
    c1 = k1**2
    c2 = k2**2
    img1_mean = nn.functional.conv2d(image1 , weight = kernel , stride = 1 , padding = pad , groups = channels)
    img2_mean = nn.functional.conv2d(image2 , weight = kernel , stride = 1 , padding = pad , groups = channels)
    img1_img2_mean = img1_mean * img2_mean
    img1_mean_sq = torch.square(img1_mean)
    img2_mean_sq = torch.square(img2_mean)
    
    img1_sigma = nn.functional.conv2d(torch.square(image1) , weight = kernel , stride = 1 , padding = pad , groups = channels) - img1_mean_sq
    img2_sigma = nn.functional.conv2d(torch.square(image2) , weight = kernel , stride = 1 , padding = pad , groups = channels) - img2_mean_sq
    img1_img2_sigma = nn.functional.conv2d(image1 * image2 , weight = kernel , stride = 1 , padding = pad , groups = channels) - img1_img2_mean
    
    expnaded_mean = 2 * img1_img2_mean + c1
    expnaded_sigma = 2 * img1_img2_sigma + c2
    collective_sq_mean = img1_mean_sq + img2_mean_sq + c1
    collective_sq_sigma = img1_sigma + img2_sigma + c2
    
    ssim_score = (expnaded_mean * expnaded_sigma) / (collective_sq_mean * collective_sq_sigma)
    return ssim_score.mean()


#functional testing of SSIM and PSNR
# if __name__ == "__main__":
#     img1 = np.asarray(Image.open("/home/yogeesh/workspace/datasets/DBlur/Gopro/train/blur/0.png"))    
#     img2 = np.asarray(Image.open("/home/yogeesh/workspace/datasets/DBlur/Gopro/train/blur/1.png"))
    
#     image1 = torch.tensor(img1.transpose((2,0,1))).unsqueeze(0)
#     image2 = torch.tensor(img2.transpose((2,0,1))).unsqueeze(0)
#     ssim = calc_ssim(image1 , image2)
#     psnr = calc_psnr(image1 , image2)
#     print("SSIM : {} , PSNR : {}".format(ssim , psnr)) 
    
    
    