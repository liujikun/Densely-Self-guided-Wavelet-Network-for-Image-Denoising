#!/usr/bin/env python

import numpy as np
import os.path
import shutil
from scipy.io.matlab.mio import savemat, loadmat
import network
import argparse
import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import color
import utils
import cv2
from thop import profile
time_total = 0
# def denoiser(noisy):
#     # TODO: plug in your denoiser here

#     out_rgb = colornet(noisy)
#     return out_rgb

parser = argparse.ArgumentParser()
parser.add_argument('--netroot', type = str, default = 'MyDNN_denoise_epoch491_best.pth', help = 'model root')
parser.add_argument('--use_ensemble', type = bool, default = True, help = 'using True for the best performance and please change this to False for runtime testing')
parser.add_argument('--saveroot', type = str, default = './test_results', help = 'result images saveroot')
parser.add_argument('--sample_folder', type = str, default = './sample', help = 'color image baseroot')
opt = parser.parse_args()
# TODO: Download noisy images from:
#  https://competitions.codalab.org/my/datasets/download/69da74e3-e1ed-4e36-9c90-fed7ca04d06f

# TODO: update your working directory; it should contain the .mat file containing noisy images
work_dir = './'
denoiser = torch.load(opt.netroot).cuda()
# macs, params = profile(model, inputs=(input, ))
# denoiser1 = torch.load(opt.netroot1).cuda()
# load noisy images
noisy_fn = 'siddplus_test_noisy_srgb.mat'
noisy_key = 'siddplus_test_noisy_srgb'
noisy_mat = loadmat(os.path.join(work_dir, noisy_fn))[noisy_key]

# denoise
n_im, h, w, c = noisy_mat.shape
results = noisy_mat.copy()

for i in range(n_im):
    noisy = np.reshape(noisy_mat[i, :, :, :], (h, w, c))
    noisy = np.array(noisy).astype(np.float64)
    noisy = (noisy - 128)/128.0
    noisy_img = torch.from_numpy(noisy.transpose(2, 0, 1).astype(np.float32)).contiguous()
    noisy_img = noisy_img.reshape([1, noisy_img.shape[0], noisy_img.shape[1], noisy_img.shape[2]]).cuda()
    if opt.use_ensemble:
        # 8 image with flip and rotate 
        denoised_temp = denoiser(noisy_img)
        denoised_total = denoised_temp[0,:,:,:].clone().data.permute(1, 2, 0).cpu().numpy()
        torch.cuda.empty_cache()
        # rotate
        for rotate in range(1,4):
            noisy_temp = np.rot90(noisy, rotate)
            noisy_img = torch.from_numpy(noisy_temp.transpose(2, 0, 1).astype(np.float32)).contiguous()
            noisy_img = noisy_img.reshape([1, noisy_img.shape[0], noisy_img.shape[1], noisy_img.shape[2]]).cuda()
            denoised_temp =  denoiser(noisy_img)
            denoised_temp = denoised_temp[0,:,:,:].clone().data.permute(1, 2, 0).cpu().numpy()
            denoised_temp = np.rot90(denoised_temp, 4-rotate)
            denoised_total = denoised_total + denoised_temp
            torch.cuda.empty_cache()
        # horizontal flip
        
        noisy = cv2.flip(noisy, flipCode = 0)
        # rotate
        for rotate in range(4):
            noisy_temp = np.rot90(noisy, rotate)
            noisy_img = torch.from_numpy(noisy_temp.transpose(2, 0, 1).astype(np.float32)).contiguous()
            noisy_img = noisy_img.reshape([1, noisy_img.shape[0], noisy_img.shape[1], noisy_img.shape[2]]).cuda()
            denoised_temp =  denoiser(noisy_img)
            denoised_temp = denoised_temp[0,:,:,:].clone().data.permute(1, 2, 0).cpu().numpy()
            if rotate ==0:
                denoised_temp = cv2.flip(denoised_temp, flipCode = 0)
            else:
                denoised_temp = np.rot90(denoised_temp, 4-rotate)
                denoised_temp = cv2.flip(denoised_temp, flipCode = 0)
            denoised_total = denoised_total + denoised_temp
            torch.cuda.empty_cache()

        denoised = denoised_total/8
    else:
    # single image —— using this code to test runtime
        time_start=time.time()
        denoised = denoiser(noisy_img)    
        time_end=time.time()
        time_total=time_end-time_start+time_total
        denoised = denoised[0,:,:,:].clone().data.permute(1, 2, 0).cpu().numpy()

    
    # Recover normalization
    denoised = denoised * 128 + 128
    denoised = np.clip(denoised, 0, 255)
    denoised = denoised.astype(np.uint8)

    denoised_copy = cv2.cvtColor(denoised,cv2.COLOR_BGR2RGB)
    save_denoised_name = str(i) + 'denoised'+'.png'
    save_denoised_path = os.path.join(opt.sample_folder, save_denoised_name)
    cv2.imwrite(save_denoised_path, denoised_copy)

    results[i, :, :, :] = denoised
    print(i)
# create results directory
res_dir = 'res_dir'
os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

# save denoised images in a .mat file with dictionary key "results"
res_fn = os.path.join(work_dir, res_dir, 'results.mat')
res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
savemat(res_fn, {res_key: results})

# submission indormation
# TODO: update the values below; the evaluation code will parse them
if opt.use_ensemble:
    runtime = 0.15846
else:
    runtime = time_total/1024.0  # seconds / per image 

cpu_or_gpu = 0  # 0: GPU, 1: CPU
use_metadata = 0  # 0: no use of metadata, 1: metadata used
other = '(optional) any additional description or information'

# prepare and save readme file
readme_fn = os.path.join(work_dir, res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
with open(readme_fn, 'w') as readme_file:
    readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
    readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
    readme_file.write('Metadata[1] / No Metadata[0]: %s\n' % str(use_metadata))
    readme_file.write('Other description: %s\n' % str(other))

# compress results directory
res_zip_fn = 'results_dir'
shutil.make_archive(os.path.join(work_dir, res_zip_fn), 'zip', os.path.join(work_dir, res_dir))

#  TODO: upload the compressed .zip file here:
#  https://competitions.codalab.org/competitions/22231#participate-submit_results