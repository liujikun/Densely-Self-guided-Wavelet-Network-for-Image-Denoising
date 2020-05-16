# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:33:42 2019
@author: liujikun@hit.edu.cn
"""
import scipy.io
import numpy as np
import random
import math
import cv2
from PIL import Image
from scipy.stats import norm
import matplotlib.pyplot as plt
import colour_demosaicing
import torch
def mosaic_bayer(rgb, pattern, noiselevel):
    num = np.zeros(len(pattern))
    for i in range (len(pattern)):
        if pattern[i]=='r' or pattern[i]=='R':
            num[i]=0
        if pattern[i]=='g' or pattern[i]=='G':
            num[i]=1
        if pattern[i]=='b' or pattern[i]=='B':
            num[i]=2
#    print(num)
    mosaic = np.zeros((np.size(rgb, 0), np.size(rgb, 1), 3))
    mask = np.zeros((np.size(rgb, 0), np.size(rgb, 1), 3))
    B = np.zeros((np.size(rgb, 0), np.size(rgb, 1)))
    for i in range(np.size(rgb, 0)):
        for j in range(np.size(rgb, 1)):
            if i%2==0 and j%2==0:
                B[int(i),int(j)]=rgb[int(i),int(j),int(num[0])]
                mask[int(i),int(j),int(num[0])]=1
                mosaic[int(i),int(j),int(num[0])]=B[int(i),int(j)]
            if i%2==0 and j%2!=0:
                B[int(i),int(j)]=rgb[int(i),int(j),int(num[1])]
                mask[int(i),int(j),int(num[1])]=1
                mosaic[int(i),int(j),int(num[1])]=B[int(i),int(j)]
            if i%2!=0 and j%2==0:
                B[int(i),int(j)]=rgb[int(i),int(j),int(num[2])]
                mask[int(i),int(j),int(num[2])]=1
                mosaic[int(i),int(j),int(num[2])]=B[int(i),int(j)]
            if i%2!=0 and j%2!=0:
                B[int(i),int(j)]=rgb[int(i),int(j),int(num[3])]
                mask[int(i),int(j),int(num[3])]=1
                mosaic[int(i),int(j),int(num[3])]=B[int(i),int(j)]
    B = B + noiselevel/255*norm.ppf(np.random.rand(np.size(B,0), np.size(B,1)))
    return B, mosaic, mask
def ICRF_Map(Img,I,B):
    w=np.size(Img,0)
    h=np.size(Img,1)
    c=np.size(Img,2)
    bin1 = len(I)
    Size = w*h*c
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    Img=np.reshape(Img, (Size))
    for i in range(0,Size):
        temp = Img[i]
        start_bin = 0
        if temp > min_tiny_bin:
            start_bin = math.floor(temp/tiny_bin - 1)-1
        for b in range(start_bin,bin1):
            tempB = B[b]
            if tempB >= temp:
                index = b
                if index > 0:
                    comp1 = tempB - temp
                    comp2 = temp - B[index-1]
                    if comp2 < comp1:
                        index = index-1
                Img[i] = I[index]
                break
    Img=np.reshape(Img,(w,h,c))
    return Img
def CRF_Map(Img,I,B):
    w=np.size(Img,0)
    h=np.size(Img,1)
    c=np.size(Img,2)
    bin1 = len(I)
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    Size = w*h*c
    Img=np.reshape(Img, (Size))
    for i in range(0,Size):
        temp = Img[i]
        if temp<0:
            temp=0
            Img[i]=0
        if temp>1:
            temp=1
            Img[i]=1
        start_bin = 0
        if temp > min_tiny_bin:
            start_bin = math.floor(temp/tiny_bin - 1)-1
        for b in range(start_bin,bin1):
            tempB = I[b]
            if tempB >= temp:
                index = b
                if index > 0:
                    comp1 = tempB - temp
                    comp2 = temp - I[index-1]
                    if comp2 < comp1:
                        index = index-1
                Img[i] = B[index]
                break
    Img=np.reshape(Img,(w,h,c))
    return Img

def AddNoiseAGWN_random(clean_img):
    
    clean_img = torch.squeeze(clean_img)
    clean_img = clean_img.cpu().detach().numpy()
    clean_img = clean_img.transpose(1, 2, 0)
    clean_img=clean_img.astype(np.float32)
    sigma_c=np.ones(3)
    sigma_c[0] = 16*random.random() # original 0.16
    sigma_c[1] = 16*random.random() 
    sigma_c[2] = 16*random.random() 
    noise_c_map = np.tile(sigma_c[np.newaxis,np.newaxis,:],(np.size(clean_img,0),np.size(clean_img,1),1))
    noise_c = np.multiply(noise_c_map, norm.ppf(np.random.rand(np.size(clean_img,0), np.size(clean_img,1),np.size(clean_img,2))))
    noisy_img = clean_img + noise_c
    noisy_img = noisy_img.transpose(2, 0, 1)
    noisy_img = noisy_img[np.newaxis,:,:,:]
    noisy_img = torch.from_numpy(noisy_img)
    noisy_img = noisy_img.float()
    return noisy_img.cuda()

def Poisson_Gaussian_random(clean_img):
    
    # clean_img = torch.squeeze(clean_img)
    # clean_img = clean_img.cpu().detach().numpy()
    # clean_img = clean_img.transpose(1, 2, 0)
    # clean_img=clean_img.astype(np.float32)

    sigma_s=np.ones(3)
    sigma_c=np.ones(3)
    sigma_s[0] = 0.16*random.random() # original 0.16 Possion
    sigma_s[1] = 0.16*random.random() 
    sigma_s[2] = 0.16*random.random() 
    sigma_c[0] = 0.06*random.random() # original 0.16 Gaussion
    sigma_c[1] = 0.06*random.random() 
    sigma_c[2] = 0.06*random.random() 
    noise_s_map = np.multiply(sigma_s[np.newaxis,np.newaxis,:], clean_img)
    noise_s = np.multiply(norm.ppf(np.random.rand(np.size(clean_img,0), np.size(clean_img,1),np.size(clean_img,2))),noise_s_map)
    noisy_img = clean_img + noise_s
    noise_c_map = np.tile(sigma_c[np.newaxis,np.newaxis,:],(np.size(clean_img,0),np.size(clean_img,1),1))
    noise_c = np.multiply(noise_c_map, norm.ppf(np.random.rand(np.size(clean_img,0), np.size(clean_img,1),np.size(clean_img,2))))
    noisy_img = noisy_img + noise_c

    # noisy_img = noisy_img.transpose(2, 0, 1)
    # noisy_img = noisy_img[np.newaxis,:,:,:]
    # noisy_img = torch.from_numpy(noisy_img)
    # noisy_img = noisy_img.float()

    noise_level_map = noise_c_map + noise_s_map
    # noise_level_map = noise_level_map.transpose(2, 0, 1)
    # noise_level_map = noise_level_map[np.newaxis,:,:,:]
    # noise_level_map = torch.from_numpy(noise_level_map)
    # noise_level_map = noise_level_map.float()

    return noisy_img, noise_level_map

def AddNoiseMosai_random(x,I,B,Iinv,Binv):
#     default value
    channel = np.size(x,2)
    sigma_s=np.ones(3)
    sigma_c=np.ones(3)
    sigma_s[0] = 0.16*random.random() # original 0.16
    sigma_s[1] = 0.16*random.random() 
    sigma_s[2] = 0.16*random.random() 
    sigma_c[0] = 0.06*random.random() # original 0.16
    sigma_c[1] = 0.06*random.random() 
    sigma_c[2] = 0.06*random.random() 
    
    rand_index = np.random.permutation(201)
    crf_index = rand_index[0]
    pattern = np.random.permutation(5)
    temp_x = x
    # # x -> L
    temp_x = ICRF_Map(temp_x,Iinv[crf_index,:],Binv[crf_index,:])
    
    noise_s_map = np.multiply(sigma_s[np.newaxis,np.newaxis,:], temp_x)
    noise_s = np.multiply(norm.ppf(np.random.rand(np.size(x,0), np.size(x,1),np.size(x,2))),noise_s_map)
    temp_x = temp_x + noise_s
    
    noise_c_map = np.tile(sigma_c[np.newaxis,np.newaxis,:],(np.size(x,0),np.size(x,1),1))
    noise_c = np.multiply(noise_c_map, norm.ppf(np.random.rand(np.size(temp_x,0), np.size(temp_x,1),np.size(temp_x,2))))
    temp_x = temp_x + noise_c
    
    noise_level_map=noise_s_map+noise_c_map

#   add Mosai
    if pattern[0] == 0:
        B_b,_ ,mask= mosaic_bayer(temp_x, 'gbrg', 0)
    elif pattern[0] == 1:
        B_b,_ ,_= mosaic_bayer(temp_x, 'grbg', 0)
    elif pattern[0] == 2:
        B_b,_ ,_= mosaic_bayer(temp_x, 'bggr', 0)
    elif pattern[0] == 3:
        B_b,_ ,_= mosaic_bayer(temp_x, 'rggb', 0)
    else:
        B_b = temp_x
    temp_x = B_b.astype(np.float32)
    
#   DeMosai
    if pattern[0] == 0:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='GBRG')
    elif pattern[0] == 1:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='GRBG')
    elif pattern[0] == 2:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='BGGR')
    elif pattern[0] == 3:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='RGGB')
    else:
        lin_rgb = temp_x
    temp_x=lin_rgb.astype(np.float32)
    
#   L -> x
    temp_x = CRF_Map(temp_x,I[crf_index,:],B[crf_index,:])
    
    return temp_x, noise_level_map
    
def AddNoiseMosai(x,I,B,Iinv,Binv,sigma_s,sigma_c,crf_index, pattern):
    temp_x = x
    # x -> L
    temp_x = ICRF_Map(temp_x,Iinv[crf_index,:],Binv[crf_index,:])
    
    noise_s_map = np.multiply(sigma_s[np.newaxis,np.newaxis,:], temp_x)
    noise_s = np.multiply(norm.ppf(np.random.rand(np.size(x,0), np.size(x,1),np.size(x,2))),noise_s_map)
    temp_x = temp_x + noise_s
    
    noise_c_map = np.tile(sigma_c[np.newaxis,np.newaxis,:],(np.size(x,0),np.size(x,1),1))
    noise_c = np.multiply(noise_c_map, norm.ppf(np.random.rand(np.size(temp_x,0), np.size(temp_x,1),np.size(temp_x,2))))
    temp_x = temp_x + noise_c
    noise_level_map=noise_s_map+noise_c_map
#   add Mosai
    if pattern == 0:
        B_b,_ ,mask= mosaic_bayer(temp_x, 'gbrg', 0)
    elif pattern == 1:
        B_b,_ ,_= mosaic_bayer(temp_x, 'grbg', 0)
    elif pattern == 2:
        B_b,_ ,_= mosaic_bayer(temp_x, 'bggr', 0)
    elif pattern == 3:
        B_b,_ ,_= mosaic_bayer(temp_x, 'rggb', 0)
    else:
        B_b = temp_x
    temp_x = B_b.astype(np.float32)
#   DeMosai
    if pattern == 0:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='GBRG')
    elif pattern == 1:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='GRBG')
    elif pattern == 2:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='BGGR')
    elif pattern == 3:
        lin_rgb=colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(temp_x, pattern='RGGB')
    else:
        lin_rgb = temp_x
    temp_x=lin_rgb.astype(np.float32)
#   L -> x
    temp_x = CRF_Map(temp_x,I[crf_index,:],B[crf_index,:])
    return temp_x, noise_level_map
def NormMinandMax(npdarr, min=-1, max=1):

    arr = npdarr.flatten()
    Ymax = np.max(arr)  # 计算最大值
    Ymin = np.min(arr)  # 计算最小值
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (npdarr - Ymin)

    return last
def noise_generate(clean_img):
    # load CRF parameters
    CRF_data = scipy.io.loadmat('./201_CRF_data.mat')
    B=CRF_data['B']
    I=CRF_data['I']
    I_gl=I
    B_gl=B
    Inv=scipy.io.loadmat('./dorfCurvesInv.mat')
    invI=Inv['invI']
    invB=Inv['invB']
    I_inv_gl=invI
    B_inv_gl=invB

    # Realistic Noise Model
    sigma_s = np.array([0.08, 0.08, 0.08]) # recommend 0~0.16
    sigma_c = np.array([0.03, 0.03, 0.03])  # recommend 0~0.06
    CRF_index = 4  # 1~201
    pattern = 0

    # # load images
    # data_path = 'F:\sensetime\denoising\pytoflow\vimeo_septuplet\sequences';
    # noise_path = 'F:\sensetime\denoising\pytoflow\vimeo_septuplet\sequences_with_noise';
    # clean_img=cv2.imread('F:\\sensetime\\denoising\\pytoflow\\vimeo_septuplet\\sequences\\00014\\0020\\im1.png')
    # b,g,r = cv2.split(clean_img) 
    # clean_img = cv2.merge([r,g,b])


    # clean_img = torch.squeeze(clean_img)
    # clean_img = clean_img.cpu().detach().numpy()
    # clean_img = clean_img.transpose(1, 2, 0)
    clean_img = (clean_img + 1) / 2


    noisy_img, noise_level_map = AddNoiseMosai_random(clean_img,I_gl,B_gl,I_inv_gl,B_inv_gl)
    # scipy.io.savemat('./noise_level_map.mat', {'noise_level_map':[noise_level_map]})
    # using AGWN
    # noisy_img = AddNoiseAGWN_random(clean_img)


    # noisy_img = noisy_img.transpose(2, 0, 1)
    # noisy_img = noisy_img[np.newaxis,:,:,:]
    # noise_level_map = noise_level_map.transpose(2, 0, 1)
    # noise_level_map = noise_level_map[np.newaxis,:,:,:]
    # noisy_img = torch.from_numpy(noisy_img)
    # noisy_img = noisy_img.float()
    # noise_level_map = torch.from_numpy(noise_level_map)
    # noise_level_map = noise_level_map.float()
    # noisy_img, noise_level_map = AddNoiseMosai(clean_img,I_gl,B_gl,I_inv_gl,B_inv_gl, sigma_s,sigma_c, CRF_index, pattern)
    return noisy_img,noise_level_map

# plt.imshow(noisy_img)

# scipy.io.savemat('./noisy_img.mat', {'noisy_img':[noisy_img]})
#scipy.io.savemat('./img1.mat', {'img1':[img1]})