import os
import numpy as np
import cv2
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import noise_generator
import utils
import matplotlib
import time
import scipy.io
class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]


class Noise2CleanDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.in_root = opt.in_root
        fh = open(self.in_root, 'r')
        imgs = []
        target = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0]))
            target.append(words[1])
        self.imgs = imgs
        self.target = target
        fh.close()

        # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, noise, clean):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                noise = np.rot90(noise, rotate)
                clean = np.rot90(clean, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                noise = cv2.flip(noise, flipCode = 1)
                clean = cv2.flip(clean, flipCode = 1)
        return noise, clean
    def NormMinandMax(self, npdarr, min=-1, max=1):

        arr = npdarr.flatten()
        Ymax = np.max(arr)  # 计算最大值
        Ymin = np.min(arr)  # 计算最小值
        k = (max - min) / (Ymax - Ymin)
        last = min + k * (npdarr - Ymin)

        return last
    def img_sharpen(self, img):
        #自定义卷积核
        kernel_sharpen = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

        # kernel_sharpen = np.array([
        #         [-1,-1,-1,-1,-1],
        #         [-1,2,2,2,-1],
        #         [-1,2,8,2,-1],
        #         [-1,2,2,2,-1], 
        #         [-1,-1,-1,-1,-1]])/8.0
        #卷积
        output = cv2.filter2D(img,-1,kernel_sharpen)
        return output

    def __getitem__(self, index):
        # Define path
        noise_path= self.imgs[index]
        clean_path=self.target[index]
        # Read images
        # input
        
        noise_r = Image.open(noise_path).convert('RGB')
        noise_r = np.array(noise_r).astype(np.float32)
        h, w = noise_r.shape[:2]
        # print(h,w)
        rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
        noise_r = noise_r[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
         
        # noise_r = self.NormMinandMax(noise_r, -1, 1)
        noise_r = (noise_r - 128) / 128
        # noise_r = (noise_r) / 255.0
        # output
        clean = Image.open(clean_path).convert('RGB')
        clean = np.array(clean).astype(np.float32)
        clean = clean[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
        # clean = self.img_sharpen(clean)
        # clean = self.NormMinandMax(clean, -1, 1)
        clean = (clean - 128) / 128
        # clean = (clean) / 255.0
        noise_s, noise_level_map = noise_generator.Poisson_Gaussian_random(clean)
        noise_s = np.array(noise_s).astype(np.float32)
        noise_level_map = np.array(noise_level_map).astype(np.float32)
        noise_s = np.clip(noise_s, -1, 1)
        noise_level_map = np.clip(noise_level_map, -1, 1)
        # noise_level_map = self.NormMinandMax(noise_level_map, -1, 1)
        noise_r = torch.from_numpy(noise_r.transpose(2, 0, 1).astype(np.float32)).contiguous()
        noise_s = torch.from_numpy(noise_s.transpose(2, 0, 1).astype(np.float32)).contiguous()
        clean = torch.from_numpy(clean.transpose(2, 0, 1).astype(np.float32)).contiguous()
        noise_level_map = torch.from_numpy(noise_level_map.transpose(2, 0, 1).astype(np.float32)).contiguous()
        return noise_r, noise_s, clean, noise_level_map
    
    def __len__(self):
        return len(self.imgs)
class ValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.val_root = opt.val_root
        fh = open(self.val_root, 'r')
        imgs = []
        target = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0]))
            target.append(words[1])
        self.imgs = imgs
        self.target = target
        fh.close()

        # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, noise, clean):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                noise = np.rot90(noise, rotate)
                clean = np.rot90(clean, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                noise = cv2.flip(noise, flipCode = 1)
                clean = cv2.flip(clean, flipCode = 1)
        return noise, clean
    
    def NormMinandMax(self, npdarr, min=-1, max=1):

        arr = npdarr.flatten()
        Ymax = np.max(arr)  # 计算最大值
        Ymin = np.min(arr)  # 计算最小值
        k = (max - min) / (Ymax - Ymin)
        last = min + k * (npdarr - Ymin)

        return last

    def img_sharpen(self, img):
        #自定义卷积核
        kernel_sharpen = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

        # kernel_sharpen = np.array([
        #         [-1,-1,-1,-1,-1],
        #         [-1,2,2,2,-1],
        #         [-1,2,8,2,-1],
        #         [-1,2,2,2,-1], 
        #         [-1,-1,-1,-1,-1]])/8.0
        #卷积
        output = cv2.filter2D(img,-1,kernel_sharpen)
        return output

    def __getitem__(self, index):
        # Define path
        noise_path= self.imgs[index]
        clean_path=self.target[index]
        # Read images
        # input
        
        noise_r = Image.open(noise_path).convert('RGB')
        noise_r = np.array(noise_r).astype(np.float32)
        h, w = noise_r.shape[:2]
        # print(h,w)
        rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
        noise_r = noise_r[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
         
        # noise_r = self.NormMinandMax(noise_r, -1, 1)
        noise_r = (noise_r - 128) / 128.0
        # noise_r = (noise_r) / 255.0
        # output
        clean = Image.open(clean_path).convert('RGB')
        clean = np.array(clean).astype(np.float32)
        clean = clean[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
        # clean = self.img_sharpen(clean)
        # clean = self.NormMinandMax(clean, -1, 1)
        clean = (clean - 128) /128.0
        noise_r, clean = self.img_aug(noise_r, clean)
        noise_s, noise_level_map = noise_generator.Poisson_Gaussian_random(clean)
        noise_s = np.array(noise_s).astype(np.float32)
        noise_level_map = np.array(noise_level_map).astype(np.float32)
        noise_s = np.clip(noise_s, -1, 1)
        noise_level_map = np.clip(noise_level_map, -1, 1)
        # noise_level_map = self.NormMinandMax(noise_level_map, -1, 1)
        noise_r = torch.from_numpy(noise_r.transpose(2, 0, 1).astype(np.float32)).contiguous()
        noise_s = torch.from_numpy(noise_s.transpose(2, 0, 1).astype(np.float32)).contiguous()
        clean = torch.from_numpy(clean.transpose(2, 0, 1).astype(np.float32)).contiguous()
        noise_level_map = torch.from_numpy(noise_level_map.transpose(2, 0, 1).astype(np.float32)).contiguous()
        return noise_r, noise_s, clean, noise_level_map
    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.test_root = opt.test_root
        fh = open(self.test_root, 'r')
        imgs = []
        target = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0]))
            target.append(words[1])
        self.imgs = imgs
        self.target = target
        fh.close()

        # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, noise, clean):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                noise = np.rot90(noise, rotate)
                clean = np.rot90(clean, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                noise = cv2.flip(noise, flipCode = 1)
                clean = cv2.flip(clean, flipCode = 1)
        return noise, clean
    
    def NormMinandMax(self, npdarr, min=-1, max=1):

        arr = npdarr.flatten()
        Ymax = np.max(arr)  # 计算最大值
        Ymin = np.min(arr)  # 计算最小值
        k = (max - min) / (Ymax - Ymin)
        last = min + k * (npdarr - Ymin)

        return last

    def img_sharpen(self, img):
        #自定义卷积核
        kernel_sharpen = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

        # kernel_sharpen = np.array([
        #         [-1,-1,-1,-1,-1],
        #         [-1,2,2,2,-1],
        #         [-1,2,8,2,-1],
        #         [-1,2,2,2,-1], 
        #         [-1,-1,-1,-1,-1]])/8.0
        #卷积
        output = cv2.filter2D(img,-1,kernel_sharpen)
        return output

    def __getitem__(self, index):
        # Define path
        noise_path= self.imgs[index]
        clean_path=self.target[index]
        # Read images
        # input
        
        noise_r = Image.open(noise_path).convert('RGB')
        noise_r = np.array(noise_r).astype(np.float32)
        h, w = noise_r.shape[:2]
        # print(h,w)
        rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
        noise_r = noise_r[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
         
        # noise_r = self.NormMinandMax(noise_r, -1, 1)
        noise_r = (noise_r - 128) / 128.0
        # noise_r = (noise_r) / 255.0
        # output
        clean = Image.open(clean_path).convert('RGB')
        clean = np.array(clean).astype(np.float32)
        clean = clean[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
        # clean = self.img_sharpen(clean)
        # clean = self.NormMinandMax(clean, -1, 1)
        clean = (clean - 128) /128.0
        noise_r, clean = self.img_aug(noise_r, clean)
        noise_s, noise_level_map = noise_generator.Poisson_Gaussian_random(clean)
        noise_s = np.array(noise_s).astype(np.float32)
        noise_level_map = np.array(noise_level_map).astype(np.float32)
        noise_s = np.clip(noise_s, -1, 1)
        noise_level_map = np.clip(noise_level_map, -1, 1)
        # noise_level_map = self.NormMinandMax(noise_level_map, -1, 1)
        noise_r = torch.from_numpy(noise_r.transpose(2, 0, 1).astype(np.float32)).contiguous()
        noise_s = torch.from_numpy(noise_s.transpose(2, 0, 1).astype(np.float32)).contiguous()
        clean = torch.from_numpy(clean.transpose(2, 0, 1).astype(np.float32)).contiguous()
        noise_level_map = torch.from_numpy(noise_level_map.transpose(2, 0, 1).astype(np.float32)).contiguous()
        return noise_r, noise_s, clean, noise_level_map
    def __len__(self):
        return len(self.imgs)