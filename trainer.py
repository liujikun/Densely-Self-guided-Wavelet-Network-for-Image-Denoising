import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import dataset
import utils
import os
import noise_generator
from tensorboardX import SummaryWriter
import scipy.io
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils import load_dict
import cv2
from sklearn.metrics import mean_squared_error
#TV loss(total variation regularizer)
def TVLoss(x):

    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    loss = 2*(h_tv/count_h+w_tv/count_w)/batch_size
    return loss

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def Asymmetricloss(noise_est, noise_level_map, alpha = 0.3):
    batch_size = noise_est.size()[0]
    h = noise_est.size()[2]
    w = noise_est.size()[3]
    x = abs(noise_est) - abs(noise_level_map)
    mse = torch.mul(noise_est - noise_level_map, noise_est - noise_level_map)
    mask = torch.lt(x,0)
    res = 0.3 * mse
    res[mask] = (1-alpha) * mse[mask]
    res = torch.mean(res)
    return res 

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def MyDNN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = os.path.join(opt.save_path, opt.task)
    sample_folder = os.path.join(opt.sample_path, opt.task)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_L2 = torch.nn.MSELoss().cuda()
    mse_loss = nn.MSELoss().cuda()
    ms_ssim_module = MS_SSIM(data_range=2, size_average=True, channel=3, nonnegative_ssim=True)
    # Pretrained VGG
    # vgg = MINCFeatureExtractor(opt).cuda()
    # Initialize Generator
    generator = utils.create_MyDNN(opt)
    use_checkpoint = False
    if use_checkpoint:
        checkpoint_path = './MyDNN1_denoise_epoch175_bs1'
        # Load a pre-trained network
        pretrained_net = torch.load(checkpoint_path + '.pth')
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            if epoch < 200:
                lr = 0.0001
            if epoch >= 200:
                lr = 0.00005
            if epoch >= 300:
                lr = 0.00001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator,val_PSNR, best_PSNR):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_best_model and best_PSNR == val_PSNR:
            torch.save(generator, 'final_%s_epoch%d_best.pth' % (opt.task, epoch))
            print('The best model is successfully saved at epoch %d' % (epoch))
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'MyDNN1_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'MyDNN1_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'final_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))

            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'final_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------


    # Define the dataloader
    # trainset = dataset.TestDataset(opt)
    trainset = dataset.Noise2CleanDataset(opt)
    print('The overall number of training images:', len(trainset))
    testset = dataset.TestDataset(opt)
    valset = dataset.ValDataset(opt)
    print('The overall number of val images:', len(valset))
    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    val_loader = DataLoader(valset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    test_loader = DataLoader(testset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    best_PSNR = 0
    # For loop training
    for epoch in range(opt.epochs):
        total_loss = 0
        total_ploss = 0
        total_sobel = 0
        total_Lap = 0
        for i, (true_input, simulated_input, true_target, noise_level_map) in enumerate(dataloader):

            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            simulated_input = simulated_input.cuda()
            noise_level_map = noise_level_map.cuda()
            # Train Generator
            optimizer_G.zero_grad()
            pre_clean = generator(true_input)

            # Parse through VGGMINC layers
            # features_y = vgg(pre_clean)
            # features_x = vgg(true_input)
            # content_loss =  criterion_L2(features_y, features_x).

            pre = pre_clean[0,:,:,:].data.permute(1, 2, 0).cpu().numpy()
            pre = rgb2gray(pre)
            true = true_input[0,:,:,:].data.permute(1, 2, 0).cpu().numpy()
            true = rgb2gray(true)
            laplacian_pre=cv2.Laplacian(pre,cv2.CV_32F)#CV_64F为图像深度
            laplacian_gt=cv2.Laplacian(true,cv2.CV_32F)#CV_64F为图像深度
            sobel_pre = 0.5*(cv2.Sobel(pre,cv2.CV_32F,1,0,ksize=5) + cv2.Sobel(pre,cv2.CV_32F,0,1,ksize=5))#1，0参数表示在x方向求一阶导数
            sobel_gt = 0.5*(cv2.Sobel(true,cv2.CV_32F,1,0,ksize=5) + cv2.Sobel(true,cv2.CV_32F,0,1,ksize=5))#0,1参数表示在y方向求一阶导数
            sobel_loss = mean_squared_error(sobel_pre, sobel_gt)
            laplacian_loss = mean_squared_error(laplacian_pre, laplacian_gt)
            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(pre_clean, true_target) 

            # MS-SSIM loss
            ms_ssim_loss = 1 - ms_ssim_module(pre_clean + 1, true_target + 1)

            # Overall Loss and optimize
            loss =  Pixellevel_L1_Loss + 0.5*laplacian_loss
            # loss =  Pixellevel_L1_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()
            total_loss = Pixellevel_L1_Loss.item() + total_loss
            # total_ploss = content_loss.item() + total_ploss
            total_sobel = sobel_loss + total_sobel
            total_Lap = laplacian_loss + total_Lap

            # # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [laplacian_loss Loss: %.4f] [sobel_loss Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(), laplacian_loss.item(), sobel_loss.item(), time_left))
            img_list = [pre_clean, true_target, true_input]
            name_list = ['pred', 'gt', 'noise']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'MyDNN_MS_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)



            # Learning rate decrease at certain epochs
            lr = adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
        print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [laplacian_loss Loss: %.4f] [sobel_loss Loss: %.4f] Time_left: %s" %
            ((epoch + 1), opt.epochs, i, len(dataloader), total_loss/320, total_Lap/320, total_sobel/320, time_left))
        ### Validation
        val_PSNR = 0
        be_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, simulated_input, true_target, noise_level_map) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                pre_clean = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(pre_clean, true_target, 255) * true_input.shape[0]
            be_PSNR += utils.psnr(true_input, true_target, 255) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image
        be_PSNR = be_PSNR / num_of_val_image

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))
        print('PSNR before denoising %d: %.4f' % ((epoch + 1), be_PSNR))
        best_PSNR = max(val_PSNR,best_PSNR)
        # Save model at certain epochs or iterations
        save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator, val_PSNR, best_PSNR)

