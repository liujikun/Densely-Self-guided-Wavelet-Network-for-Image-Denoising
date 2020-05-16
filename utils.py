import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import os
import cv2
import network
import scipy.io
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

def create_generator(opt):
    if opt.pre_train:
        # Initialize the network
        generator = network.Generator(opt)
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Initialize the network
        generator = network.Generator(opt)
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_name + '.pth')
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator

def create_discriminator(opt):
    # Initialize the network
    discriminator = network.PatchDiscriminator70(opt)
    # Init the network
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Discriminators is created!')
    return discriminator
def create_CBD_generator(opt):
    # Initialize the network
    CBD_generator = network.CBD_Generator(opt)
    # Init the network
    network.weights_init(CBD_generator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('CBD_generator is created!')
    return CBD_generator


def create_CBD(opt):
    # Initialize the network
    CBD = network.CBD(opt)
    # Init the network
    network.weights_init(CBD, init_type = opt.init_type, init_gain = opt.init_gain)
    print('CBD is created!')
    return CBD

def create_MyCBD(opt):
    # Initialize the network
    MyCBD = network.MyCBD(opt)
    # Init the network
    network.weights_init(MyCBD, init_type = opt.init_type, init_gain = opt.init_gain)
    print('MyCBD is created!')
    return MyCBD

def create_MyWCNN(opt):
    # Initialize the network
    MyWCNN = network.MyWCNN(opt)
    # Init the network
    network.weights_init(MyWCNN, init_type = opt.init_type, init_gain = opt.init_gain)
    print('MyWCNN is created!')
    return MyWCNN

def create_MWCNN(opt):
    # Initialize the network
    MWCNN = network.MWCNN(opt)
    # Init the network
    network.weights_init(MWCNN, init_type = opt.init_type, init_gain = opt.init_gain)
    print('MWCNN is created!')
    return MWCNN

def create_DnCNN(opt):
    # Initialize the network
    DnCNN = network.DnCNN(opt)
    # Init the network
    network.weights_init(DnCNN, init_type = opt.init_type, init_gain = opt.init_gain)
    print('DnCNN is created!')
    return DnCNN

def create_MyDNN(opt):
    # Initialize the network
    MyDNN = network.MyDNN(opt)
    # Init the network
    network.weights_init(MyDNN, init_type = opt.init_type, init_gain = opt.init_gain)
    print('MyDNN is created!')
    return MyDNN

def create_FaceDNN(opt):
    # Initialize the network
    FaceDNN = network.FaceDNN(opt)
    # Init the network
    network.weights_init(FaceDNN, init_type = opt.init_type, init_gain = opt.init_gain)
    print('FaceDNN is created!')
    return FaceDNN

def create_UresNet(opt):
    # Initialize the network
    UresNet = network.UresNet(opt)
    # Init the network
    network.weights_init(UresNet, init_type = opt.init_type, init_gain = opt.init_gain)
    print('UresNet is created!')
    return UresNet

def create_MyCBD_MWCNN(opt):
    # Initialize the network
    MyCBD_MWCNN = network.MyCBD_MWCNN(opt)
    # Init the network
    network.weights_init(MyCBD_MWCNN, init_type = opt.init_type, init_gain = opt.init_gain)
    print('MyCBD_MWCNN is created!')
    return MyCBD_MWCNN

def create_MyCBD_MyWCNN(opt):
    # Initialize the network
    MyCBD_MyWCNN = network.MyCBD_MyWCNN(opt)
    # Init the network
    network.weights_init(MyCBD_MyWCNN, init_type = opt.init_type, init_gain = opt.init_gain)
    print('MyCBD_MyWCNN is created!')
    return MyCBD_MyWCNN

def create_MINCFeatureExtractor(opt):
    # Initialize the network
    MINCFeatureExtractor = network.MINCFeatureExtractor(opt)
    return MINCFeatureExtractor

def create_perceptualnet():
    # Initialize the network
    perceptualnet = network.PerceptualNet()
    vgg16 = tv.models.vgg16(pretrained = True)
    # Init the network
    load_dict(perceptualnet, vgg16)
    print('PerceptualNet is created!')
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(filespath) 
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def normalize_ImageNet_stats(batch):
    # adapt to the training style of VGG
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch_out = (batch - mean) / std
    return batch_out

def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 128 + 128
        # img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def save_sample_test(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 128 + 128
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(1, 2, 0).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 255):
    target = target * 128 + 128
    # target = target * 255
    # target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    # target = np.clip(target, 0, pixel_max_cnt)
    pred = pred * 128 + 128
    # pred = pred * 255
    # pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    # pred = np.clip(pred, 0, pixel_max_cnt)
    # mse = np.subtract(pred, target).mean()
    # p = 20 * np.log10(pixel_max_cnt / mse)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p