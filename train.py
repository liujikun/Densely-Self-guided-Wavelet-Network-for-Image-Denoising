import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train ot not') # for second stage, change it to False
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 25, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_best_model', type = bool, default = True, help = 'save best model ot not')
    parser.add_argument('--save_by_iter', type = int, default = 10000000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--save_name_mode', type = bool, default = True, help = 'True for concise name, and False for exhaustive name')
    parser.add_argument('--load_name', type = str, default = 'Pre_model', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 500, help = 'number of epochs of training') # for second stage, change it to 30
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'crop size')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0004, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--b3', type = float, default = 0.9, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 15, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 1600, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.9, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_tv', type = float, default = 0.5, help = 'coefficient for TV Loss')
    parser.add_argument('--lambda_percep', type = float, default = 1, help = 'coefficient for Perceptual Loss')
    parser.add_argument('--lambda_gan', type = float, default = 0.01, help = 'coefficient for GAN Loss')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--sal_channels', type = int, default = 1, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # GAN parameters
    parser.add_argument('--gan_mode', type = str, default = 'LSGAN', help = 'type of GAN: [LSGAN | WGAN], WGAN is recommended')
    parser.add_argument('--additional_training_d', type = int, default = 1, help = 'number of training D more times than G')
    # Dataset parameters
    parser.add_argument('--task', type = str, default = 'denoise', help = 'the specific task of the system')
    parser.add_argument('--angle_aug', type = bool, default = True, help = 'data augmentation')
    parser.add_argument('--in_root', type = str, default = './name_list.txt', help = 'color image baseroot')
    parser.add_argument('--val_root', type = str, default = './val_gt.txt', help = 'color image baseroot')
    parser.add_argument('--save_path', type = str, default = './output', help = 'color image baseroot')
    parser.add_argument('--sample_path', type = str, default = './sample', help = 'color image baseroot')
    parser.add_argument('--test_root', type = str, default = './val_gt.txt', help = 'color image baseroot')
    parser.add_argument('--baseroot', type = str, default = '/data/MyESRGAN/face_512', help = 'Face images set path')
    parser.add_argument('--use_blur', type=bool, default=True)
    parser.add_argument('--d_Down', type=list, default=[0,3])
    parser.add_argument('--d_Noise', type=list, default=[0,4])
    parser.add_argument('--d_Blur', type=list, default=[0,8])
    parser.add_argument('--d_Defocus', type=list, default=[3,8])
    parser.add_argument('--d_Motion', type=list, default=[3,6])
    parser.add_argument('--d_fix_distortion', type=int, default=0)
    parser.add_argument('--low_light', type=int, default=0)
    parser.add_argument('--d_Beautify', type=bool, default=False)
    opt = parser.parse_args()

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    
    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    if opt.pre_train:
        print('MyDNN-training settings: [Epochs: %d] [Batch size: %d] [Learning rate: %.4f] [Saving mode: %s]'
            % (opt.epochs, opt.batch_size, opt.lr_g, opt.save_mode))
        trainer.MyDNN(opt)
    else:
        print('NoiseEstimator-training settings: [Epochs: %d] [Batch size: %d] [Learning rate: %.4f] [Saving mode: %s]'
            % (opt.epochs, opt.batch_size, opt.lr_g, opt.save_mode))
        trainer.NoiseEstimator(opt)   
