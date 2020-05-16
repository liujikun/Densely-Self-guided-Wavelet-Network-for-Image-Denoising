import torch
import torch.nn as nn
from network_module import *
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ----------------------------------------
#      Wavelet CBDNet  whole network
# ----------------------------------------
class Block_of_DMT1(nn.Module):
    def __init__(self):
        super(Block_of_DMT1,self).__init__()
 
        #DMT1
        self.conv1_1=nn.Conv2d(in_channels=160,out_channels=160,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(160, affine=True)
        self.relu1_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        return output 
 
class Block_of_DMT2(nn.Module):
    def __init__(self):
        super(Block_of_DMT2,self).__init__()
 
        #DMT1
        self.conv2_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn2_1=nn.BatchNorm2d(256, affine=True)
        self.relu2_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        return output 


 
class Block_of_DMT3(nn.Module):
    def __init__(self):
        super(Block_of_DMT3,self).__init__()
 
        #DMT1
        self.conv3_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_1=nn.BatchNorm2d(256, affine=True)
        self.relu3_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        return output 



class Block_of_DMT4(nn.Module):
    def __init__(self):
        super(Block_of_DMT4,self).__init__()
 
        #DMT1
        self.conv4_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn4_1=nn.BatchNorm2d(256, affine=True)
        self.relu4_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        return output


class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()
        # self.bn1 = nn.BatchNorm2d(int(channel_in/2.), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.bn2 = nn.BatchNorm2d(int(channel_in/2.), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.bn3 = nn.BatchNorm2d(channel_in, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        residual = x
        # out = self.relu1(self.bn1(self.conv_1(x)))
        # conc = torch.cat([x, out], 1)
        # out = self.relu2(self.bn2(self.conv_2(conc)))
        # conc = torch.cat([conc, out], 1)
        # out = self.relu3(self.bn3(self.conv_3(conc)))
        
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        return out



class MyDNN(nn.Module):
    def __init__(self, opt):
        super(MyDNN, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(in_channels = 3,  out_channels = 160, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'prelu', norm = opt.norm)
        self.E2 = Conv2dLayer(in_channels = 3*4, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'prelu',norm = opt.norm)
        self.E3 = Conv2dLayer(in_channels = 3*4*4, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'prelu',norm = opt.norm)
        self.E4 = Conv2dLayer(in_channels = 3*4*16, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'prelu',norm = opt.norm)
        # Bottle neck
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.blockDMT11 = self.make_layer(_DCR_block, 320)
        self.blockDMT12 = self.make_layer(_DCR_block, 320)
        self.blockDMT13 = self.make_layer(_DCR_block, 320)
        self.blockDMT14 = self.make_layer(_DCR_block, 320)
        self.blockDMT21 = self.make_layer(_DCR_block, 512)
        # self.blockDMT22 = self.make_layer(_DCR_block, 512)
        # self.blockDMT23 = self.make_layer(_DCR_block, 512)
        # self.blockDMT24 = self.make_layer(_DCR_block, 512)
        self.blockDMT31 = self.make_layer(_DCR_block, 512)
        # self.blockDMT32 = self.make_layer(_DCR_block, 512)
        # self.blockDMT33 = self.make_layer(_DCR_block, 512)
        # self.blockDMT34 = self.make_layer(_DCR_block, 512)
        self.blockDMT41 = self.make_layer(_DCR_block, 256)
        # self.blockDMT42 = self.make_layer(_DCR_block, 256)
        # self.blockDMT43 = self.make_layer(_DCR_block, 256)
        # self.blockDMT44 = self.make_layer(_DCR_block, 256)
        # self.DRB11 = ResidualDenseBlock_5C(nf=320, gc=64)
        # self.DRB12 = ResidualDenseBlock_5C(nf=320, gc=64)
        # self.DRB21 = ResidualDenseBlock_5C(nf=512, gc=64)
        # self.DRB31 = ResidualDenseBlock_5C(nf=512, gc=64)
        # self.DRB41 = ResidualDenseBlock_5C(nf=256, gc=64)
        # Decoder
        self.D1 = Conv2dLayer(in_channels = 256, out_channels = 1024, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'prelu', norm = opt.norm)
        self.D2 = Conv2dLayer(in_channels = 512, out_channels = 1024, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'prelu', norm = opt.norm)
        self.D3 = Conv2dLayer(in_channels = 512, out_channels = 640,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'prelu', norm = opt.norm)
        self.D4 = Conv2dLayer(in_channels = 320, out_channels = 3, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = opt.pad, norm = 'none', activation = 'none')
        self.D5 = Conv2dLayer(in_channels = 320, out_channels = 3, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = opt.pad, norm = 'none', activation = 'tanh')
        # channel shuffle
        self.S1 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = opt.pad, activation = 'none', norm = 'none')
        self.S2 = Conv2dLayer(in_channels = 512, out_channels = 512, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = opt.pad, activation = 'none', norm = 'none')
        self.S3 = Conv2dLayer(in_channels = 320, out_channels = 320, kernel_size=1, stride = 1, padding = 0, dilation = 1,  pad_type = opt.pad, activation = 'none', norm = 'none')
        self.S4 = Conv2dLayer(in_channels = 320, out_channels = 320, kernel_size=1, stride = 1, padding = 0, dilation = 1, groups=3*320, pad_type = opt.pad, activation = 'none', norm = 'none')
        # self.S5 = Conv2dLayer(in_channels = 3*320, out_channels = 320, kernel_size=1, stride = 1, padding = 0, dilation = 1, pad_type = opt.pad, activation = 'none', norm = 'none')

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh
    def forward(self, x):
        noisy = x
        E1 = self.E1(x)
        DMT1_yl,DMT1_yh = self.DWT(x)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        E2 = self.E2(DMT1)
        
        DMT2_yl, DMT2_yh = self.DWT(DMT1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        E3 = self.E3(DMT2)

        DMT3_yl, DMT3_yh = self.DWT(DMT2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        E4 = self.E4(DMT3)

        E4 = self.blockDMT41(E4)
        # E4_2 = E4_1 + E4
        # E4_3=self.blockDMT42(E4_2)
        # E4 = E4_2+E4_3

        # E4 = self.blockDMT43(E4)
        D1=self.D1(E4)
        D1=self._Itransformer(D1)
        IDMT4=self.IDWT(D1)
        D1=torch.cat((IDMT4, E3), 1)
        D1 = self.S1(D1)
        D2=self.blockDMT31(D1)
        # D2_2 = D1 + D2_1
        # D2_3=self.blockDMT32(D2_2)
        # D2 = D2_2+D2_3
        # D2=self.blockDMT33(D1)
        D2=self.D2(D2)

        D2=self._Itransformer(D2)
        IDMT3=self.IDWT(D2)
        D2=torch.cat((IDMT3, E2), 1)
        D2 = self.S2(D2)
        D3=self.blockDMT21(D2)
        # D3_2 = D3_1 + D2
        # D3_3=self.blockDMT22(D3_2)
        # D3 = D3_2+D3_3
        # D3=self.blockDMT23(D2)
        D3=self.D3(D3)

        D3=self._Itransformer(D3)
        IDMT2=self.IDWT(D3)
        D3=torch.cat((IDMT2, E1), 1)
        
        # res branch
        res_part1 = self.S3(D3)
        res_part2 = self.blockDMT11(res_part1)
        res_part3 = res_part2 + res_part1
        res_part4 = self.blockDMT12(res_part3)
        res_part5 = res_part4 + res_part3
        res_part6 = self.D4(res_part5)
        res_part = noisy - res_part6

        # # end2end branch
        e2e_part1 = self.S4(D3)
        e2e_part2 = self.blockDMT13(e2e_part1)
        e2e_part3 = e2e_part2 + e2e_part1
        e2e_part4 = self.blockDMT14(e2e_part3)
        e2e_part5 = e2e_part4 + e2e_part3
        e2e_part = self.D5(e2e_part5)

        x = (e2e_part + res_part)/2

        return x
