import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools
from math import ceil,floor

#UNET paper submodules
class ConvBlock(nn.Module):
    def __init__(self, input_channels,mid_channels, out_channels, BN= False ):
        super(ConvBlock,self).__init__()

        self.conv1= nn.Conv2d(input_channels,mid_channels,3,1,1)
        self.activation1= nn.ReLU()
        self.conv2= nn.Conv2d(mid_channels,out_channels,3,1,1)
        self.activation2= nn.ReLU()
        self.bn = BN
        if(self.bn):
            self.bn1=nn.BatchNorm2d(mid_channels)
            self.bn2=nn.BatchNorm2d(out_channels)


    def forward(self, x):

        if (self.bn):
            x=self.activation1( self.bn1(self.conv1(x))  )
            x=self.activation2(  self.bn2(self.conv2(x))  )
        else:
            x=self.activation1((self.conv1(x))  )
            x=self.activation2(  (self.conv2(x))  )
        return x
class Down(nn.Module):
    def __init__(self):
        super(Down,self).__init__()
        self.downsample= nn.MaxPool2d(2)
    def forward(self, x):
        return self.downsample(x)
class Up(nn.Module):
    def __init__(self, input_channels,output_channels, scale=2):
        super(Up,self).__init__()
        self.upsample= nn.Upsample(scale_factor=scale,mode='nearest')
        self.pad= nn.ZeroPad2d((1,0,1,0))
        self.conv = nn.Conv2d(input_channels,output_channels,2)
    def forward(self, x):
        x= self.conv(self.pad(self.upsample(x)))
        return x
class Merge(nn.Module):
    def __init__(self):
        super(Merge,self).__init__()
    def forward(self, x1,x2):
        return torch.cat((x1,x2),1)

#Paper implementation of UNET with a sigmoid activation layer at the end
class UNet(nn.Module):
    def __init__(self,input_channels,output_channels, BN= False):
        super(UNet,self).__init__()
        self.block1A = ConvBlock(input_channels,64,64,BN)
        self.down1=Down()
        self.block2A = ConvBlock(64,128,128,BN)
        self.down2=Down()
        self.block3A = ConvBlock(128,256,256,BN)
        self.down3=Down()
        self.block4A = ConvBlock(256,512,512,BN)
        self.down4=Down()
        self.blockMid= ConvBlock(512,1024,1024,BN)
        self.up4=Up(1024,512)
        self.merge4=Merge()
        self.block4B=ConvBlock(1024,512,512,BN)
        self.up3=Up(512,256)
        self.merge3=Merge()
        self.block3B=ConvBlock(512,256,256,BN)
        self.up2=Up(256,128)
        self.merge2=Merge()
        self.block2B=ConvBlock(256,128,128,BN)
        self.up1=Up(128,64)
        self.merge1=Merge()
        self.block1B=ConvBlock(128,64,64,BN)
        self.finalConv=nn.Conv2d(64,output_channels,1)
        self.activation=nn.Sigmoid()
    def forward(self, x):

        shorcut1=self.block1A(x)
        x=self.down1(shorcut1)

        shorcut2=self.block2A(x)
        x=self.down2(shorcut2)

        shorcut3=self.block3A(x)
        x=self.down3(shorcut3)

        shorcut4=self.block4A(x)
        x=self.down4(shorcut4)

        x=self.blockMid(x)

        x=self.up4(x)
        x=self.merge4(x,shorcut4)
        x=self.block4B(x)

        x=self.up3(x)
        x=self.merge3(x,shorcut3)
        x=self.block3B(x)

        x=self.up2(x)
        x=self.merge2(x,shorcut2)
        x=self.block2B(x)

        x=self.up1(x)
        x=self.merge1(x,shorcut1)
        x=self.block1B(x)

        x=self.finalConv(x)
        x=self.activation(x)

        return x


#Added 2nd level skip connections with upsampling
class Up_UNet(nn.Module):
    def __init__(self,input_channels,output_channels,BN=False):
        super(Up_UNet,self).__init__()
        self.block1A = ConvBlock(input_channels,64,64,BN)
        self.down1=Down()
        self.block2A = ConvBlock(64,128,128,BN)
        self.down2=Down()
        self.block3A = ConvBlock(128,256,256,BN)
        self.down3=Down()
        self.block4A = ConvBlock(256,512,512,BN)
        self.down4=Down()
        self.blockMid= ConvBlock(512,1024,1024,BN)
        self.up4=Up(1024,512)
        self.merge4=Merge()
        self.block4B=ConvBlock(1024,512,512,BN)
        self.up3=Up(512,256)
        self.up3b=Up(512,256)
        self.merge3=Merge()
        self.block3B=ConvBlock(768,256,256,BN)
        self.up2=Up(256,128)
        self.up2b=Up(256,128)
        self.merge2=Merge()
        self.block2B=ConvBlock(384,128,128,BN)
        self.up1=Up(128,64)
        self.up1b=Up(128,64)
        self.merge1=Merge()
        self.block1B=ConvBlock(192,64,64,BN)
        self.finalConv=nn.Conv2d(64,output_channels,1)
        self.activation=nn.Sigmoid()
    def forward(self, x):

        shorcut1=self.block1A(x)
        x=self.down1(shorcut1)

        shorcut2=self.block2A(x)
        x=self.down2(shorcut2)

        shorcut3=self.block3A(x)
        x=self.down3(shorcut3)

        shorcut4=self.block4A(x)
        x=self.down4(shorcut4)

        x=self.blockMid(x)

        x=self.up4(x)
        x=self.merge4(x,shorcut4)
        x=self.block4B(x)

        x=self.up3(x)

        x2=self.up3b(shorcut4)

        x=self.merge3(x,shorcut3)
        x=self.merge3(x,x2)
        x=self.block3B(x)

        x=self.up2(x)
        x=self.merge2(x,shorcut2)
        x=self.merge2(x,self.up2b(shorcut3))
        x=self.block2B(x)

        x=self.up1(x)
        x=self.merge1(x,shorcut1)
        x=self.merge1(x,self.up1b(shorcut2))
        x=self.block1B(x)

        x=self.finalConv(x)
        x=self.activation(x)

        return x
        
 

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsampling=1, activation = nn.ReLU, normalization = nn.BatchNorm2d, padding= 'zeros'):
        super(ResConvBlock,self).__init__()
        #padding setup
        p = (kernel_size - downsampling) /2
        pad = ( int(ceil(p)),int( floor(p) ) , int( ceil(p) ) , int(floor(p) ) )
        if(padding == 'zeros'):
            self.padding = nn.ZeroPad2d(pad )
        elif(padding == 'reflection'):
            self.padding = nn.ReflectionPad2d(pad)
        else:
            self.padding = None
        #normalization setup
        if(normalization ==None):
            self.normalization = None
        else:
            self.normalization = normalization(out_channels)
        #activation setup
        if(activation ==None):
            self.activation = None
        else:
            self.activation = activation()  
        #convolution setup
        self.convolution= nn.Conv2d(in_channels,out_channels,kernel_size,downsampling)
    def forward(self,x):
        if(self.padding):
            x= self.padding(x)
        x= self.convolution(x)
        if(self.normalization):
            x=self.normalization(x)
        if(self.activation):
            x=self.activation(x)
        return x  




class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=2, activation = nn.ReLU, normalization = nn.BatchNorm2d):
        super(DeconvBlock,self).__init__()
        #padding setup
        if(kernel_size - upsampling)%2 ==0:
            p2=0
            p = (kernel_size - upsampling ) //2
        else:
            p2=1
            p = (kernel_size - upsampling +1 ) //2
        
        #normalization setup
        if(normalization ==None):
            self.normalization = None
        else:  
            self.normalization = normalization(out_channels)
        #activation setup
        if(activation ==None):
            self.activation = None
        else:
            self.activation = activation()  
        #convolution setup
        self.deconvolution = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,upsampling,p,p2)
    def forward(self,x):

        x= self.deconvolution(x)
        if(self.normalization):
            x=self.normalization(x)
        if(self.activation):
            x=self.activation(x)
        return x   



