import torch
import torch.nn as nn

class DoubleConv2D(nn.Module):
    """
    Two 3x3 convolutuions layers used throughout the 2D U-Net
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialise the double convolution block
        """
        
        super().__init__()
        
        self.block = nn.Sequential(
            #learn spatial features from the input feature maps
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            
            #normalise feature maps to improve training stability
            nn.BatchNorm2d(out_channels),
            
            #add non linearity so the network can learn complex features
            nn.ReLU(inplace = True), 
            
            #Second convolution further refines the learned features
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding= 1),

            #normalise the second convolution output
            nn.BatchNorm2d(out_channels),
            
            #second activation
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        """
        Pass the input through the double convoltion block
        """
    
class UNet2D(nn.Module):
    """
    2D UNet baseline adapted for HipMRI segmentation
    Based on the UNet architecture proposed by Ronneberger, Fischer and Brox (2015)
    """ 
    def __init__(self, in_channels = 1, num_classes = 6):
        """
        Initialise the 2D UNet architecture
        """
        super().__init__()
        
        #Encoders
        self.enc1 = DoubleConv2D(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv2D(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv2D(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv2D(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        #Bottleneck
        self.bottleneck = DoubleConv2D(256, 512)
        
        #decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.dec4 = DoubleConv2D(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.dec3 = DoubleConv2D(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.dec2 = DoubleConv2D(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
        self.dec1 = DoubleConv2D(64, 32)
        
        #Output layer
        self.output = nn.Conv2d(32, num_classes, kernel_size = 1)
    
    def forward(self, x):
        """
        Perform the forward pass through the encoder, bottleneck and decoder
        """
        #Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(x)
        e3 = self.enc3(x)
        e4 = self.enc4(x)
        
        #Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        #Decoder with skip connection
        d4 = self.up4
        d4 = torch.cat((d4, e4), dim = 1)
        d4 = self.dec4(d4)
        
        d3 = self.up3
        d3 = torch.cat((d3, e3), dim = 1)
        d3 = self.dec3(d3)
        
        d2 = self.up2
        d2 = torch.cat((d2, e2), dim = 1)
        d2 = self.dec2(d2)
        
        d1 = self.up1
        d1 = torch.cat((d1, e1), dim = 1)
        d1 = self.dec1(d1)
        
        #Segment 
        return self.output(d1)
        
        
        
    