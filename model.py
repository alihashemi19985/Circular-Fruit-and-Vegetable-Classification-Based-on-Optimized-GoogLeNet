import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from Swish import Swish
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
class Inception(nn.Module):
    
    def __init__(self, in_channels=3, use_auxiliary=True, num_classes=33):
        super(Inception, self).__init__()   
        
        self.swish = Swish()
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(3376, num_classes)
        
        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            self.auxiliary4a = Auxiliary(992, num_classes)
            self.auxiliary4d = Auxiliary(1872, num_classes)
        
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(384, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(672, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(992, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(1296, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(1584, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(1872, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(2320, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(2768, 384, 192, 384, 48, 128, 128)

    def forward(self, x):
        y = None
        z = None
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        
        x = self.inception3a(x) 
         
        x = self.inception3b(x)
        
        x = self.maxpool(x)
         
        x = self.inception4a(x)
        
        if self.training and self.use_auxiliary:
            y = self.auxiliary4a(x)
        
        x = self.inception4b(x)
        
        x = self.inception4c(x)
        
        x = self.inception4d(x)
        
        if self.training and self.use_auxiliary:
            z = self.auxiliary4d(x)
         
        x = self.inception4e(x)
        
        x = self.maxpool(x)
        
        x = self.inception5a(x)
        
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        
        x = self.linear(x)
        
        return x, y, z

    
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = Swish()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    

class InceptionBlock(nn.Module):
    
    def __init__(self, 
                 im_channels, 
                 num_1x1, 
                 num_3x3_red, 
                 num_3x3, 
                 num_5x5_red, 
                 num_5x5, 
                 num_pool_proj):
        
        super(InceptionBlock, self).__init__()
        
        self.one_by_one = ConvBlock(im_channels, num_1x1, kernel_size=1)
        
        self.tree_by_three_red = ConvBlock(im_channels, num_3x3_red, kernel_size=1)  
        #self.tree_by_three = ConvBlock(num_3x3_red, num_3x3, kernel_size=3, padding=1)
        
        #self.five_by_five_red = ConvBlock(im_channels, num_5x5_red, kernel_size=1)
        #self.five_by_five = ConvBlock(num_5x5_red, num_5x5, kernel_size=5, padding=2)
            
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.pool_proj = ConvBlock(im_channels, num_pool_proj, kernel_size=1)
         
        self.Densblock = DenseBlock(im_channels)    
        #self.transition = TransitionLayer(im_channels * 2)    
    def forward(self, x):
        x1 = self.one_by_one(x)
        
        x2 = self.tree_by_three_red(x)
        #x2 = self.tree_by_three(x2)
        
        #x3 = self.five_by_five_red(x)
        #x3 = self.five_by_five(x3)
        
        #x4 = self.maxpool(x)
        #x4 = self.pool_proj(x4)
        
        x5 = self.Densblock(x)
        #x5 = self.transition(x)
        x = torch.cat([x1, x2,x5], 1)
        return x 
    

class Auxiliary(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(Auxiliary, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = ConvBlock(in_channels, 128, kernel_size=1)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(0.7)
        self.relu = Swish()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1x1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        return x 
class DenseBlock(nn.Module):
    def __init__(self, input_channel):
        super(DenseBlock, self).__init__()
        
        # Batch normalization and ReLU before each convolution
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.conv1 = nn.Conv2d(input_channel,  32, kernel_size=1, bias=False)
        self.swish = Swish()
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d( 32, 32, kernel_size=1, bias=False)
    def forward(self, x):
        out = self.swish(self.bn1(x))
        out = self.conv1(out)
        out = self.swish(self.bn2(out))
        out = self.conv2(out)
        out = torch.cat((x, out), 1)  # Concatenate input and outpu       
        return out    
    
