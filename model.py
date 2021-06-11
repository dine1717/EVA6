from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

# dropout_value = 0.05

class BatchNet(nn.Module):
    def __init__(self):
        super(BatchNet, self).__init__()
#####################################################################################################
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(8),
        ) # input size = 28 output_size = 26, in channels  = 1 out channels = 16

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
        ) # input_size = 26 , output_size = 24, 16 to 32 channels
#####################################################################################################

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # input size  = 24, output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # input sizez = 24, output_size = 12
#####################################################################################################

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            # nn.Dropout(dropout_value)
        ) # input size = 12, output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            # nn.Dropout(dropout_value)
        ) # input size = 10, output_size = 8
#####################################################################################################

        # adding a 1x1 kernel  block here to reduce paramters
        self.convblock5a = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # input size  = 8, output_size = 8
        self.pool1 = nn.MaxPool2d(2, 2) # input sizez = 8, output_size = 4
#####################################################################################################

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(15),
            # nn.Dropout(dropout_value)
        ) # input size  = 4,  output_size = 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(15),
            # nn.Dropout(0.03)
        ) # input size  = 4, output_size = 4
#####################################################################################################        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1 channels  = 16

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock5a(x)
        x = self.pool1(x)        
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class LayerNet(nn.Module):
    def __init__(self):
        super(LayerNet, self).__init__()
#####################################################################################################
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.GroupNorm(1,8),
        ) # input size = 28 output_size = 26, in channels  = 1 out channels = 16
        # self.convblock1a = nn.LayerNorm(self.convblock1.size()[1:], elementwise_affine=True)

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,12),
        ) # input_size = 26 , output_size = 24, 16 to 32 channels
#####################################################################################################

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # input size  = 24, output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # input sizez = 24, output_size = 12
#####################################################################################################

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(1,12),
            # nn.Dropout(dropout_value)
        ) # input size = 12, output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(1,16),
            # nn.Dropout(dropout_value)
        ) # input size = 10, output_size = 8
#####################################################################################################

        # adding a 1x1 kernel  block here to reduce paramters
        self.convblock5a = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # input size  = 8, output_size = 8
        self.pool1 = nn.MaxPool2d(2, 2) # input sizez = 8, output_size = 4
#####################################################################################################

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(1,15),
            # nn.Dropout(dropout_value)
        ) # input size  = 4,  output_size = 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(1,15),
            # nn.Dropout(0.03)
        ) # input size  = 4, output_size = 4
#####################################################################################################        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1 channels  = 16

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1(x)
        # x = self.convblock1a(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock5a(x)
        x = self.pool1(x)        
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class GroupNet(nn.Module):
    def __init__(self):
        super(GroupNet, self).__init__()
#####################################################################################################
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.GroupNorm(2,8),
        ) # input size = 28 output_size = 26, in channels  = 1 out channels = 16
        # self.convblock1a = nn.LayerNorm(self.convblock1.size()[1:], elementwise_affine=True)

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2,12),
        ) # input_size = 26 , output_size = 24, 16 to 32 channels
#####################################################################################################

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # input size  = 24, output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # input sizez = 24, output_size = 12
#####################################################################################################

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(2,12),
            # nn.Dropout(dropout_value)
        ) # input size = 12, output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(2,16),
            # nn.Dropout(dropout_value)
        ) # input size = 10, output_size = 8
#####################################################################################################

        # adding a 1x1 kernel  block here to reduce paramters
        self.convblock5a = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # input size  = 8, output_size = 8
        self.pool1 = nn.MaxPool2d(2, 2) # input sizez = 8, output_size = 4
#####################################################################################################

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(3,15),
            # nn.Dropout(dropout_value)
        ) # input size  = 4,  output_size = 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(3,15),
            # nn.Dropout(0.03)
        ) # input size  = 4, output_size = 4
#####################################################################################################        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1 channels  = 16

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1(x)
        # x = self.convblock1a(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock5a(x)
        x = self.pool1(x)        
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)