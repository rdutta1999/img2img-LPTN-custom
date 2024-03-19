import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
from Utils import laplacian_pyramid, reconstruct_image
import cv2
import numpy as np
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(features, features, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU()            
        )
    def forward(self,x):
        return x+self.layers(x)
class LPTN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.construct_pyramid=laplacian_pyramid
        self.reconstruct_image=reconstruct_image
        self.depth=3
        self.low_frequency_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=64,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(in_channels=64, out_channels=16,kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=3,kernel_size=3, stride=1, padding=1),
    )
    def forward(self, x):
        pyramid=laplacian_pyramid(x,self.depth)
        low_freq_component=pyramid[-1]
        low_freq_output=self.low_frequency_layers(low_freq_component)
        low_freq_output=low_freq_output+low_freq_component
        return F.tanh(low_freq_output)
            
net=LPTN_Network()

img=cv2.imread("image.png")
    
image=img

image=np.float32(np.transpose(image,(2,0,1)))
inp=torch.tensor(np.array([image]))
# inp=torch.rand((1,3,100,300))
print(net(inp))

