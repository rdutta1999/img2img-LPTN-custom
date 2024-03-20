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
        self.depth=4
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
        self.high_frequency_layers=nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16,kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16),
            nn.Conv2d(in_channels=16, out_channels=1,kernel_size=3, stride=1,padding=1),
        )
        self.other_freq_1_layers=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=1, kernel_size=3,stride=1, padding=1),
        )
        self.other_freq_2_layers=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=1, kernel_size=3,stride=1, padding=1),
        )
        
    def forward(self, x):
        pyramid=laplacian_pyramid(x,self.depth) 
        low_freq_component=pyramid[-1]
        low_freq_output=self.low_frequency_layers(low_freq_component)
        low_freq_output=low_freq_output+low_freq_component
        low_freq_output=F.tanh(low_freq_output)
        high_freq_component=pyramid[-2]
        low_freq_component_upsampled=F.interpolate(low_freq_component, size=(high_freq_component.shape[2],high_freq_component.shape[3]), mode="bilinear")
        low_freq_output_upsampled=F.interpolate(low_freq_output, size=(high_freq_component.shape[2],high_freq_component.shape[3]), mode="bilinear")
        high_freq_input= torch.concat((high_freq_component,low_freq_component_upsampled,low_freq_output_upsampled), dim=1)
        # print(high_freq_input.shape)
        mask= self.high_frequency_layers(high_freq_input)
        high_freq_output=high_freq_component*mask
        other_freq_component_1=pyramid[-3]
        other_freq_component_2=pyramid[-4]
        mask_upsampled=F.interpolate(mask, size=(other_freq_component_1.shape[2], other_freq_component_1.shape[3]), mode="bilinear")
        mask_upsampled_output=self.other_freq_1_layers(mask_upsampled)
        other_freq_component_1_output=mask_upsampled_output*other_freq_component_1
        mask_upsampled_2=F.interpolate(mask_upsampled_output, size=(other_freq_component_2.shape[2], other_freq_component_2.shape[3]), mode="bilinear")
        mask_upsampled_2_output= self.other_freq_2_layers(mask_upsampled_2)
        other_freq_component_2_output=mask_upsampled_2_output*other_freq_component_2
        
        return [other_freq_component_2_output, other_freq_component_1_output, high_freq_output, low_freq_output]
net=LPTN_Network()

img=cv2.imread("image.png")
    
image=img

image=np.float32(np.transpose(image,(2,0,1)))
inp=torch.tensor(np.array([image]))
# inp=torch.rand((1,3,100,300))
translated_pyr=net(inp)

img=reconstruct_image(translated_pyr, net.depth)

cv2.imwrite("LPTN_Output.png", img.detach().numpy().transpose(2,3,1,0).squeeze())