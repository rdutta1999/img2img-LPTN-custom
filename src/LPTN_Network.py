import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from Utils import laplacian_pyramid, reconstruct_image

# Define residual block as stated in paper, residual block has two conv layers with leaky relus in between and allows for the input to factor into the output
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(features, features,3,stride = 1,padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(features, features, 3, stride = 1,padding = 1),
        )
    # Pass through layers and add input to the output
    def forward(self,x):
        return x + self.layers(x)

#Module for network
class LPTN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Assign the pyramid construction method and the reconstruction method to the object
        self.construct_pyramid = laplacian_pyramid
        self.reconstruct_image = reconstruct_image
        
        # 4 layers in pyramid (including last layer)
        self.depth = 4

        # The deepest layers in the network corresponding to the lower frequencies
        self.low_frequency_layers = nn.Sequential(
            nn.Conv2d(3,16,3,padding = 1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 64,kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(in_channels = 64, out_channels = 16,kernel_size = 3, stride = 1,padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 3,kernel_size = 3, stride = 1, padding = 1),
    )
        # High frequency layers (as defined in the paper, should probably have a better name)
        self.high_frequency_layers = nn.Sequential(
            nn.Conv2d(in_channels = 9, out_channels = 64,kernel_size = 3, stride = 1,padding = 1),
            nn.LeakyReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(in_channels = 64, out_channels = 3,kernel_size = 3, stride = 1,padding = 1),
        )
        self.other_freq_layer_0 = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 1)
        )
        # Shallow layers for second highest frequencies (maybe a better name here?)
        self.other_freq_1_layers = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 1),
        )
        # Shallow layers for highest frequencies
        self.other_freq_2_layers = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 1),
        )

    def forward(self, x):
        #Construct pyramid from input, x is a tensor of dimensions (N, C, W, H), typically C will be 3 as we're working with RGB images
        
        pyramid = laplacian_pyramid(x, self.depth, next(self.parameters()).device) 
        # Extract bottom layer of the pyramid
        low_freq_component = pyramid[-1]
        # Pass through the low frequency layers
        low_freq_output = self.low_frequency_layers(low_freq_component)
        
        # Following the paper closely here, variable names could use some work as earlier
        low_freq_output = low_freq_output + low_freq_component
        low_freq_output = torch.tanh(low_freq_output)
        # torchvision.utils.save_image(low_freq_output, "Our_Result_1Level_Output.png")
        # input()
        # High frequency component
        high_freq_component = pyramid[-2]
        # Using interpolate from torch to upsample, not sure if that is the best method?
        low_freq_component_upsampled = F.interpolate(low_freq_component, size = (high_freq_component.shape[2],high_freq_component.shape[3]), mode = "bilinear")
        low_freq_output_upsampled = F.interpolate(low_freq_output, size = (high_freq_component.shape[2],high_freq_component.shape[3]), mode = "bilinear")
        high_freq_input = torch.concat((high_freq_component,low_freq_component_upsampled,low_freq_output_upsampled), dim = 1)
        mask = self.high_frequency_layers(high_freq_input)
        high_freq_output = high_freq_component * mask + high_freq_component
        high_freq_output = self.other_freq_layer_0(high_freq_output)
        other_freq_component_1 = pyramid[-3]
        other_freq_component_2 = pyramid[-4]
        mask = F.interpolate(mask, size = (other_freq_component_1.shape[2], other_freq_component_1.shape[3]), mode = "bilinear")
        other_freq_component_1_output = other_freq_component_1 * mask + other_freq_component_1
        # mask_upsampled_output=self.other_freq_1_layers(mask_upsampled)
        other_freq_component_1_output = self.other_freq_1_layers(other_freq_component_1_output)
        mask = F.interpolate(mask, size = (other_freq_component_2.shape[2], other_freq_component_2.shape[3]), mode = "bilinear")
        other_freq_component_2_output = other_freq_component_2 * mask + other_freq_component_2
        other_freq_component_2_output = self.other_freq_2_layers(other_freq_component_2_output)
        # Return pyramid of components in order from largest to smallest to allow for reconstruction, should we return reconstruction as well? Depends on loss implementation
        # return mask_upsampled_2_output
        return reconstruct_image([other_freq_component_2_output, other_freq_component_1_output, high_freq_output, low_freq_output], self.depth)