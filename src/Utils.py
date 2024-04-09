import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from  scipy.ndimage import gaussian_filter

def gaussian_pyramid(frame, height):
    Gauss = frame.copy()
    gpA = [Gauss]
    for i in range(height):
        Gauss = cv2.pyrDown(Gauss)
        gpA.append(Gauss)
    return gpA

def laplacian_pyramid(img, levels, device):
    pyramid = []
    
    current_level = img.clone()
    
    gaussian_kernal=(1/273) * torch.Tensor([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4], [1,4,7,4,1]]).repeat(current_level.shape[1],1,1,1).to(device=device)
    
    for i in range(levels - 1):
        filtered= F.conv2d(current_level,gaussian_kernal, padding=2, groups=current_level.shape[1])
        # down = F.avg_pool2d(filtered, kernel_size=2, stride=2)
        down = filtered[:,:,::2,::2]
        
        up = F.interpolate(down, scale_factor=2, mode='bilinear', align_corners=True)
        if up.shape[2] != current_level.shape[2] or up.shape[3] != current_level.shape[3]:
            up = F.interpolate(up, size=(current_level.shape[2], current_level.shape[3]))        
        residual = current_level - up
        pyramid.append(residual)
        current_level = down
    pyramid.append(current_level)
    return pyramid

def reconstruct_image(lpA,height):
    img=lpA[-1]
    for j in range(height-2,-1,-1):
        img=F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
        if img.shape[2] != lpA[j].shape[2] or img.shape[3] != lpA[j].shape[3]:
            img = F.interpolate(img, size=(lpA[j].shape[2], lpA[j].shape[3]))        
        
        img=lpA[j]+img
    return img
    
def display_images(images, titles=None, cols=3, figsize=(15, 10)):
    rows = len(images) // cols + (0 if len(images) % cols == 0 else 1)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i])
    
    for ax in axes[len(images):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("Pyramid.png")