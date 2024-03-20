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

# def upsample_image(img, scale):
    
# def laplacian_pyramid_torch(img, height):
#     laplacian_pyr=[]
#     current=img.clone()
#     gaussian_kernal=(1/273) * torch.Tensor([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4], [1,4,7,4,1]]).repeat(current.shape[1],1,1,1)
#     for i in range(height):        
        
#         filtered= F.conv2d(current,gaussian_kernal, padding=2, groups=current.shape[1])
#         downsampled= filtered[:,:,::2,::2]
#         output = F.upsample(downsampled, scale_factor=2, mode='bilinear')
#         if(output.shape[2]!=current.shape[2]):
#             output=output[:,:,0:-1,:]
#         if(output.shape[3]!=current.shape[3]):
#             output=output[:,:,:,0:-1]
        
#         print(current.shape)
#         print(downsampled.shape)
#         print(output.shape)
#         diff= current-output
#         laplacian_pyr.append(diff)
#         current=downsampled
#     return laplacian_pyr    
def laplacian_pyramid(img, levels):
    pyramid = []
    
    current_level = img.clone()
    
    gaussian_kernal=(1/273) * torch.Tensor([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4], [1,4,7,4,1]]).repeat(current_level.shape[1],1,1,1)
    
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

# def laplacian_pyramid(gpA,height):
#     laplacian_pyr = [gpA[-1]]
#     for i in range(height,0,-1):
#         size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
#         gaussian_expanded = cv2.pyrUp(gpA[i], dstsize=size)
#         laplacian = cv2.subtract(gpA[i-1],gaussian_expanded)
#         laplacian_pyr.append(laplacian)
#         cv2.imwrite(f"image{i}.png", laplacian)   
#     return laplacian_pyr

def reconstruct_image(lpA,height):
    img=lpA[-1]
    for j in range(height-2,-1,-1):
        img=F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
        if img.shape[2] != lpA[j].shape[2] or img.shape[3] != lpA[j].shape[3]:
            img = F.interpolate(img, size=(lpA[j].shape[2], lpA[j].shape[3]))        
        
        img=lpA[j]+img
    return img
        
        
# def reconstruct_image(lbImage, height):
#     ls_ = lbImage[height-1]     
#     for j in range(height-2,0,-1):
#         size = (lbImage[j].shape[0], lbImage[j].shape[1])
#         ls_ = cv2.pyrUp(ls_)
#         ls_ = cv2.add(ls_,lbImage[j])
#     # ls_ = cv2.cvtColor(ls_, cv2.COLOR_YCR_CB2BGR)                    
#     return ls_

# def reconstruct_image_torch(lpA,height):
#     ls_ = lbImage[height-1]     
#     for j in range(height-2,0,-1):
#         size = (lbImage[j].shape[0], lbImage[j].shape[1])
#         print(size)
#         print(lbImage[j].shape)
#         print(ls_.shape)
#         ls_ = cv2.pyrUp(ls_)
#         ls_ = cv2.add(ls_,lbImage[j])
    
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

img=cv2.imread("image.png")
    
image=img

image=np.float32(np.transpose(image,(2,0,1)))
inputs=torch.tensor(np.array([image]))
# frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# height = 4

# inputs=torch.randn(2,3,256,256)* 255
# inputs.cuda()
# print(inputs.shape)
lpA=laplacian_pyramid(inputs,5)
# lpA=[l.numpy().transpose(2,3,1,0).squeeze() for l in lpA]
# print([l for l in lpA])
# # gpA=gaussian_pyramid(frame,height)
# # lpA=laplacian_pyramid(gpA, height)

img2= reconstruct_image(lpA, 5)

img2=img2.numpy().transpose(2,3,1,0).squeeze()
cv2.imwrite("torch_reconstruction.png",img2)

# cv2.imwrite("Pyramid_2.jpeg", img)
# display_images(lpA)
# cv2.waitKey(0)
