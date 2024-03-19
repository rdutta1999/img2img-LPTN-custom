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
    
def laplacian_pyramid_torch(img, height):
    laplacian_pyr=[]
    current=img.clone()
    gaussian_kernal=(1/273) * torch.Tensor([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4], [1,4,7,4,1]]).repeat(current.shape[1],1,1,1)
    for i in range(height):        
        
        filtered= F.conv2d(current,gaussian_kernal, padding=2, groups=current.shape[1])
        downsampled= filtered[:,:,::2,::2]
        output = F.upsample(downsampled, scale_factor=2, mode='bilinear')
        print(current.shape)
        print(downsampled.shape)
        print(output.shape)
        diff= current-output
        laplacian_pyr.append(diff)
        current=downsampled
    return laplacian_pyr    

def laplacian_pyramid(gpA,height):
    laplacian_pyr = [gpA[-1]]
    for i in range(height,0,-1):
        size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gpA[i], dstsize=size)
        laplacian = cv2.subtract(gpA[i-1],gaussian_expanded)
        laplacian_pyr.append(laplacian)
        cv2.imwrite(f"image{i}.png", laplacian)   
    return laplacian_pyr

def reconstruct_image(lbImage, height):
    ls_ = lbImage[height-1]     
    for j in range(height-2,0,-1):
        size = (lbImage[j].shape[0], lbImage[j].shape[1])
        print(size)
        print(lbImage[j].shape)
        print(ls_.shape)
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_,lbImage[j])
    # ls_ = cv2.cvtColor(ls_, cv2.COLOR_YCR_CB2BGR)                    
    return ls_
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
image=img[0:-1,:,:]
image=np.float32(np.transpose(image,(2,0,1)))
inputs=torch.tensor(np.array([image]))
# frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# height = 4

# inputs=torch.randn(2,3,256,256)* 255
# inputs.cuda()
# print(inputs.shape)
lpA=laplacian_pyramid_torch(inputs,3)

lpA=[l.numpy().transpose(2,3,1,0).squeeze() for l in lpA]
print([l for l in lpA])
# gpA=gaussian_pyramid(frame,height)
# lpA=laplacian_pyramid(gpA, height)

img= reconstruct_image(lpA, 3)

cv2.imwrite("torch_reconstruction.png",img)

# cv2.imwrite("Pyramid_2.jpeg", img)
# display_images(lpA)
# cv2.waitKey(0)
