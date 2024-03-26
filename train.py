import os
import cv2
import torch
import math
import random
import platform
import numpy as np
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import U2NET
from pathlib import Path
from natsort import natsorted
# from tqdm.notebook import tqdm
from tqdm import tqdm
from PIL import Image, ImageOps
from torch.autograd import Variable
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

def resize_to_inputsz(x, **kwargs):
    
    # For RGB images with 3 channels:
    if x.shape[-1] == 3:        
        y = Image.fromarray(x)
        y = y.resize(INPUT_SZ, resample = Image.LANCZOS)
        y = np.array(y)
        
    #For Masks with 1 channel:
    if x.shape[-1] == 1:
        x = x.squeeze()
        y = Image.fromarray(x)
        y = y.resize(INPUT_SZ, resample = Image.LANCZOS)
        y = np.array(y)
        y = np.expand_dims(y, -1)
            
    return y

def scale_0_1(x, **kwargs):
    return x / 255.

def augment_normalize(doAugment = True, doNormalize = True, doTensored = True):
    transform = []
    
    if doAugment:
        doCropTransform = True
        doGeometricTransform = True
        doVisualTransform = True       
        
        if doCropTransform:
            transform.extend([
                A.CropAndPad(percent = (-0.02,-0.2), keep_size = False, 
                             sample_independently = True, p = 0.5),
            ])
            
        if doGeometricTransform:
            transform.extend([
                A.HorizontalFlip(p = 0.5)
            ])
        
        if doVisualTransform:
            transform.extend([                
                A.OneOf([
                    A.ChannelShuffle(p = 0.2),
                    A.RGBShift(p = 0.4),
                    A.HueSaturationValue(p = 0.4),                    
                ], p = 0.3),
                
                A.OneOf([
                    A.ColorJitter(p = 0.2),
                    A.CLAHE(p = 0.3),
                    A.RandomBrightnessContrast(p = 0.3),
                    A.RandomGamma(p = 0.2),
                ], p = 0.3),
                
                A.OneOf([
                    A.GaussNoise(p = 0.5),
                    A.ISONoise(p = 0.5),
                    
                ], p = 0.2),

                A.OneOf([
                    #A.AdvancedBlur(p = 0.1),
                    A.Blur(p = 0.1),
                    A.GaussianBlur(p = 0.3),
                    A.GlassBlur(p = 0.1),
                    A.MedianBlur(p = 0.3),
                    A.MotionBlur(p = 0.2),      
                ], p = 0.2),
            
                #A.PixelDropout(p = 0.1),  
            ])
            
    transform.extend([
        A.Lambda(image = resize_to_inputsz, mask = resize_to_inputsz, p = 1.0),
        A.Lambda(mask = scale_0_1, p = 1.0),            
    ])


    if doNormalize:
        transform.append(A.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225],
                                     p = 1.0)
        )

        
    if doTensored:
        transform.append(ToTensorV2(p = 1.0, transpose_mask = True))
        
    return A.Compose(transform)

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, preprocessing = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.preprocessing = preprocessing
        
        self.image_ids = natsorted(os.listdir(self.images_dir))
        self.n_imgs = len(self.image_ids)
        print("Number of Images found: ", self.n_imgs)
        
    def __getitem__(self, i):
        image_id = self.image_ids[i]
        name = "".join(image_id.split(".")[:-1])
        
        image_path = os.path.join(self.images_dir, image_id)
        mask_path = os.path.join(self.masks_dir, name + ".png")
        
        
        image = Image.open(image_path).convert("RGB")
        #image = ImageOps.exif_transpose(image)
        
        mask = Image.open(mask_path).convert("L")
        assert (image.size == mask.size)
        
        image_arr = np.array(image)     
        
        mask_arr = np.array(mask)
        mask_arr[mask_arr > 0] = 255
        mask_arr = np.expand_dims(mask_arr, -1)
        
        image_T, mask_T = None, None 
        if self.preprocessing:
            sample = self.preprocessing(image = image_arr, mask = mask_arr)
            image_T, mask_T = sample['image'], sample['mask']

        return image_T, mask_T

    def __len__(self):
        return self.n_imgs

def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    X_DIR = "C:/Users/RAJDEEP/Downloads/archive/images"
    Y_DIR = "C:/Users/RAJDEEP/Downloads/archive/annotations"

    X_TRAIN_DIR = os.path.join(X_DIR, "train")
    Y_TRAIN_DIR = os.path.join(Y_DIR, "train")

    X_VALID_DIR = os.path.join(X_DIR, "val")
    Y_VALID_DIR = os.path.join(Y_DIR, "val")

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CHECKPOINT_DIR = "./model_checkpoints"

    TRAIN_BS = 8
    VALID_BS = 1

    INPUT_SZ = (320, 320)
    #INPUT_SZ = (720, 720)


    train_dataset = CustomDataset(X_TRAIN_DIR, 
                                Y_TRAIN_DIR, 
                                preprocessing = augment_normalize(doAugment = True, 
                                                                    doNormalize = True,
                                                                    doTensored = True))
    val_dataset = CustomDataset(X_VALID_DIR, 
                                Y_VALID_DIR, 
                                preprocessing = augment_normalize(doAugment = False, 
                                                                doNormalize = True,
                                                                doTensored = True))

    train_loader = DataLoader(train_dataset, 
                            batch_size = TRAIN_BS, 
                            shuffle = True, 
                            num_workers = 0, 
                            pin_memory = True)

    val_loader = DataLoader(val_dataset, 
                            batch_size = VALID_BS, 
                            shuffle = False, 
                            num_workers = 0, 
                            pin_memory = True)
    
