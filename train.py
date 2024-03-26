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

from LPTN_Network import LPTN_Network

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
    
def train(train_loader, model, criterion, optimizer, scheduler, epoch, beta, use_weighted_loss_train):
    model.train()
    stream = tqdm(train_loader)
    
    for i, (images, targets) in enumerate(stream, start=1):        
        images = images.to(DEVICE, non_blocking = True, dtype = torch.float)
        targets = targets.to(DEVICE, non_blocking = True, dtype = torch.float)
        
        #optimizer.zero_grad(set_to_none = True)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled = use_amp):
            outputs = model(images)  
            loss = criterion(outputs, targets, includeBoundaryLoss = use_weighted_loss_train)
     
        if moving_loss['train']:
            moving_loss['train'] = beta * moving_loss['train'] + (1-beta) * loss.item()
        else:
            moving_loss['train'] = loss.item()
            
        loss_values['train'].append(moving_loss['train'])
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        
        if not skip_lr_sched:
            scheduler.step()      

        stream.set_description(
            "Epoch: {epoch}.  --Train--  Loss: {m_loss:04f}".format(epoch = epoch, m_loss = moving_loss['train'])
        )

def validate(val_loader, model, criterion, epoch, beta):
    model.eval()
    stream = tqdm(val_loader)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(stream, start=1):
            images = images.to(DEVICE, non_blocking=True, dtype = torch.float)
            targets = targets.to(DEVICE, non_blocking=True, dtype = torch.float)
            
            with torch.cuda.amp.autocast(enabled = use_amp):
                outputs = model(images)            
                loss = criterion(outputs, targets)

            if moving_loss['valid']:
                moving_loss['valid'] = beta * moving_loss['valid'] + (1-beta) * loss.item()
            else:
                moving_loss['valid'] = loss.item()
            
            loss_values['valid'].append(moving_loss['valid'])
            stream.set_description(
                "Epoch: {epoch}.  --Valid--  Loss: {m_loss:04f}".format(epoch = epoch, m_loss = moving_loss['valid'])
            )

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch, 
                       n_epochs, ckpt_dir, save_freq, beta, use_weighted_loss_train):    
    os.makedirs(ckpt_dir, exist_ok = True)
    
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):        
        train(train_loader, model, criterion, optimizer, scheduler, epoch, beta, use_weighted_loss_train)
        validate(val_loader, model, criterion, epoch, beta)
        
        ckpt_path = os.path.join(ckpt_dir, "{epoch}.pth".format(epoch = epoch))
        
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, ckpt_path)
        
    return model

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


    # ToDO  - Define the Loss Functions

    # Define the Model
    model = LPTN_Network()


    # Training Params / HyperParams
    start_epoch = 0
    n_epochs = 15

    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = learning_rate, 
                                                    steps_per_epoch = len(train_loader), 
                                                    epochs = n_epochs)

    # DEFINE LOSS FUNC
    #loss_fn = 
    save_freq = 3
    beta = 0.9
    use_weighted_loss_train = True

    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

    moving_loss = {'train': 0, 'valid': 0}
    loss_values = {"train": [], "valid": []}

    model.to(DEVICE)

    model = train_and_validate(model, 
                           train_loader, 
                           val_loader,
                           criterion = loss_fn, # ToDO
                           optimizer = optimizer,
                           scheduler = scheduler,
                           start_epoch = start_epoch,
                           n_epochs = n_epochs,                           
                           ckpt_dir = CHECKPOINT_DIR,
                           save_freq = save_freq,
                           beta = beta,
                           use_weighted_loss_train = use_weighted_loss_train)
    
