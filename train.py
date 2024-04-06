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
import sys
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from tqdm import tqdm
from natsort import natsorted
from PIL import Image, ImageOps
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

np.random.seed(10)
random.seed(16)
torch.manual_seed(30)

sys.path.append("./src/")

from loss import CustomLoss
from LPTN_Network import LPTN_Network
from Discriminator import Discriminator
from metrics import calculate_psnr, calculate_ssim

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
def get_random_crop(image, **kwargs):

    max_x = image.shape[1] - INPUT_SZ[0]
    max_y = image.shape[0] - INPUT_SZ[1]

    x = random.randint(0, image.shape[1] - INPUT_SZ[1])
    y = random.randint(0, image.shape[0] - INPUT_SZ[0])

    crop = image[y: y + INPUT_SZ[0], x: x + INPUT_SZ[1]]

    return crop
def scale_0_1(x, **kwargs):
    
    return x / 255.

def augment_normalize(doAugment = True, doNormalize = True, doTensored = True):
    transform = []
    
    if doAugment:
        doCropTransform = False
        doGeometricTransform = True
        doVisualTransform = False       
        
        if doCropTransform:
            transform.extend([
                A.CropAndPad(percent = (-0.02,-0.2), keep_size = False, 
                             sample_independently = True, p = 0.5),
            ])
            
        if doGeometricTransform:
            transform.extend([
                A.Flip(p=0.8)
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
                    # A.AdvancedBlur(p = 0.1),
                    A.Blur(p = 0.1),
                    A.GaussianBlur(p = 0.3),
                    A.GlassBlur(p = 0.1),
                    A.MedianBlur(p = 0.3),
                    A.MotionBlur(p = 0.2),      
                ], p = 0.2),
            
                # A.PixelDropout(p = 0.1),  
            ])
            
    transform.extend([
        A.Lambda(image = get_random_crop, mask = get_random_crop, p = 1.0),
        A.Lambda(mask = scale_0_1, p = 1.0),          
    ])

    # if doNormalize:
    #     transform.append(A.Normalize(mean = [0.485, 0.456, 0.406],
    #                                  std = [0.229, 0.224, 0.225],
    #                                  p = 1.0)
    #     )

    if doTensored:
        transform.append(ToTensorV2(p = 1.0, transpose_mask = True))
        
    return A.Compose(transform)

class CustomDataset(Dataset):
    def __init__(self, images_dir, target_dir, preprocessing = None):
        self.images_dir = images_dir
        self.target_dir = target_dir
        self.preprocessing = preprocessing

        
        self.image_ids = natsorted(os.listdir(self.images_dir))        
        self.target_ids = natsorted(os.listdir(self.target_dir))
        self.n_imgs = len(self.image_ids)
        print("Number of Images found: ", self.n_imgs)
        
    def __getitem__(self, i):
        image_id = self.image_ids[i]
        target_id = self.target_ids[i]
        
        image_path = os.path.join(self.images_dir, image_id)
        target_path = os.path.join(self.target_dir, target_id)

        image = Image.open(image_path)
        target = Image.open(target_path)
        # image = ImageOps.exif_transpose(image)

        image_arr = np.array(image).astype(np.float32) / 255     
        target_arr = np.array(target).astype(np.float32) / 255     
        
        image_T, target_T = None, None 
        if self.preprocessing:
            image_T = self.preprocessing(image = image_arr)['image']
            target_T = self.preprocessing(image = target_arr)['image']
        return image_T, target_T

    def __len__(self):
        return self.n_imgs
     
def train(train_loader, model, disc, criterion, optimizer_model, optimizer_disc, epoch, beta):
    model.train()
    disc.train()
    stream = tqdm(train_loader, disable=False)
    loss={
        f"Epoch{epoch}":{
            
        }
    }
    
    for i, (images, targets) in enumerate(stream, start=1): 
        images = images.to(DEVICE, non_blocking = True, dtype = torch.float)
        targets = targets.to(DEVICE, non_blocking = True, dtype = torch.float)
        if(len(images.shape)==3):
            images=images[None,:,:,:]

        # Optimizing the Generator
        for p in disc.parameters():
            p.requires_grad = False       

        optimizer_model.zero_grad()  #optimizer.zero_grad(set_to_none = True)
        outputs = model(images)        
        disc_out = disc(outputs)
        comp1=criterion[0](outputs, images) 
        comp2=criterion[1](disc_out, target_is_real = True, is_disc=False)
        loss_model =  comp1+comp2
        loss[f"Epoch{epoch}"]['reconstruction_mse']=comp1
        loss[f"Epoch{epoch}"]['reconstruction_gan']=comp2
        loss[f"Epoch{epoch}"]['reconstruction']=loss_model
        
        loss_model.backward()
        optimizer_model.step()
    
        # Optimizing the Discriminator
        for p in disc.parameters():
            p.requires_grad = True 
        
        optimizer_disc.zero_grad()
        outputs = model(images)
        disc_out_real = disc(targets)
        loss_disc_real = criterion[1](disc_out_real, target_is_real = True, is_disc=True)

        disc_out_fake = disc(outputs)
        loss_disc_fake = criterion[1](disc_out_fake, target_is_real = False, is_disc=True)

        gradient_loss = CustomLoss.compute_gradient_penalty(disc, targets, outputs)
        gp_opt = 100
        loss_disc = loss_disc_real + loss_disc_fake + (gp_opt * gradient_loss)
        loss_disc.backward()
        optimizer_disc.step()
        loss[f"Epoch{epoch}"]['discriminator_real']=loss_disc_real
        loss[f"Epoch{epoch}"]['discriminator_fake']=loss_disc_fake
        loss[f"Epoch{epoch}"]['discriminator_gradient_loss']=gradient_loss
        loss[f"Epoch{epoch}"]['discriminator']=loss_disc
        
    return loss

def validate(val_loader, model, disc, criterion, epoch, beta):
    model.eval()
    disc.eval()
    stream = tqdm(val_loader)

    loss = {
        f"Epoch{epoch}":{
            
        }
    }

    metrics = {
        f"Epoch{epoch}": {

        }
    }
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(stream, start=1):
            images = images.to(DEVICE, non_blocking=True, dtype = torch.float)
            targets = targets.to(DEVICE, non_blocking=True, dtype = torch.float)
            if(len(images.shape)==3):
                images=images[None,:,:,:]
            
            outputs = model(images)
            disc_out = disc(outputs)
            comp1=criterion[0](outputs, images) 
            comp2=criterion[1](disc_out, target_is_real = True, is_disc=False)
            loss_model =  comp1+comp2
            loss[f"Epoch{epoch}"]['reconstruction_mse']=comp1
            loss[f"Epoch{epoch}"]['reconstruction_gan']=comp2
            loss[f"Epoch{epoch}"]['reconstruction']=loss_model
            
            # loss_values['valid'].append(moving_loss['valid'])
            # stream.set_description(
            #     "Epoch: {epoch}.  --Valid--  Loss: {m_loss:04f}".format(epoch = epoch, m_loss = moving_loss['valid'])
            # )


            outputs = model(images)
            disc_out_real = disc(targets)
            loss_disc_real = criterion[1](disc_out_real, target_is_real = True, is_disc=True)

            disc_out_fake = disc(outputs)
            loss_disc_fake = criterion[1](disc_out_fake, target_is_real = False, is_disc=True)

            # gradient_loss = CustomLoss.compute_gradient_penalty(disc, targets, outputs)
            gradient_loss = 0
            gp_opt = 100
            loss_disc = loss_disc_real + loss_disc_fake + (gp_opt * gradient_loss)
            loss[f"Epoch{epoch}"]['discriminator_real']=loss_disc_real
            loss[f"Epoch{epoch}"]['discriminator_fake']=loss_disc_fake
            loss[f"Epoch{epoch}"]['discriminator_gradient_loss']=gradient_loss
            loss[f"Epoch{epoch}"]['discriminator']=loss_disc

            metrics[f"Epoch{epoch}"]["PSNR"] = [calculate_psnr(images[i].detach().cpu().numpy(), outputs[i].detach().cpu().numpy(), crop_border = 4, input_order = "CHW", test_y_channel = False) for i in range(len(images))]
            metrics[f"Epoch{epoch}"]["SSIM"] = [calculate_ssim(images[i].detach().cpu().numpy(), outputs[i].detach().cpu().numpy(), crop_border = 4, input_order = "CHW", test_y_channel = False) for i in range(len(images))]
            
    return loss, metrics

def train_and_validate(model, disc, train_loader, val_loader, criterion, optimizer_model, optimizer_disc, start_epoch, 
                       n_epochs, ckpt_dir, save_freq, beta):    
    os.makedirs(ckpt_dir, exist_ok = True)
    losses = {"Train": {}, "Valid": {}}

    metrics = {"Train": {}, "Valid": {}}
    
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):  
        print(f"epoch {epoch}")  

        train_loss = train(train_loader, model, disc, criterion, optimizer_model, optimizer_disc, epoch, beta)
        valid_loss, valid_metrics = validate(val_loader, model, disc, criterion, epoch, beta)

        # losses[f"Epoch{epoch}"]=loss[f"Epoch{epoch}"]

        losses["Train"] = train_loss
        losses["Valid"] = valid_loss

        metrics["Valid"] = valid_metrics

        print(losses)
        print()
        print(metrics)  

        ckpt_path = os.path.join(ckpt_dir, "{epoch}.pth".format(epoch = epoch))
        
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'optimizer_disc_state_dict':optimizer_disc.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                }, ckpt_path)
        
    return model,disc


if __name__=="__main__":

    if platform.system() == "Linux":
        DATA_DIR = "/home/kumar/LPTN/datasets/FiveK/FiveK_480p/"
    elif platform.system() == "Windows":
        DATA_DIR = "datasets/FiveK_480p/FiveK_480p"
    
    X_TRAIN_DIR = os.path.join(DATA_DIR, "train", "A")
    Y_TRAIN_DIR = os.path.join(DATA_DIR, "train", "B")

    X_VALID_DIR = os.path.join(DATA_DIR, "test", "A")
    Y_VALID_DIR = os.path.join(DATA_DIR, "test", "B")

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CHECKPOINT_DIR = "./model_checkpoints_5"
    
    INPUT_SZ = (256,256)
    TRAIN_BS = 32
    VALID_BS = 4

    moving_loss = {'train': 0, 'valid': 0}
    loss_values = {"train": [], "valid": []}

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


    # # Define the Model
    model = LPTN_Network()
    disc = Discriminator()

    # # Training Params / HyperParams
    start_epoch = 0
    n_epochs = 500

    learning_rate = 0.0001
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    # They used MultiStepLR scheduler.

    optimizer_model = torch.optim.Adam(optim_params, lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)

    # # DEFINE LOSS FUNC
    custom_loss = CustomLoss()

    mse_loss = custom_loss.get_reconstruction_loss
    gan_loss = custom_loss.get_gan_loss

    save_freq = 100
    beta = 0.9
    use_weighted_loss_train = True

    moving_loss = {'train': 0, 'valid': 0}
    loss_values = {"train": [], "valid": []}

    model.to(DEVICE)
    disc.to(DEVICE)
    
    model, disc = train_and_validate(model, disc, 
                           train_loader, 
                           val_loader,
                           criterion = (mse_loss, gan_loss),
                           optimizer_model = optimizer_model,
                           optimizer_disc = optimizer_disc,
                           start_epoch = start_epoch,
                           n_epochs = n_epochs,                           
                           ckpt_dir = CHECKPOINT_DIR,
                           save_freq = save_freq,
                           beta = beta)