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
# from Utils import FileClient, paths_from_folder, imfrombytes, img2tensor, augment, unpaired_random_crop
sys.path.append("./src/")
import random
# from model import U2NET
from pathlib import Path
from natsort import natsorted
from tqdm.notebook import tqdm
from tqdm import tqdm
from PIL import Image, ImageOps
from torch.autograd import Variable
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

INPUT_SZ=(256,256)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_amp = False
scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

moving_loss = {'train': 0, 'valid': 0}
loss_values = {"train": [], "valid": []}

from LPTN_Network import LPTN_Network
from Discriminator import Discriminator
from loss import CustomLoss
np.random.seed(10)
random.seed(16)
torch.manual_seed(10)
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
# import random

# from codes.data.data_util import (paths_from_folder, paths_from_lmdb)
# from codes.data.transforms import augment, unpaired_random_crop
# from codes.utils import FileClient, imfrombytes, img2tensor

# class UnPairedImageDataset(data.Dataset):

#     def __init__(self, opt):
#         super(UnPairedImageDataset, self).__init__()
#         self.opt = opt
#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None

#         self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'


#         self.paths_lq = paths_from_folder(self.lq_folder)
#         self.paths_gt = paths_from_folder(self.gt_folder)

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
#         # image range: [0, 1], float32.

#         lq_path = self.paths_lq[index % len(self.paths_lq)]
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         img_lq = imfrombytes(img_bytes, float32=True)

#         gt_path = self.paths_gt[index % len(self.paths_gt)]
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         img_gt = imfrombytes(img_bytes, float32=True)

#         img_ref = img_gt

#         # augmentation for training
#         if self.opt['phase'] == 'train':
#             if_fix = self.opt['if_fix_size']
#             gt_size = self.opt['gt_size']
#             if not if_fix and self.opt['batch_size_per_gpu'] != 1:
#                 raise ValueError(
#                     f'Param mismatch. Only support fix data shape if batchsize > 1 or num_gpu > 1.')
#             # random crop
#             img_lq, img_ref = unpaired_random_crop(img_lq, img_ref, if_fix, gt_size)
#             # flip, rotation
#             img_lq, img_ref = augment([img_lq, img_ref], self.opt['use_flip'], self.opt['use_rot'])

#         # BGR to RGB, HWC to CHW, numpy to tensor

#         img_lq, img_ref = img2tensor([img_lq, img_ref], bgr2rgb=True, float32=True)
#         # normalize        
#         if self.mean is not None or self.std is not None:
#             normalize(img_lq, self.mean, self.std, inplace=True)
#             normalize(img_ref, self.mean, self.std, inplace=True)

#         return {
#             'images': img_lq,
#             'target': img_ref,
#             'image_path': lq_path,
#             'target_path': gt_path,
#         }

#     def __len__(self):
#         return len(self.paths_lq)
#         # return 100

class CustomDataset(Dataset):
    def __init__(self, images_dir, target_dir, preprocessing = None):
        self.images_dir = images_dir
        self.target_dir=target_dir
        self.preprocessing = preprocessing

        
        self.image_ids = natsorted(os.listdir(self.images_dir))        
        self.target_ids=natsorted(os.listdir(self.target_dir))
        self.n_imgs = len(self.image_ids)
        print("Number of Images found: ", self.n_imgs)
        
    def __getitem__(self, i):
        image_id = self.image_ids[i]
        target_id = self.target_ids[i]
        
        # name = "".join(image_id.split(".")[:-1])
        
        image_path = os.path.join(self.images_dir, image_id)
        target_path=os.path.join(self.target_dir, target_id)
        image = Image.open(image_path)
        target = Image.open(target_path)

        #image = ImageOps.exif_transpose(image)
        
        # mask = Image.open(mask_path).convert("L")
#         assert (image.size == mask.size)
        
        image_arr = np.array(image).astype(np.float32)/255     
        target_arr=np.array(target).astype(np.float32)/255     
        # mask_arr = np.array(mask)
        # mask_arr[mask_arr > 0] = 255
        # mask_arr = np.expand_dims(mask_arr, -1)
        
        image_T, mask_T = None, None 
        if self.preprocessing:
            sample = self.preprocessing(image = image_arr)
            image_T = sample['image']
            sample = self.preprocessing(image = target_arr)
            target_T= sample['image']
        return image_T, target_T

    def __len__(self):
        return self.n_imgs
     
def train(train_loader, model, disc, criterion, optimizer_model, optimizer_disc, scheduler, epoch, beta, use_weighted_loss_train):
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
        # with torch.cuda.amp.autocast(enabled = use_amp):
        output = model(images)        
        disc_out = disc(output)
        comp1=criterion[0](output, images) 
        comp2=criterion[1](disc_out, target_is_real = True, is_disc=False)
        loss_model =  comp1+comp2
        loss[f"Epoch{epoch}"]['reconstruction_mse']=comp1
        loss[f"Epoch{epoch}"]['reconstruction_gan']=comp2
        loss[f"Epoch{epoch}"]['reconstruction']=loss_model
        
        
        # torch.save(model.state_dict(), "model_custom.pth")
        # input()
        # print(output)
        # input()
            
        # loss_values['train'].append(moving_loss['train'])
        # print(model.low_frequency_layers[0].weight.grad)
        
        loss_model.backward()
        optimizer_model.step()
        # print(comp1,comp2)
        # print(loss_model)
        # # print(optimizer_model)
        # # print(loss_disc.detach().item(), loss_model.detach().item())
        
        # print(model.low_frequency_layers[0].weight.grad)
        # input()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer_model)

        # scale = scaler.get_scale()
        # scaler.update()
        # skip_lr_sched = (scale > scaler.get_scale())
        
        # if not skip_lr_sched:
        #     scheduler.step()      

        # stream.set_description(
        #     "Epoch: {epoch}.  --Train--  Loss: {m_loss:04f}".format(epoch = epoch, m_loss = moving_loss['train'])
        # )

        # Optimizing the Discriminator
        for p in disc.parameters():
            p.requires_grad = True 
        
        optimizer_disc.zero_grad()
        # with torch.cuda.amp.autocast(enabled = use_amp):
        output = model(images)
        disc_out_real = disc(targets)
        loss_disc_real = criterion[1](disc_out_real, target_is_real = True, is_disc=True)

        disc_out_fake = disc(output)
        loss_disc_fake = criterion[1](disc_out_fake, target_is_real = False, is_disc=True)

        gradient_loss = CustomLoss.compute_gradient_penalty(disc, targets, output)
        gp_opt = 100
        loss_disc = loss_disc_real + loss_disc_fake + (gp_opt * gradient_loss)
        loss_disc.backward()
        optimizer_disc.step()
        loss[f"Epoch{epoch}"]['discriminator_real']=loss_disc_real
        loss[f"Epoch{epoch}"]['discriminator_fake']=loss_disc_fake
        loss[f"Epoch{epoch}"]['discriminator_gradient_loss']=gradient_loss
        loss[f"Epoch{epoch}"]['discriminator']=loss_disc
        
        # print( loss_model.detach().item(),loss_disc.detach().item())
        # torchvision.utils.save_image(output, "reconstruction.png")
        # torchvision.utils.save_image(images, "input.png")        
        # print(-torch.mean(disc_out_real).detach().item(), torch.mean(disc_out_fake).detach().item())
        # images=torchvision.transforms.transforms.F.rotate(images,270, expand=True)
        # print(images)
        # print(images.shape)
        # if moving_loss['train']:
        #     moving_loss['train'][0] = beta * moving_loss['train'] + (1-beta) * loss_model.item()
        #     moving_loss['train'][0] = beta * moving_loss['train'] + (1-beta) * loss_disc.item()
        # else:
        #     moving_loss['train'] = loss_model.item()
        
        # print(loss_disc_real.detach().item(),  loss_disc_fake.detach().item(), gradient_loss.detach().item())
        # print(loss_model.detach().item(),loss_disc.detach().item())        
        # torchvision.utils.save_image(targets, "Target.png")
    return loss
        # input()

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

def train_and_validate(model, disc, train_loader, val_loader, criterion, optimizer_model, optimizer_disc, scheduler, start_epoch, 
                       n_epochs, ckpt_dir, save_freq, beta, use_weighted_loss_train):    
    os.makedirs(ckpt_dir, exist_ok = True)
    losses={}
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):  
        print(f"epoch {epoch}")      
        loss=train(train_loader, model, disc, criterion, optimizer_model, optimizer_disc, scheduler, epoch, beta, use_weighted_loss_train)
        # validate(val_loader, model, criterion, epoch, beta)
        losses[f"Epoch{epoch}"]=loss[f"Epoch{epoch}"]
        print(losses)
        ckpt_path = os.path.join(ckpt_dir, "{epoch}.pth".format(epoch = epoch))
        
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'optimizer_disc_state_dict':optimizer_disc.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                }, ckpt_path)
        
    return model,disc

def main():
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    # ToDo - Make these args - argparse
    X_DIR = "/home/kumar/LPTN/datasets/FiveK/FiveK_480p/"
    Y_DIR = "/home/kumar/LPTN/datasets/FiveK/FiveK_480p/"

    X_TRAIN_DIR = os.path.join(X_DIR, "train/A")
    Y_TRAIN_DIR = os.path.join(Y_DIR, "train/B")
    # X_DIR = "/home/kumar/LPTN/datasets/FiveK/FiveK_480p/"
    # Y_DIR = "/home/kumar/LPTN/datasets/FiveK/FiveK_480p/"

    # X_VALID_DIR = os.path.join(X_DIR, "test/A")
    # Y_VALID_DIR = os.path.join(Y_DIR, "test/B")

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CHECKPOINT_DIR = "./model_checkpoints"

    TRAIN_BS = 16
    VALID_BS = 1

    INPUT_SZ = (320, 320)
    #INPUT_SZ = (720, 720)

    # fivek = MITAboveFiveK(root="datasets/", split="train", download=True, experts=["c"])
    train_dataset = CustomDataset(X_TRAIN_DIR, 
                                  Y_TRAIN_DIR,
                                preprocessing = augment_normalize(doAugment = True, 
                                                                    doNormalize = True,
                                                                    doTensored = True))
    val_dataset = CustomDataset(X_TRAIN_DIR, 
                                X_TRAIN_DIR,
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

    # # ToDO  - Define the Loss Functions

    # # Define the Model
    model = LPTN_Network()
    disc = Discriminator()
    # # Training Params / HyperParams
    start_epoch = 0
    n_epochs = 5

    learning_rate = 0.0001
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    optimizer_model = torch.optim.Adam(optim_params, lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
    
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = learning_rate, 
    #                                                 steps_per_epoch = len(train_loader), 
    #                                                 epochs = n_epochs)

    # # DEFINE LOSS FUNC
    custom_loss = CustomLoss()

    mse_loss = custom_loss.get_reconstruction_loss
    gan_loss = custom_loss.get_gan_loss

    save_freq = 1
    beta = 0.9
    use_weighted_loss_train = True

    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

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
                        #    scheduler = scheduler,
                           scheduler = None,
                           start_epoch = start_epoch,
                           n_epochs = n_epochs,                           
                           ckpt_dir = CHECKPOINT_DIR,
                           save_freq = save_freq,
                           beta = beta,
                           use_weighted_loss_train = use_weighted_loss_train)
    # inp=train_dataset[0][0].unsqueeze(dim=0).to(DEVICE)
    # output=model(inp)
    # # disc_output=disc(inp)
    # # disc_output_2=disc(output)
    
    # # print(disc_output)
    # # print(disc_output_2)
    # torchvision.utils.save_image(inp.to("cpu"), "original.png")
    # torchvision.utils.save_image(output, "reconstruction.png")
if __name__=="__main__":
    main()