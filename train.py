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
from torch.utils.tensorboard import SummaryWriter

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


# def get_random_crop(image, **kwargs):

#     max_x = image.shape[1] - INPUT_SZ[0]
#     max_y = image.shape[0] - INPUT_SZ[1]

#     x = random.randint(0, image.shape[1] - INPUT_SZ[1])
#     y = random.randint(0, image.shape[0] - INPUT_SZ[0])

#     crop = image[y: y + INPUT_SZ[0], x: x + INPUT_SZ[1]]

#     return crop
# def scale_0_1(x, **kwargs):
#     return x / 255.

def augment_normalize(doAugment = True, doNormalize = True, doTensored = True):
    transform = []
    
    if doAugment:
        doCropTransform = True
        doGeometricTransform = True
        doVisualTransform = False       
        
        if doCropTransform:
            transform.extend([
                # A.CropAndPad(percent = (-0.02,-0.2), keep_size = False, sample_independently = True, p = 0.5),
                A.RandomCrop(height = INPUT_SZ[0], width = INPUT_SZ[1], p = 1.0)
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
        A.Lambda(image = resize_to_inputsz, mask = resize_to_inputsz, p = 1.0),
        # A.Lambda(image = get_random_crop, mask = get_random_crop, p = 1.0),
        # A.Lambda(mask = scale_0_1, p = 1.0),         
    ])

    if doNormalize:
        # transform.append(A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], p = 1.0))
        transform.append(A.Normalize(mean = 0, std = 1, p = 1.0))

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

        image_arr = np.array(image)  
        target_arr = np.array(target) 
        
        image_T, target_T = None, None 
        if self.preprocessing:
            image_T = self.preprocessing(image = image_arr)['image']
            target_T = self.preprocessing(image = target_arr)['image']
        return image_T, target_T

    def __len__(self):
        return self.n_imgs
     
def train(train_loader, generator, discriminator, criterion, optimizer_generator, optimizer_discriminator, epoch, iteration):
    # Switching both the models to Training mode
    generator.train()
    discriminator.train()

    stream = tqdm(train_loader)

    for _, (images, targets) in enumerate(stream, start = 1):
        images = images.to(DEVICE, non_blocking = True, dtype = torch.float)
        targets = targets.to(DEVICE, non_blocking = True, dtype = torch.float)

        ############################
        # Optimizing the Generator #
        ############################
        for p in discriminator.parameters():
            p.requires_grad = False    

        optimizer_generator.zero_grad()  #optimizer.zero_grad(set_to_none = True)

        generated_images = generator(images)
        discriminator_preds = discriminator(generated_images)

        reconstruction_loss = criterion[0](generated_images, images) 
        gan_loss = criterion[1](discriminator_preds, target_is_real = True, is_disc = False)
        generator_loss = reconstruction_loss + gan_loss

        WRITER.add_scalar('Loss/train/generator/reconstruction', reconstruction_loss.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/generator/gan', gan_loss.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/generator/total', generator_loss.detach().cpu().numpy(), global_step = iteration)
        
        generator_loss.backward()
        optimizer_generator.step()
    
        ###############################
        # Optimizing the Discrminator #
        ###############################
        for p in discriminator.parameters():
            p.requires_grad = True 
        
        optimizer_discriminator.zero_grad()

        generated_images = generator(images)
        discriminator_preds_real = discriminator(targets)
        discriminator_preds_fake = discriminator(generated_images)

        discriminator_loss_real = criterion[1](discriminator_preds_real, target_is_real = True, is_disc=True)
        discriminator_loss_fake = criterion[1](discriminator_preds_fake, target_is_real = False, is_disc=True)
        gradient_loss = CustomLoss.compute_gradient_penalty(discriminator, targets, generated_images)
        disciminator_loss = discriminator_loss_real + discriminator_loss_fake + (100 * gradient_loss)

        disciminator_loss.backward()
        optimizer_discriminator.step()

        WRITER.add_scalar('Loss/train/discriminator/gan_real', discriminator_loss_real.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/discriminator/gan_fake', discriminator_loss_fake.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/discriminator/gradient_penalty', gradient_loss.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/discriminator/total', disciminator_loss.detach().cpu().numpy(), global_step = iteration)

        iteration += 1
        stream.set_description(f"Epoch: {epoch}")
        
    return iteration

def validate(val_loader, generator, discriminator, criterion, epoch, iteration):
    # Switching both the models to Evaluation mode
    generator.eval()
    discriminator.eval()

    stream = tqdm(val_loader)
    
    with torch.no_grad():
        for _, (images, targets) in enumerate(stream, start = 1):
            images = images.to(DEVICE, non_blocking=True, dtype = torch.float)
            targets = targets.to(DEVICE, non_blocking=True, dtype = torch.float)
            
            ############################
            # Validating the Generator #
            ############################
            generated_images = generator(images)
            discriminator_preds = discriminator(generated_images)

            reconstruction_loss = criterion[0](generated_images, images) 
            gan_loss = criterion[1](discriminator_preds, target_is_real = True, is_disc=False)
            generator_loss =  reconstruction_loss + gan_loss

            WRITER.add_scalar('Loss/valid/generator/reconstruction', reconstruction_loss.detach().cpu().numpy(), global_step = iteration)
            WRITER.add_scalar('Loss/valid/generator/gan', gan_loss.detach().cpu().numpy(), global_step = iteration)
            WRITER.add_scalar('Loss/valid/generator/total', generator_loss.detach().cpu().numpy(), global_step = iteration)

            ###############################
            # Validating the Discrminator #
            ###############################
            generated_images = generator(images)
            discriminator_preds_real = discriminator(targets)
            discriminator_preds_fake = discriminator(generated_images)

            discriminator_loss_real = criterion[1](discriminator_preds_real, target_is_real = True, is_disc=True)
            discriminator_loss_fake = criterion[1](discriminator_preds_fake, target_is_real = False, is_disc=True)
            gradient_loss = torch.tensor(0, dtype = torch.int8) #CustomLoss.compute_gradient_penalty(discriminator, targets, outputs)
            loss_disc = discriminator_loss_real + discriminator_loss_fake + (100 * gradient_loss)

            WRITER.add_scalar('Loss/valid/discriminator/gan_real', discriminator_loss_real.detach().cpu().numpy(), global_step = iteration)
            WRITER.add_scalar('Loss/valid/discriminator/gan_fake', discriminator_loss_fake.detach().cpu().numpy(), global_step = iteration)
            WRITER.add_scalar('Loss/valid/discriminator/gradient_penalty', gradient_loss.detach().cpu().numpy(), global_step = iteration)
            WRITER.add_scalar('Loss/valid/discriminator/total', loss_disc.detach().cpu().numpy(), global_step = iteration)

            psnrs = [calculate_psnr(images[i].detach().cpu().numpy(), generated_images[i].detach().cpu().numpy(), crop_border = 4, input_order = "CHW", test_y_channel = False) for i in range(len(images))]
            ssims = [calculate_ssim(images[i].detach().cpu().numpy(), generated_images[i].detach().cpu().numpy(), crop_border = 4, input_order = "CHW", test_y_channel = False) for i in range(len(images))]

            WRITER.add_scalar('Metrics/valid/PSNR', sum(psnrs), global_step = iteration)
            WRITER.add_scalar('Metrics/valid/SSIM', sum(ssims), global_step = iteration)

            iteration += 1
            stream.set_description(f"Epoch: {epoch}")

    return iteration

def train_and_validate(generator, discriminator, train_loader, val_loader, criterion, optimizer_generator, optimizer_discriminator, start_epoch, n_epochs, ckpt_dir, save_freq):
    os.makedirs(ckpt_dir, exist_ok = True)
    train_iteration, valid_iteration = 0, 0
    
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):
        print(f"epoch {epoch}", train_iteration, valid_iteration) 

        train_iteration = train(train_loader, generator, discriminator, criterion, optimizer_generator, optimizer_discriminator, epoch, train_iteration)
        valid_iteration = validate(val_loader, generator, discriminator, criterion, epoch, valid_iteration)

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, "{epoch}.pth".format(epoch = epoch))

            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_generator_state_dict': optimizer_generator.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                }, ckpt_path)
        
    return generator, discriminator


if __name__=="__main__":

    # Data Paths
    if platform.system() == "Linux":
        DATA_DIR = "/home/kumar/LPTN/datasets/FiveK/FiveK_480p/"
    elif platform.system() == "Windows":
        DATA_DIR = "datasets/FiveK_480p/FiveK_480p"
    
    X_TRAIN_DIR = os.path.join(DATA_DIR, "train", "A")
    Y_TRAIN_DIR = os.path.join(DATA_DIR, "train", "B")

    X_VALID_DIR = os.path.join(DATA_DIR, "test", "A")
    Y_VALID_DIR = os.path.join(DATA_DIR, "test", "B")

    # Checkpoint Path
    CHECKPOINT_DIR = "./model_checkpoints"

    # Use GPU if available
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Input Size and Batch Sizes
    INPUT_SZ = (256,256)
    TRAIN_BS = 32
    VALID_BS = 4

    # Tensorboard
    WRITER = SummaryWriter()

    # Datasets
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
    # Dataloaders
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


    # Defining the Models
    generator = LPTN_Network()
    discriminator = Discriminator()

    # Training Params / HyperParams
    start_epoch = 0
    n_epochs = 500

    learning_rate = 0.0001
    optim_params = []
    for k, v in generator.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    # They used MultiStepLR scheduler.

    # Optimizers
    optimizer_generator = torch.optim.Adam(optim_params, lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)

    # Loss functions
    custom_loss = CustomLoss()
    mse_loss = custom_loss.get_reconstruction_loss
    gan_loss = custom_loss.get_gan_loss

    # Training
    save_freq = 100
    beta = 0.9
    use_weighted_loss_train = True

    generator.to(DEVICE)
    discriminator.to(DEVICE)
    
    generator, discriminator = train_and_validate(generator = generator, 
                                                discriminator = discriminator, 
                                                train_loader = train_loader, 
                                                val_loader = val_loader,
                                                criterion = (mse_loss, gan_loss),
                                                optimizer_generator = optimizer_generator,
                                                optimizer_discriminator = optimizer_discriminator,
                                                start_epoch = start_epoch,
                                                n_epochs = n_epochs,                           
                                                ckpt_dir = CHECKPOINT_DIR,
                                                save_freq = save_freq)

    WRITER.flush()