import os
import sys
import torch
import random
import platform
import numpy as np
import albumentations as A
from torch.utils import data as data
# import torchvision
from tqdm import tqdm
from natsort import natsorted
from PIL import Image, ImageOps
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import cv2
import argparse
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

def augment_normalize(doAugment = True, doNormalize = True, doTensored = True, doResize=True):
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
                A.Flip(p = 0.8)
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
    if(doResize):
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
        image = ImageOps.exif_transpose(image)
        target = ImageOps.exif_transpose(target)

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
    generator_reconstruction_losses=[]
    generator_gan_losses=[]
    generator_total_losses=[]
    discriminator_losses_fake=[]
    discriminator_losses_real=[]
    discriminator_gradient_penalties=[]
    discriminator_losses_total=[]
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
        generator_reconstruction_losses.append(reconstruction_loss.detach().cpu().numpy())
        generator_gan_losses.append(gan_loss.detach().cpu().numpy())
        generator_total_losses.append(generator_loss.detach().cpu().numpy())

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
        print(discriminator_preds_real.mean().item(), discriminator_preds_fake.mean().item())

        discriminator_loss_real = criterion[1](discriminator_preds_real, target_is_real = True, is_disc = True)
        discriminator_loss_fake = criterion[1](discriminator_preds_fake, target_is_real = False, is_disc = True)
        gradient_loss = CustomLoss.compute_gradient_penalty(discriminator, targets, generated_images)
        disciminator_loss = discriminator_loss_real + discriminator_loss_fake + (100 * gradient_loss)
        disciminator_loss.backward()
        optimizer_discriminator.step()
        # print(discriminator_loss_fake.detach().cpu().numpy())
        # input()
        discriminator_losses_fake.append(discriminator_loss_fake)
        discriminator_losses_real.append(discriminator_loss_real)
        discriminator_gradient_penalties.append(gradient_loss)
        discriminator_losses_total.append(disciminator_loss)
        WRITER.add_scalar('Loss/train/discriminator/gan_real', discriminator_loss_real.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/discriminator/gan_fake', discriminator_loss_fake.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/discriminator/gradient_penalty', gradient_loss.detach().cpu().numpy(), global_step = iteration)
        WRITER.add_scalar('Loss/train/discriminator/total', disciminator_loss.detach().cpu().numpy(), global_step = iteration)

        iteration += 1
        stream.set_description(f"Epoch: {epoch}")
    WRITER.add_scalar("Averages/train/generator/reconstruction_loss", sum(generator_reconstruction_losses)/ len(generator_reconstruction_losses), global_step = epoch)
    WRITER.add_scalar("Averages/train/generator/gan_loss", sum(generator_gan_losses)/ len(generator_gan_losses), global_step = epoch)
    WRITER.add_scalar("Averages/train/generator/total_loss", sum(generator_total_losses)/ len(generator_total_losses), global_step = epoch)
    
    WRITER.add_scalar('Averages/train/discriminator/gan_real_loss', sum(discriminator_losses_real)/ len(discriminator_losses_real), global_step = epoch)
    WRITER.add_scalar('Averages/train/discriminator/gan_fake_loss', sum(discriminator_losses_fake)/ len(discriminator_losses_fake), global_step = epoch)
    WRITER.add_scalar('Averages/train/discriminator/gradient_penalty_loss', sum(discriminator_gradient_penalties)/ len(discriminator_gradient_penalties), global_step = epoch)
    WRITER.add_scalar('Averages/train/discriminator/total_average_loss', sum(discriminator_losses_total)/ len(discriminator_losses_total), global_step = epoch)

    return iteration

def validate(val_loader, generator, discriminator, criterion, epoch, iteration, visualize):
    # Switching both the models to Evaluation mode
    generator.eval()
    discriminator.eval()

    stream = tqdm(val_loader)
    generator_reconstruction_losses=[]
    generator_gan_losses=[]
    generator_total_losses=[]
    discriminator_losses_fake=[]
    discriminator_losses_real=[]
    discriminator_gradient_penalties=[]
    discriminator_losses_total=[]
    all_psnrs=[]
    all_ssims=[]
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
            generator_reconstruction_losses.append(reconstruction_loss.detach().cpu().numpy())
            generator_gan_losses.append(gan_loss.detach().cpu().numpy())
            generator_total_losses.append(generator_loss.detach().cpu().numpy())
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
            discriminator_losses_fake.append(discriminator_loss_fake)
            discriminator_losses_real.append(discriminator_loss_real)
            discriminator_gradient_penalties.append(gradient_loss)
            discriminator_losses_total.append(loss_disc)

            psnrs = [calculate_psnr(images[i].detach().cpu().numpy(), generated_images[i].detach().cpu().numpy(), crop_border = 4, input_order = "CHW", test_y_channel = False) for i in range(len(images))]
            ssims = [calculate_ssim(images[i].detach().cpu().numpy(), generated_images[i].detach().cpu().numpy(), crop_border = 4, input_order = "CHW", test_y_channel = False) for i in range(len(images))]
            all_psnrs.append(sum(psnrs))
            all_ssims.append(sum(ssims))
            
            WRITER.add_scalar('Metrics/valid/PSNR', sum(psnrs), global_step = iteration)
            WRITER.add_scalar('Metrics/valid/SSIM', sum(ssims), global_step = iteration)
            
            iteration += 1
            stream.set_description(f"Epoch: {epoch}")
    WRITER.add_scalar("Averages/valid/generator/reconstruction_loss", sum(generator_reconstruction_losses)/ len(generator_reconstruction_losses), global_step = epoch)
    WRITER.add_scalar("Averages/valid/generator/gan_loss", sum(generator_gan_losses)/ len(generator_gan_losses), global_step = epoch)
    WRITER.add_scalar("Averages/valid/generator/total_loss", sum(generator_total_losses)/ len(generator_total_losses), global_step = epoch)
    
    WRITER.add_scalar('Averages/valid/discriminator/gan_real_loss', sum(discriminator_losses_real)/ len(discriminator_losses_real), global_step = epoch)
    WRITER.add_scalar('Averages/valid/discriminator/gan_fake_loss', sum(discriminator_losses_fake)/ len(discriminator_losses_fake), global_step = epoch)
    WRITER.add_scalar('Averages/valid/discriminator/gradient_penalty', sum(discriminator_gradient_penalties)/ len(discriminator_gradient_penalties), global_step = epoch)
    WRITER.add_scalar('Averages/valid/discriminator/total_loss', sum(discriminator_losses_total)/ len(discriminator_losses_total), global_step = epoch)
    WRITER.add_scalar('Averages/valid/PSNR', sum(psnrs)/len(psnrs), global_step = epoch)
    WRITER.add_scalar('Averages/valid/SSIM', sum(all_ssims)/len(all_ssims), global_step = epoch)

    return iteration

def visualization(generator, val_loader, ckpt_dir,epoch):
    val_dataset = CustomDataset(val_loader.dataset.images_dir, 
                            val_loader.dataset.target_dir,
                            preprocessing = augment_normalize(doAugment = False, 
                                                            doNormalize = True,
                                                            doTensored = True, doResize=False))
 
    new_loader=DataLoader(dataset=val_dataset,
                          batch_size=1,
                          shuffle = False, 
                          num_workers = 0, 
                          pin_memory = True)

    stream = tqdm(new_loader)    
    
    with torch.no_grad():
        for _, (images, targets) in enumerate(stream, start = 0):
            assert new_loader.dataset.image_ids[_] == new_loader.dataset.target_ids[_]
            save_dir=os.path.join(ckpt_dir,"Visualizations","{epoch}".format(epoch = epoch))
            if(not os.path.exists(save_dir)):
                os.makedirs(save_dir, exist_ok=True)
            save_path=os.path.join(save_dir, new_loader.dataset.image_ids[_])
            images = images.to(DEVICE, non_blocking = True, dtype = torch.float)
            targets = targets.to(DEVICE, non_blocking = True, dtype = torch.float)
            in_image= cv2.cvtColor((images.squeeze(0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
            out_image=generator(images).squeeze(0).permute(1, 2, 0).cpu().numpy()            
            out_image=cv2.cvtColor((255*(out_image - np.min(out_image))/np.ptp(out_image)).astype(np.uint8), cv2.COLOR_RGB2BGR)             
            # out_image= cv2.cvtColor((generator(images).squeeze(0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            actual_image=cv2.cvtColor((targets.squeeze(0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            saved_img=np.hstack([in_image, out_image,actual_image])
            cv2.imwrite(save_path, saved_img)
            
def train_and_validate(generator, discriminator, train_loader, val_loader, criterion, optimizer_generator, optimizer_discriminator, start_epoch, n_epochs, ckpt_dir, save_freq, visualize):
    os.makedirs(ckpt_dir, exist_ok = True)
    train_iteration, valid_iteration = 0, 0
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):
        train_iteration = train(train_loader, generator, discriminator, criterion, optimizer_generator, optimizer_discriminator, epoch, train_iteration)
        valid_iteration = validate(val_loader, generator, discriminator, criterion, epoch, valid_iteration, visualize)

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir,"Checkpoints" ,"{epoch}.pth".format(epoch = epoch))
            if(not os.path.exists(os.path.join(ckpt_dir,"Checkpoints" ))):
                os.makedirs(os.path.join(ckpt_dir,"Checkpoints" ), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_generator_state_dict': optimizer_generator.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                }, ckpt_path)
            if(visualize):
                visualization(generator, val_loader, ckpt_dir, epoch)
    return generator, discriminator

def generate_checkpoint_dir(base_dir):
    return os.path.join(base_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))    
    
def main(args):
    global DEVICE, WRITER, INPUT_SZ
    # Data Paths
    # if platform.system() == "Linux":
    #     DATA_DIR ="/home/kumar/LPTN/datasets/FiveK/FiveK_480p/"
    # elif platform.system() == "Windows":
    #     DATA_DIR = "datasets/FiveK_480p/FiveK_480p"
    DATA_DIR=args.DataDir
    if(args.DatasetType.lower()=="fivek"):
        X_TRAIN_DIR = os.path.join(DATA_DIR, "train", "A")
        Y_TRAIN_DIR = os.path.join(DATA_DIR, "train", "B")

        X_VALID_DIR = os.path.join(DATA_DIR, "test", "A")
        Y_VALID_DIR = os.path.join(DATA_DIR, "test", "B")
    elif(args.DatasetType.lower()=="summer2winter"):
        raise Exception("Not implemented YET!")
    else:
        raise Exception("Dataset type not supported!")

    # Checkpoint Path
    if(not args.ResumeTraining):
        CHECKPOINT_DIR = generate_checkpoint_dir(args.ExperimentDir)
    elif(args.ResumeDir is not None):
        CHECKPOINT_DIR = args.ResumeDir
    else:
        all_subdirs = [os.path.join(args.ExperimentDir, d) for d in os.listdir(args.ExperimentDir) if os.path.isdir(os.path.join(args.ExperimentDir, d))]
        # print([d for d in os.listdir(args.ExperimentDir)])
        if(len(all_subdirs)==0):
            raise Exception("No directory to resume training from")
        CHECKPOINT_DIR=max(all_subdirs, key=os.path.getmtime)
    # Use GPU if available
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Input Size and Batch Sizes
    INPUT_SZ = (args.InputSize,args.InputSize)
    TRAIN_BS = args.TrainBS
    VALID_BS = args.ValidBS

    # Tensorboard
    if(not args.ResumeTraining):
        WRITER = SummaryWriter()
    elif(args.TensorboardDir is not None):
        WRITER=SummaryWriter(log_dir=args.TensorboardDir)
    else:
        log_dir="./runs"
        all_subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        for d in all_subdirs:
            if(CHECKPOINT_DIR[-8:] in d):
                WRITER=SummaryWriter(log_dir=d)
        
    # Datasets
    train_dataset = CustomDataset(X_TRAIN_DIR, 
                                Y_TRAIN_DIR,
                                preprocessing = augment_normalize(doAugment = True, 
                                                                doNormalize = True,
                                                                doTensored = True, doResize=True))
    val_dataset = CustomDataset(X_VALID_DIR, 
                                Y_VALID_DIR,
                                preprocessing = augment_normalize(doAugment = False, 
                                                                doNormalize = True,
                                                                doTensored = True, doResize=True))
    # val_dataset=None
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
    

    # Training Params / HyperParams
    start_epoch = 0
    n_epochs = args.NumEpochs
    learning_rate = 0.0001
    if(not args.ResumeTraining):
        generator = LPTN_Network()
        discriminator = Discriminator()    
        optim_params = []
        for k, v in generator.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        # Optimizers                    
        optimizer_generator = torch.optim.Adam(optim_params, lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
        generator.to(DEVICE)
        discriminator.to(DEVICE)
    elif(args.ResumeTraining):
        checkpoint_names=os.listdir(os.path.join(CHECKPOINT_DIR, "Checkpoints"))
        checkpoint_paths=[os.path.join(CHECKPOINT_DIR,"Checkpoints",n)for n in checkpoint_names]
        checkpoint_path=max(checkpoint_paths, key=os.path.getmtime)
        
        full_model_state_dict=torch.load(checkpoint_path)
        start_epoch=full_model_state_dict['epoch']
        generator = LPTN_Network()
        discriminator = Discriminator()      
        generator.to(DEVICE)
        discriminator.to(DEVICE)
        optim_params = []
        for k, v in generator.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        optimizer_generator = torch.optim.Adam(optim_params, lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)              
        generator.load_state_dict(full_model_state_dict['generator_state_dict'])
        discriminator.load_state_dict(full_model_state_dict['discriminator_state_dict'])        
        optimizer_generator.load_state_dict(full_model_state_dict['optimizer_generator_state_dict'])
        optimizer_discriminator.load_state_dict(full_model_state_dict['optimizer_discriminator_state_dict'])

    # Loss functions
    custom_loss = CustomLoss()
    mse_loss = custom_loss.get_reconstruction_loss
    gan_loss = custom_loss.get_gan_loss

    # Training
    save_freq = args.SaveFreq
    beta = 0.9
    use_weighted_loss_train = True
    visualize=True
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
                                                save_freq = save_freq, 
                                                visualize=visualize)
    # inp=train_dataset[0][0].unsqueeze(dim=0).to(DEVICE)
    # output=generator(inp)
    # # disc_output=disc(inp)
    # # disc_output_2=disc(output)
    
    # # print(disc_output)
    # # print(disc_output_2)
    # torchvision.utils.save_image(inp.to("cpu"), "original.png")
    # torchvision.utils.save_image(output, "reconstruction.png")

    WRITER.flush()    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--DataDir", help="Path To directory containing training/validation data", type=str)
    parser.add_argument("--DatasetType", help="Type of dataset [Currently Supported FiveK and Summer2Winter]", type=str)
    parser.add_argument("--NumEpochs", help="How many epochs to train for", type=int, default=200)
    parser.add_argument("--ExperimentDir", help="The experiment directory", type=str, default="./Experiments")
    parser.add_argument("--ResumeTraining", help="If to resume training", action="store_true")
    parser.add_argument("--ResumeDir", help="The checkpoint directory from which to resume training", type=str)
    parser.add_argument("--TensorboardDir", help="The events dir to put the new tensorboard in (tensorboard should combine them)", type=str)
    parser.add_argument("--InputSize",help="What size to resize the input to (only support nxn inputs)", type=int, default=256)
    parser.add_argument("--TrainBS", help="Train batch size", type=int, default=32)
    parser.add_argument("--ValidBS", help="Valid batch size", type=int, default=4)
    parser.add_argument("--SaveFreq", help="How often to save the model", type=int, default=10)
    # Data Paths
    args=parser.parse_args()
    print(args)
    main(args)
