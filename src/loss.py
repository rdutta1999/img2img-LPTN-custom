import math
import torch
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.real_label_val = 1.0
        self.fake_label_val = 0.0    

        self.loss_weight = 1.0    

    def get_reconstruction_loss(self, pred_img, inp_img, loss_wt = 1000):
        mse_loss = torch.nn.MSELoss(reduction='mean')
        return loss_wt * mse_loss(pred_img, inp_img)
    
    def get_gan_loss(self, input, target_is_real, is_disc=False):

        # target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        # target_val = input.new_ones(input.size()) * target_val

        if is_disc:
            if target_is_real:
                loss = -torch.mean(input)
            else:
                loss = torch.mean(input)
        else:
            loss = -torch.mean(input)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
    
    # NEED TO REWRITE GRADIENT PENALTY
    def compute_gradient_penalty_2(D, real_samples, fake_samples):

        # Random weight term for interpolation between real and fake samples
        alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def compute_gradient_penalty(discriminator_model, real_imgs, fake_imgs):
        batch_size = real_imgs.size(0)

        # Random weight term for interpolation between real and fake images
        weight_term = torch.rand((batch_size, 1, 1, 1), device="cuda")

        # Interpolating between real and fake samples using the weight term
        interpolated_imgs = (weight_term * real_imgs + ((1 - weight_term) * fake_imgs)).requires_grad_(True)

        # Pass the interpolated image through the Discriminator
        disc_interpolates = discriminator_model(interpolated_imgs)

        fake = torch.ones((batch_size, 1, 1, 1), device="cuda", requires_grad = False)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(outputs = disc_interpolates,
                                        inputs = interpolated_imgs,
                                        grad_outputs = fake,
                                        create_graph = True,
                                        # retain_graph = True,
                                    )[0]
        
        print("Gradient Penalties: ")
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
        return gradient_penalty
        