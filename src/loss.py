import math
import torch
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class PerceptualLoss(nn.Module):

    def __init__(self, layer_weights, vgg_type='vgg19', use_input_norm=True,
                 range_norm=False, perceptual_weight=1.0, style_weight=0., criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(layer_name_list=list(layer_weights.keys()), vgg_type=vgg_type,
                                       use_input_norm=use_input_norm, range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')
        
    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(
                        x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) -
                        self._gram_mat(gt_features[k]),
                        p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(
                        self._gram_mat(x_features[k]),
                        self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss


    def _gram_mat(self, x):
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'standard':
            self.loss = None
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'standard':
            if is_disc:
                if target_is_real:
                    loss = -torch.mean(input)
                else:
                    loss = torch.mean(input)
            else:
                loss = -torch.mean(input)
        elif self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

# def r1_penalty(real_pred, real_img):
#     """R1 regularization for discriminator. The core idea is to
#         penalize the gradient on real data alone: when the
#         generator distribution produces the true data distribution
#         and the discriminator is equal to 0 on the data manifold, the
#         gradient penalty ensures that the discriminator cannot create
#         a non-zero gradient orthogonal to the data manifold without
#         suffering a loss in the GAN game.

#         Ref:
#         Eq. 9 in Which training methods for GANs do actually converge.
#         """
#     grad_real = autograd.grad(
#         outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
#     grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
#     return grad_penalty


# def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
#     noise = torch.randn_like(fake_img) / math.sqrt(
#         fake_img.shape[2] * fake_img.shape[3])
#     grad = autograd.grad(
#         outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
#     path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

#     path_mean = mean_path_length + decay * (
#         path_lengths.mean() - mean_path_length)

#     path_penalty = (path_lengths - path_mean).pow(2).mean()

#     return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def compute_gradient_penalty(D, real_samples, fake_samples):

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


class CustomLoss(nn.Module):
    def __init__(self):
        self.real_label_val = 1.0
        self.fake_label_val = 0.0    

        self.loss_weight = 1.0    

    def reconstruction_loss(self, pred_img, inp_img, loss_wt = 1.0):
        mse_loss = torch.nn.MSELoss()
        return loss_wt * mse_loss(pred_img, inp_img)
    
    def gan_loss(self, input, target_is_real, is_disc=False):

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
    
    def compute_gradient_penalty():
        pass