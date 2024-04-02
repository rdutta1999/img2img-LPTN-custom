from torch import nn as nn

# def discriminator_block(in_filters, out_filters, normalization=False):

#     layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
#     layers.append(nn.LeakyReLU(0.2))
#     if normalization:
#         layers.append(nn.InstanceNorm2d(out_filters, affine=True))
#     return layers

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            # *discriminator_block(16, 32),
            # *discriminator_block(32, 64),
            # *discriminator_block(64, 128),
            # *discriminator_block(128, 128),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 54, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)