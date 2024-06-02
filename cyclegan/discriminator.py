import torch.nn as nn


def discriminator_conv_block(in_features, out_features, kernel_size, stride, padding, use_norm=True):
    layers = [nn.Conv2d(in_features, out_features, kernel_size, stride, padding)]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_features))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, in_features=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            discriminator_conv_block(in_features=in_features, out_features=64, kernel_size=4, stride=2, padding=1,
                                     use_norm=False),
            discriminator_conv_block(in_features=64, out_features=128, kernel_size=4, stride=2, padding=1),
            discriminator_conv_block(in_features=128, out_features=256, kernel_size=4, stride=2, padding=1),
            discriminator_conv_block(in_features=256, out_features=512, kernel_size=4, stride=1, padding=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
