import torch.nn as nn


def discriminator_conv_block(in_filters, out_filters, kernel_size, stride, padding, use_norm=True):
    layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)
