import torch.nn as nn


def generator_conv_block(in_features, out_features, kernel_size, stride, padding, activation, use_dropout=False):
    layers = [
        nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=False),
        nn.InstanceNorm2d(out_features),
        activation
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            generator_conv_block(in_features=in_features, out_features=in_features, kernel_size=3, stride=1, padding=1,
                                 activation=nn.ReLU()),
            generator_conv_block(in_features=in_features, out_features=in_features, kernel_size=3, stride=1, padding=1,
                                 activation=nn.Identity(), use_dropout=True)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = generator_conv_block(in_features=3, out_features=64, kernel_size=7, stride=1, padding=3,
                                            activation=nn.ReLU())
        self.downsample = nn.Sequential(
            generator_conv_block(in_features=64, out_features=128, kernel_size=3, stride=2, padding=1,
                                 activation=nn.ReLU()),
            generator_conv_block(in_features=128, out_features=256, kernel_size=3, stride=2, padding=1,
                                 activation=nn.ReLU())
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(9)])
        self.upsample = nn.Sequential(
            generator_conv_block(in_features=256, out_features=128, kernel_size=3, stride=1, padding=1,
                                 activation=nn.ReLU()),
            nn.Upsample(scale_factor=2),
            generator_conv_block(in_features=128, out_features=64, kernel_size=3, stride=1, padding=1,
                                 activation=nn.ReLU()),
            nn.Upsample(scale_factor=2)
        )
        self.final = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.initial(x)
        x = self.downsample(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.final(x)
        return self.tanh(x)
