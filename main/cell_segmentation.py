import torch
import torch.nn as nn
from torchvision import models
import warnings


__all__ = ['UNetSqueeze', 'UNetVgg', 'SegNet']


class _UNetSqueezeBasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, t_conv_kernel_size):
        super(_UNetSqueezeBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, mid_channels, t_conv_kernel_size, 2),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetSqueeze(models.SqueezeNet):
    def __init__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(UNetSqueeze, self).__init__()
            # self.load_state_dict(torch.load(os.path.expanduser('~/.torch/models/squeezenet1_0-a815701f.pth')))

        del self.classifier

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 3, 2),
            _UNetSqueezeBasicBlock(512, 256, 128, 2),
            _UNetSqueezeBasicBlock(256, 128, 64, 3),
            _UNetSqueezeBasicBlock(160, 80, 40, 8),
            nn.Conv2d(40, 4, 3, 1, 1)
        ])

    def forward(self, x):
        activations = []
        for layer in self.features:
            x = layer(x)
            activations.append(x)

        x = self.decoder[0](x)
        x = self.decoder[1](torch.cat((x, activations[7]), dim=1))
        x = self.decoder[2](torch.cat((x, activations[4]), dim=1))
        x = self.decoder[3](torch.cat((x, activations[1]), dim=1))
        x = self.decoder[4](x)

        return x


# Code from torchvision package https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def make_layers(cfg, batch_norm=False, return_indices=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=return_indices)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if return_indices:
        return nn.ModuleList(layers)
    return nn.Sequential(*layers)


vgg_config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class _UNetVggBasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_UNetVggBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, 2, 2)
        )

    def forward(self, x):
        return self.block(x)


class UNetVgg(nn.Module):
    def __init__(self):
        super(UNetVgg, self).__init__()

        # Construct and load vgg feature extracter
        self.features = make_layers(vgg_config['A'], batch_norm=True)

        # Construct decoder
        self.decoder = nn.ModuleList([
            _UNetVggBasicBlock(512, 1024, 512),
            _UNetVggBasicBlock(1024, 512, 512),
            _UNetVggBasicBlock(1024, 512, 256),
            _UNetVggBasicBlock(512, 256, 128),
            _UNetVggBasicBlock(256, 128, 64),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 4, 3, 1, 1)
        ])

    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)

        x = self.decoder[0](x)
        x = self.decoder[1](torch.cat((x, features[27]), dim=1))
        x = self.decoder[2](torch.cat((x, features[20]), dim=1))
        x = self.decoder[3](torch.cat((x, features[13]), dim=1))
        x = self.decoder[4](torch.cat((x, features[6]), dim=1))
        x = self.decoder[5](torch.cat((x, features[2]), dim=1))
        x = self.decoder[6](x)
        x = self.decoder[7](x)

        return x


class _SegNetBasicBlock(nn.Module):
    def __init__(self, conv_num, in_channels, out_channels):
        super(_SegNetBasicBlock, self).__init__()
        self.block = nn.ModuleList([nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)])
        for _ in range(conv_num):
            self.block.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

    def forward(self, x, indices)


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.features = make_layers(vgg_config['D'], batch_norm=True, return_indices=True)
        self.decoder = nn.ModuleList([
            torch.nn.MaxUnpool2d,

        ])

    def forward(self, x):
        pass
