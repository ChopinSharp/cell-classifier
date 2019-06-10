import torch
import torch.nn as nn
from torchvision import models


__all__ = ['UNetVgg', 'UNetVggVar', 'SegNetVgg', 'LinkNetRes', 'UNetVggVar2']


def _make_vgg_encoder_layers(version, batch_norm=True, return_indices=False):
    config = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    }
    assert version in config.keys()
    layers = []
    in_channels = 3
    for v in config[version]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=return_indices)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


class _UNetVggDecoderBasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_UNetVggDecoderBasicBlock, self).__init__()
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

        # Construct and load vgg11_bn encoder
        self.features = nn.Sequential(*_make_vgg_encoder_layers('vgg11'))

        # Construct decoder
        self.decoder = nn.ModuleList([
            _UNetVggDecoderBasicBlock(512, 1024, 512),
            _UNetVggDecoderBasicBlock(1024, 512, 512),
            _UNetVggDecoderBasicBlock(1024, 512, 256),
            _UNetVggDecoderBasicBlock(512, 256, 128),
            _UNetVggDecoderBasicBlock(256, 128, 64),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 4, 3, 1, 1)
            )
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

        return x

    def __repr__(self):
        return self.__class__.__name__


class UNetVggVar(nn.Module):
    """
    Almost the same as UNetVgg, but use addition instead of concatenation when joining main branch with skip branch.
    """

    def __init__(self):
        super(UNetVggVar, self).__init__()

        # Construct and load vgg11_bn encoder
        self.features = nn.Sequential(*_make_vgg_encoder_layers('vgg11'))

        # Construct decoder
        self.decoder = nn.ModuleList([
            _UNetVggDecoderBasicBlock(512, 1024, 512),
            _UNetVggDecoderBasicBlock(512, 512, 512),
            _UNetVggDecoderBasicBlock(512, 512, 256),
            _UNetVggDecoderBasicBlock(256, 256, 128),
            _UNetVggDecoderBasicBlock(128, 128, 64),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 4, 3, 1, 1)
            )
        ])

    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)

        x = self.decoder[0](x)
        x = self.decoder[1](x + features[27])
        x = self.decoder[2](x + features[20])
        x = self.decoder[3](x + features[13])
        x = self.decoder[4](x + features[6])
        x = self.decoder[5](x + features[2])

        return x

    def __repr__(self):
        return self.__class__.__name__


class UNetVggVar2(nn.Module):
    """
    Almost the same as UNetVgg, but use addition instead of concatenation when joining main branch with skip branch.
    """

    def __init__(self):
        super(UNetVggVar2, self).__init__()

        # Construct and load vgg11_bn encoder
        self.features = nn.Sequential(*(_make_vgg_encoder_layers('vgg13')[:21]))

        # Construct decoder
        self.decoder = nn.ModuleList([
            _UNetVggDecoderBasicBlock(256, 512, 256),
            _UNetVggDecoderBasicBlock(256, 256, 128),
            _UNetVggDecoderBasicBlock(128, 128, 64),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 4, 3, 1, 1)
            )
        ])

    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)

        x = self.decoder[0](x)
        x = self.decoder[1](x + features[19])
        x = self.decoder[2](x + features[12])
        x = self.decoder[3](x + features[5])

        return x

    def __repr__(self):
        return self.__class__.__name__


class _SegNetDecoderBasicBlock(nn.Module):
    def __init__(self, conv_num, in_channels, out_channels, last_block=False):
        super(_SegNetDecoderBasicBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        convs = []
        for idx in range(conv_num - 1):
            convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            convs.append(nn.BatchNorm2d(in_channels))
            if not last_block:
                convs.append(nn.ReLU(inplace=True))
        convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        if not last_block:
            convs.append(nn.BatchNorm2d(out_channels))
            convs.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*convs)

    def forward(self, x, index):
        x = self.unpool(x, index)
        return self.convs(x)


class SegNetVgg(nn.Module):
    def __init__(self):
        super(SegNetVgg, self).__init__()
        # Both self.features and self.decoder are of type nn.ModuleList
        self.features = nn.ModuleList(_make_vgg_encoder_layers('vgg16', return_indices=True))
        self.decoder = nn.ModuleList([
            _SegNetDecoderBasicBlock(3, 512, 512),
            _SegNetDecoderBasicBlock(3, 512, 256),
            _SegNetDecoderBasicBlock(3, 256, 128),
            _SegNetDecoderBasicBlock(2, 128, 64),
            _SegNetDecoderBasicBlock(2, 64, 4, last_block=True)
        ])

    def forward(self, x):
        indices = []

        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, index = layer(x)
                indices.append(index)
            else:
                x = layer(x)

        for layer in self.decoder:
            x = layer(x, indices.pop())

        return x

    def __repr__(self):
        return self.__class__.__name__


class LinkNetDecoderBlock(nn.Module):
    def __init__(self, m, n, k=2):
        assert m % 4 == 0, 'm={} is not a multiple of 4'.format(m)
        super(LinkNetDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(m, m // 4, 1, 1),
            nn.BatchNorm2d(m // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m // 4, m // 4, k, 2),
            nn.Conv2d(m // 4, n, 1, 1),
            nn.BatchNorm2d(n),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LinkNetRes(nn.Module):
    """
    Use ResNet18 as encoder, use LinkNet decoder.
    """
    def __init__(self):
        super(LinkNetRes, self).__init__()
        self.features = models.resnet18()
        del self.features.avgpool
        del self.features.fc

        self.decoder = nn.ModuleList([
            LinkNetDecoderBlock(512, 256),
            LinkNetDecoderBlock(256, 128),
            LinkNetDecoderBlock(128, 64),
            nn.Sequential(
                nn.Conv2d(64, 16, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, 1, 1),
                nn.Conv2d(16, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 2, 2),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 4, 2, 2)
            )
        ])

    def forward(self, x):
        # Encoding
        skips = []
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        # skips.append(x)
        x = self.features.maxpool(x)
        x = self.features.layer1(x)
        skips.append(x)
        x = self.features.layer2(x)
        skips.append(x)
        x = self.features.layer3(x)
        skips.append(x)
        x = self.features.layer4(x)

        # Decoding
        x = self.decoder[0](x)
        for i in range(1, 4):
            x = self.decoder[i](x + skips.pop())
        x = self.decoder[4](x)
        return x

    def __repr__(self):
        return self.__class__.__name__


def test():
    model = UNetVggVar2()
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.size())


if __name__ == '__main__':
    test()

