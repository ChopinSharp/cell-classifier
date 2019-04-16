from utils import *
import torch
import torch.nn as nn
from torchvision import models, datasets
import warnings
import torch.optim as optim
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import time


def initialize_weights(*model_list):
    for model in model_list:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, t_conv_kernel_size):
        super(_DecoderBlock, self).__init__()
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


class UnetSqueeze(models.SqueezeNet):
    def __init__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(UnetSqueeze, self).__init__()

        # self.state_dict().update(filter(lambda item: item[0] in self.state_dict(), standard_model.state_dict()))
        self.load_state_dict(torch.load(os.path.expanduser('~/.torch/models/squeezenet1_0-a815701f.pth')))

        for param in self.parameters():
            param.requires_grad = False

        del self.classifier

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 3, 2),
            _DecoderBlock(512, 256, 128, 2),
            _DecoderBlock(256, 128, 64, 3),
            _DecoderBlock(160, 80, 40, 8),
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


# Copied from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


vgg_config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class _DecoderBasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_DecoderBasicBlock, self).__init__()
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


class UnetVgg(nn.Module):
    def __init__(self, pretrained=True):
        super(UnetVgg, self).__init__()

        # Construct and load vgg feature extracter
        self.features = make_layers(vgg_config['A'], batch_norm=True)
        if pretrained:
            vgg11_bn_pretrained = torch.load(os.path.expanduser('~/.torch/models/vgg11_bn-6002323d.pth'))
            self.features.state_dict().update({
                k: v for k, v in vgg11_bn_pretrained.items() if k.split('.')[0] == 'features'
            })

        # Construct decoder
        self.decoder = nn.ModuleList([
            _DecoderBasicBlock(512, 1024, 512),
            _DecoderBasicBlock(1024, 512, 512),
            _DecoderBasicBlock(1024, 512, 256),
            _DecoderBasicBlock(512, 256, 128),
            _DecoderBasicBlock(256, 128, 64),
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


class SegImgFolder(datasets.DatasetFolder):
    def __init__(self, root, linked_transforms=None):
        super(SegImgFolder, self).__init__(root, opencv_loader, ['.jpg', '.tif'])
        self.linked_transforms = linked_transforms

    def __getitem__(self, index):
        img, lbl = super(SegImgFolder, self).__getitem__(index)
        _, lbl = cv2.threshold(img, 0, lbl + 1, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        if self.linked_transforms is not None:
            img, lbl = self.linked_transforms(img, lbl)

        return img, lbl


class IOUMetric:
    @staticmethod
    def calculate_iou(gt, lbl, cls):
        or_area = np.logical_or(gt == cls, lbl == cls).sum()
        if or_area == 0:
            return 1.
        return np.logical_and(gt == cls, lbl == cls).sum() / or_area

    def __call__(self, gt, lbl):
        gt = gt.detach().numpy()
        lbl = lbl.detach().numpy()
        # print(gt.shape, lbl.shape)
        iou_0 = self.calculate_iou(gt, lbl, 0)
        iou_1 = self.calculate_iou(gt, lbl, 1)
        iou_2 = self.calculate_iou(gt, lbl, 2)
        iou_3 = self.calculate_iou(gt, lbl, 3)
        iou_avg = (iou_0 + iou_1 + iou_2 + iou_3) / 4
        return iou_0, iou_1, iou_2, iou_3, iou_avg


def visualize_result(root='data0229/val'):
    dataset = SegImgFolder(
        root,
        linked_transforms=opencv_transforms.ExtCompose([
            # opencv_transforms.ExtRandomHorizontalFlip(0.99),
            # opencv_transforms.ExtRandomVerticalFlip(0.99),
            # opencv_transforms.ExtRandomRotation(90),
            opencv_transforms.ExtRandomResizedCrop(200)
        ])
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    count = 0
    for sample, target in loader:
        img = (255 * sample.detach().numpy()[0].transpose((1, 2, 0))).astype(np.uint8)
        mask = palette[target.detach().numpy()[0]]
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(mask)
        plt.show()
        count += 1
        if count == 20:
            break


def create_dataloaders(data_dir, batch_size):
    """Create datasets and dataloaders.

    Args:
        data_dir: Top directory of data.
        batch_size: Batch size.

    Returns:
        dataloaders: A dictionary that holds dataloaders for training, validating and testing.
        dataset_mean: Estimated mean of dataset.
        dataset_std: Estimated standard deviation of dataset.

    """
    input_size = 224

    # Get dataset mean and std
    dataset_mean, dataset_std = estimate_dataset_mean_and_std(data_dir, input_size)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': opencv_transforms.ExtCompose([
            opencv_transforms.ExtRandomRotation(20),
            opencv_transforms.ExtRandomResizedCrop(input_size, (0.8, 1.0)),
            opencv_transforms.ExtRandomHorizontalFlip(),
            opencv_transforms.ExtRandomVerticalFlip(),
            opencv_transforms.ExtToTensor(),
            opencv_transforms.ExtNormalize(dataset_mean, dataset_std)
        ]),
        'val': opencv_transforms.ExtCompose([
            opencv_transforms.ExtResize(input_size),
            opencv_transforms.ExtToTensor(),
            opencv_transforms.ExtNormalize(dataset_mean, dataset_std)
        ]),
        'test': opencv_transforms.ExtCompose([
            opencv_transforms.ExtResize(input_size),
            opencv_transforms.ExtToTensor(),
            opencv_transforms.ExtNormalize(dataset_mean, dataset_std)
        ])
    }

    # Create training and validation datasets
    image_datasets = {
        x: SegImgFolder(
            os.path.join(data_dir, x),
            linked_transforms=data_transforms[x]
        )
        for x in ['train', 'val', 'test']
    }

    # Create training and validation dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
        for x in ['train', 'val', 'test']
    }

    return dataloaders, dataset_mean, dataset_std


def test_model(model, loader):
    model.eval()
    metric = IOUMetric()
    running_iou_0 = 0.
    running_iou_1 = 0.
    running_iou_2 = 0.
    running_iou_3 = 0.
    running_iou_avg = 0.

    for inputs, labels in loader:
        with torch.no_grad():
            outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        iou_0, iou_1, iou_2, iou_3, iou_avg = metric(labels, preds)
        running_iou_0 += iou_0 * inputs.size(0)
        running_iou_1 += iou_1 * inputs.size(0)
        running_iou_2 += iou_2 * inputs.size(0)
        running_iou_3 += iou_3 * inputs.size(0)
        running_iou_avg += iou_avg * inputs.size(0)

    test_iou_0 = running_iou_0 / len(loader.dataset)
    test_iou_1 = running_iou_1 / len(loader.dataset)
    test_iou_2 = running_iou_2 / len(loader.dataset)
    test_iou_3 = running_iou_3 / len(loader.dataset)
    test_iou_avg = running_iou_avg / len(loader.dataset)

    return test_iou_0, test_iou_1, test_iou_2, test_iou_3, test_iou_avg


def train_model(num_epochs, verbose=True):

    model = UnetSqueeze()
    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    initialize_weights(model.decoder)

    # Set up criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    metric = IOUMetric()
    optimizer = optim.Adam(
        model.decoder.parameters(),
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
        amsgrad=False
    )

    # Prepare dataset
    dataloaders, dataset_mean, dataset_std = create_dataloaders('../datasets/data0229', 8)

    # Main train loop
    val_iou_history = []
    loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_iou = 0.0
    since = time.time()
    # Train for some epochs
    for epoch in range(num_epochs):
        if verbose:
            print('\n+ Epoch %2d/%d' % (epoch + 1, num_epochs))
            print('+', '-' * 11)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_iou = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Record statistics
                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                # print(outputs.size(), preds.size())
                iou_0, iou_1, iou_2, iou_3, iou_avg = metric(labels, preds)
                running_iou += iou_avg * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_iou = running_iou / len(dataloaders[phase].dataset)
            if verbose:
                print('+ %s Loss: %.4f IoU: %.4f' % (phase, epoch_loss, epoch_iou))

            # Deep copy the best model so far
            if phase == 'val' and epoch_iou > best_val_iou:
                best_val_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_iou_history.append(epoch_iou)
                loss_history.append(epoch_loss)

    # Print out best val acc
    time_elapsed = time.time() - since
    print('\nTraining complete in %.0fh %.0fm %.0fs' %
          (time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print('\nBest val IoU: %f', best_val_iou)

    t_iou_0, t_iou_1, t_iou_2, t_iou_3, t_iou_avg = test_model(model, dataloaders['test'])
    print('Test results:')
    print('iou_0: %f', t_iou_0)
    print('iou_1: %f', t_iou_1)
    print('iou_2: %f', t_iou_2)
    print('iou_3: %f', t_iou_3)
    print('iou_avg: %f', t_iou_avg)

    return model, val_iou_history, loss_history


def inspect_features(features):
    x = torch.zeros(1, 3, 224, 224)

    for idx, layer in enumerate(features):
        x = layer(x)
        print(idx, layer, x.size())


def visualize_model(model, loader):
    model.eval()
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    for sample, label in loader:
        preds = model(sample).argmax(dim=1)
        batch_size = sample.size()[0]
        for i in range(batch_size):
            gd = palette[label.detach().numpy()[i]]
            pd = palette[preds.detach().numpy()[i]]
            img = (255 * sample.detach().numpy()[i].transpose((1, 2, 0))).clip(0, 255).astype(np.uint8)
            plt.subplot(3, batch_size, i + 1)
            plt.title('img')
            plt.imshow(img)
            plt.subplot(3, batch_size, i + 1 + batch_size)
            plt.title('gd')
            plt.imshow(gd)
            plt.subplot(3, batch_size, i + 1 + 2*batch_size)
            plt.title('pd')
            plt.imshow(pd)
        plt.show()


if __name__ == '__main__':

    seg_model, val_hist, loss_hist = train_model(100)
    torch.save(seg_model.state_dict(), '../results/saved_models/squeeze_unet2.0.pt')
    plt.subplot(211)
    plt.title('iou')
    plt.plot(val_hist)
    plt.subplot(212)
    plt.title('loss')
    plt.plot(loss_hist)
    plt.savefig('../results/hist.png')

    # m = UnetSqueeze()
    # m.load_state_dict(torch.load('../results/saved_models/unet_squeeze.pt'))
    # loaders, _, _ = create_dataloaders('../datasets/data0229', 4)
    # visualize_model(m, loaders['test'])

    # model_squ = models.squeezenet1_0()
    # inspect_features(model_squ.features)

    print('done')
