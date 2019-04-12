import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import warnings
import torch.optim as optim
from transfer_learning import estimate_dataset_mean_and_std
import opencv_transforms
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, t_conv_kernel_size):
        super(_DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, t_conv_kernel_size, 2)
        )

    def forward(self, x):
        return self.block(x)


class Seg(models.SqueezeNet):
    def __init__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(Seg, self).__init__()

        # self.state_dict().update(filter(lambda item: item[0] in self.state_dict(), standard_model.state_dict()))
        self.load_state_dict(torch.load('/home/mwb/.torch/models/squeezenet1_0-a815701f.pth'))

        for param in self.parameters():
            param.requires_grad = False

        del self.classifier

        self.decoder = nn.ModuleList([              # 512 * 13
            nn.ConvTranspose2d(512, 256, 3, 2),     # 256 * 27 + 256 * 27 [7]
            _DecoderBlock(512, 256, 128, 2),
            _DecoderBlock(256, 128, 64, 3),
            _DecoderBlock(160, 80, 40, 8),
            # nn.Sequential([
            #     nn.Conv2d(512, 256, 3, 1, 1),           # 256 * 27
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(inplace=True),
            #     nn.ConvTranspose2d(256, 128, 2, 2)     # 128 * 54 + 128 * 54 [4]
            # ]),
            # nn.Sequential([
            #     nn.Conv2d(256, 128, 3, 1, 1),           # 128 * 54
            #     nn.BatchNorm2d(128),
            #     nn.ReLU(inplace=True),
            #     nn.ConvTranspose2d(128, 64, 3, 2)      # 64 * 109 + 96 * 109 [1]
            # ]),
            # nn.Sequential([
            #     nn.Conv2d(160, 80, 3, 1, 1),            # 80 * 109
            #     nn.BatchNorm2d(80),
            #     nn.ReLU(inplace=True),
            #     nn.ConvTranspose2d(80, 40, 8, 2)       # 40 * 224
            # ]),
            nn.Conv2d(40, 4, 3, 1, 1)                   # 1 * 224
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


def opencv_loader(path):
    return cv2.imread(path, cv2.IMREAD_ANYDEPTH)


class SegImgFolder(datasets.DatasetFolder):
    def __init__(self, root, linked_transforms=None):
        super(SegImgFolder, self).__init__(root, opencv_loader, ['.jpg', '.tif'])
        self.linked_transforms = linked_transforms

    def __getitem__(self, index):
        img, lbl = super(SegImgFolder, self).__getitem__(index)
        img = opencv_transforms.Resize(224)(img)
        _, lbl = cv2.threshold(img, 0, lbl + 1, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        if self.linked_transforms is not None:
            img, lbl = self.linked_transforms(img, lbl)

        img = opencv_transforms.ToTensor()(img)
        lbl = torch.from_numpy(np.ascontiguousarray(lbl))
        return img, lbl


def visualize_dataset(root='data0229/val'):
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
        count  += 1
        if count == 20:
            break



def create_dataloaders(data_dir, batch_size):
    """
    Create datasets and dataloaders.
    :param data_dir: Top directory of data.
    :param batch_size: Batch size.
    :return dataloaders: A dictionary that holds dataloaders for training, validating and testing.
    :return dataset_mean: Estimated mean of dataset.
    :return dataset_std: Estimated standard deviation of dataset.
    """

    input_size = 224

    # Get dataset mean and std
    dataset_mean, dataset_std = estimate_dataset_mean_and_std(data_dir, input_size)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            opencv_transforms.RandomRotation(45),
            opencv_transforms.RandomResizedCrop(input_size),
            opencv_transforms.RandomHorizontalFlip(),
            opencv_transforms.RandomVerticalFlip(),
            opencv_transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'val': transforms.Compose([
            opencv_transforms.Resize(input_size),
            opencv_transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'test': transforms.Compose([
            opencv_transforms.Resize(input_size),
            opencv_transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])
    }

    # Create training and validation datasets
    image_datasets = {
        x: datasets.DatasetFolder(
            os.path.join(data_dir, x),
            opencv_loader,
            ['jpg', 'tif'],
            transform=data_transforms[x]
        )
        for x in ['train', 'val', 'test']
    }

    # Create training and validation dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
        for x in ['train', 'val', 'test']
    }

    return dataloaders, dataset_mean, dataset_std




def train():
    model = Seg()

    # Initialization
    for param in model.decoder.parameters():
        nn.init.xavier_normal_(param)

    # Set up criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.decoder.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
        amsgrad=False
    )

    # Prepare dataset
    for i in range(50):

        pass


if __name__ == '__main__':

    visualize_dataset()

    print('done')
