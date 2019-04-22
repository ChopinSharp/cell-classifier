from utils import *
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
import os
import cv2
import numpy as np
import copy
import torch.utils.model_zoo as model_zoo
from main.cell_segmentation import *
import argparse


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


def load_pretrained_weights(model):
    pretrained_model = None
    if isinstance(model, UNetSqueeze):
        pretrained_model = model_zoo.load_url(model_urls['squeezenet1_0'])
    elif isinstance(model, UNetVgg):
        pretrained_model = model_zoo.load_url(model_urls['vgg11_bn'])
    elif isinstance(model, SegNet):
        pretrained_model = model_zoo.load_url(model_urls['vgg16_bn'])

    model.features.load_state_dict({
        k[len('features.'):]: v for k, v in pretrained_model.items() if k.split('.')[0] == 'features'
    })


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


class SegmentationImageFolder(datasets.DatasetFolder):
    def __init__(self, root, linked_transforms=None):
        super(SegmentationImageFolder, self).__init__(root, opencv_loader, ['.jpg', '.tif'])
        self.linked_transforms = linked_transforms

    def __getitem__(self, index):
        img, lbl = super(SegmentationImageFolder, self).__getitem__(index)
        _, lbl = cv2.threshold(img, 0, lbl + 1, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        if self.linked_transforms is not None:
            img, lbl = self.linked_transforms(img, lbl)

        return img, lbl


class IoUMetric:
    @staticmethod
    def calculate_iou(gt, lbl, cls):
        or_area = np.logical_or(gt == cls, lbl == cls).sum()
        if or_area == 0:
            return 1.
        return np.logical_and(gt == cls, lbl == cls).sum() / or_area

    def __call__(self, gt, lbl):
        gt = gt.detach().cpu().numpy()
        lbl = lbl.detach().cpu().numpy()
        # print(gt.shape, lbl.shape)
        iou_0 = self.calculate_iou(gt, lbl, 0)
        iou_1 = self.calculate_iou(gt, lbl, 1)
        iou_2 = self.calculate_iou(gt, lbl, 2)
        iou_3 = self.calculate_iou(gt, lbl, 3)
        iou_avg = (iou_0 + iou_1 + iou_2 + iou_3) / 4
        return iou_0, iou_1, iou_2, iou_3, iou_avg


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
        x: SegmentationImageFolder(
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
    metric = IoUMetric()
    running_iou_0 = 0.
    running_iou_1 = 0.
    running_iou_2 = 0.
    running_iou_3 = 0.
    running_iou_avg = 0.

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
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


def train_model(model, criterion, metric, optimizers, dataloaders, epochs, verbose=True, viz_info=None):

    # Initialize weights
    initialize_weights(model.decoder)
    load_pretrained_weights(model)

    # Book keeping
    val_iou_history = []
    loss_history = []
    best_model_info = {
        'model_dict': copy.deepcopy(model.state_dict()),
        'val_iou': 0.,
        'epoch': 0,
        'half': 'empty'
    }

    # Visdom setup
    viz_board = VisdomBoard(port=2335, info=viz_info, metric_label='IoU', dummy=(not viz_info))
    epoch_idx = 1

    # Train model
    for half in ['Phase 1', 'Phase 2']:
        for param in model.features.parameters():
            param.requires_grad = (half == 'Phase 2')
        for epoch in range(epochs[half]):
            if verbose:
                print('\n+ [%s] Epoch %2d/%d' % (half, epoch + 1, epochs[half]))
                print('+', '-' * 24)

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
                    optimizers[half].zero_grad()

                    # Forward, track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizers[half].step()

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
                if phase == 'val' and epoch_iou > best_model_info['val_iou']:
                    best_model_info['val_iou'] = epoch_iou
                    best_model_info['model_dict'] = copy.deepcopy(model.state_dict())
                    best_model_info['epoch'] = epoch + 1
                    best_model_info['half'] = half

                # Record and visualize training dynamics
                if phase == 'val':
                    val_iou_history.append(epoch_iou)
                    loss_history.append(epoch_loss)
                    if epoch_idx < epochs['Phase 1']:
                        viz_board.update_phase_1(epoch_idx, epoch_iou, epoch_loss)
                    elif epoch_idx == epochs['Phase 1']:
                        viz_board.update_last_phase_1(epoch_idx, epoch_iou, epoch_loss)
                    else:
                        viz_board.update_phase_2(epoch_idx, epoch_iou, epoch_loss)
                    epoch_idx += 1

    # Load best model weights
    model.load_state_dict(best_model_info['model_dict'])
    print('\n* Best val IoU: %(val_iou)f at [%(half)s] epoch %(epoch)d\n' % best_model_info)

    return val_iou_history, loss_history


def validate_model(model, lr_candidates, wd_candidates, epochs, batch_size=8, verbose=True):
    # Move model to gpu if possible
    model = model.to(device)

    # Set up criterion and metric
    criterion = nn.CrossEntropyLoss()
    metric = IoUMetric()

    # Prepare dataset
    dataloaders, dataset_mean, dataset_std = create_dataloaders('../datasets/data0229', batch_size)

    # Validate model
    best_model_info = {'lr': 0., 'wd': 0., 'val_iou': 0., 'model_dict': None}
    for lr in lr_candidates:
        for wd in wd_candidates:
            print("* Tuning lr=%g, wd=%g" % (lr, wd))
            # Construct optimizer
            optimizers = {
                'Phase 1': optim.Adam(
                    model.decoder.parameters(),
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=wd,
                    amsgrad=False
                ),
                'Phase 2': optim.Adam(
                    model.parameters(),
                    lr=lr / 10,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=wd,
                    amsgrad=False
                )
            }
            # Train model
            val_iou_history, loss_history = train_model(model, criterion, metric, optimizers, dataloaders, epochs,
                                                        verbose=verbose, viz_info='lr=%g wd=%g' % (lr, wd))
            # Save best model information
            this_val_iou = max(val_iou_history)
            if this_val_iou > best_model_info['val_iou']:
                best_model_info['lr'] = lr
                best_model_info['wd'] = wd
                best_model_info['val_iou'] = this_val_iou
                best_model_info['model_dict'] = copy.deepcopy(model.state_dict())
    print('* Found best model at lr=%(lr)g, wd=%(wd)g, val_iou=%(val_iou)g\n' % best_model_info)

    # Test model
    t_iou_0, t_iou_1, t_iou_2, t_iou_3, t_iou_avg = test_model(model, dataloaders['test'])
    print('Test results:')
    print('iou_0: %f' % t_iou_0)
    print('iou_1: %f' % t_iou_1)
    print('iou_2: %f' % t_iou_2)
    print('iou_3: %f' % t_iou_3)
    print('iou_avg: %f' % t_iou_avg)

    # Reload best model
    model.load_state_dict(best_model_info['model_dict'])


# def inspect_features(features):
#     x = torch.zeros(1, 3, 224, 224)
#
#     for idx, layer in enumerate(features):
#         x = layer(x)
#         print(idx, layer, x.size())


def visualize_model(model, loader):
    model.eval()
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    for sample, label in loader:
        preds = model(sample).argmax(dim=1)
        batch_size = sample.size()[0]
        for i in range(batch_size):
            gd = palette[label.detach().cpu().numpy()[i]]
            pd = palette[preds.detach().cpu().numpy()[i]]
            img = (255 * sample.detach().cpu().numpy()[i].transpose((1, 2, 0))).clip(0, 255).astype(np.uint8)
            # plt.subplot(3, batch_size, i + 1)
            # plt.title('img')
            # plt.imshow(img)
            # plt.subplot(3, batch_size, i + 1 + batch_size)
            # plt.title('gd')
            # plt.imshow(gd)
            # plt.subplot(3, batch_size, i + 1 + 2*batch_size)
            # plt.title('pd')
            # plt.imshow(pd)
        # plt.show()


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default='UNet-Vgg', type=str, choices=['Unet-Squeeze', 'UNet-Vgg', 'SegNet'])
    # args = parser.parse_args()

    model = UNetVgg()
    validate_model(model, [1e-5, 5e-5], [1e-5, 5e-5], {'Phase 1': 2, 'Phase 2': 10})
    # torch.save(model.state_dict(), '../results/saved_models/squeeze_unet2.0.pt')

    print('\ndone')


if __name__ == '__main__':
    main()

