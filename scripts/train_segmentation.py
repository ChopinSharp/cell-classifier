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
import time
import matplotlib.pyplot as plt
from PIL import Image


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
    elif isinstance(model, SegNetVgg16):
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


def set_parameter_requires_grad(model, phase):
    if isinstance(model, UNetSqueeze):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2' and int(name.split('.')[0]) >= 7)
    elif isinstance(model, UNetVgg):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2')# and int(name.split('.')[0]) >= 8) # 15
    elif isinstance(model, SegNetVgg16):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2')# and int(name.split('.')[0]) >= 14) # 24
    else:
        raise Exception('Model type is not supported.')


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


def get_scaled_size(size):
    # Scale both height and width to multiples of 32
    return size[0] // 32 * 32, size[1] // 32 * 32


def create_dataloaders(data_dir, batch_size, img_size=(400, 600), verbose=True):
    """Create datasets and dataloaders.

    Args:
        data_dir: Top directory of data.
        batch_size: Batch size.

    Returns:
        dataloaders: A dictionary that holds dataloaders for training, validating and testing.
        dataset_mean: Estimated mean of dataset.
        dataset_std: Estimated standard deviation of dataset.

    """

    scaled_size = get_scaled_size(img_size)
    print('* val and test images are scaled to', scaled_size)

    # Get dataset mean and std
    dataset_mean, dataset_std = estimate_dataset_mean_and_std(data_dir)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': opencv_transforms.ExtCompose([
            opencv_transforms.ExtRandomRotation(45),
            opencv_transforms.ExtRandomCrop(224),
            opencv_transforms.ExtRandomHorizontalFlip(),
            opencv_transforms.ExtRandomVerticalFlip(),
            opencv_transforms.ExtToTensor(),
            opencv_transforms.ExtNormalize(dataset_mean, dataset_std)
        ]),
        'val': opencv_transforms.ExtCompose([
            opencv_transforms.ExtResize(scaled_size),
            opencv_transforms.ExtToTensor(),
            opencv_transforms.ExtNormalize(dataset_mean, dataset_std)
        ]),
        'test': opencv_transforms.ExtCompose([
            opencv_transforms.ExtResize(scaled_size),
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
    batch_sizes = {'train': batch_size, 'val': 4, 'test': 4}
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], shuffle=True, num_workers=1)
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


def train_model(model, criterion, metric, optimizers, dataloaders, epochs, verbose=True, viz_info=None, port=2337):

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
        'outer_phase': 'empty'
    }

    # Visdom setup
    viz_board = VisdomBoard(port=port, info=viz_info, metric_label='IoU', dummy=(not viz_info))
    epoch_idx = 1

    # Train model
    for outer_phase in ['Phase 1', 'Phase 2']:
        set_parameter_requires_grad(model, outer_phase)
        for epoch in range(epochs[outer_phase]):
            if verbose:
                print('\n+ [%s] Epoch %2d/%d' % (outer_phase, epoch + 1, epochs[outer_phase]))
                print('+', '-' * 24)

            # Each epoch has a training and validation phase
            for inner_phase in ['train', 'val']:
                if inner_phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_iou = 0.0

                # Iterate over data.
                for inputs, labels in dataloaders[inner_phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizers[outer_phase].zero_grad()

                    # Forward, track history if only in train
                    with torch.set_grad_enabled(inner_phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if inner_phase == 'train':
                        loss.backward()
                        optimizers[outer_phase].step()

                    # Record statistics
                    running_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    # print(outputs.size(), preds.size())
                    iou_0, iou_1, iou_2, iou_3, iou_avg = metric(labels, preds)
                    running_iou += iou_avg * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[inner_phase].dataset)
                epoch_iou = running_iou / len(dataloaders[inner_phase].dataset)
                if verbose:
                    print('+ %s Loss: %.4f IoU: %.4f' % (inner_phase, epoch_loss, epoch_iou))

                # Deep copy the best model so far
                if inner_phase == 'val' and epoch_iou > best_model_info['val_iou']:
                    best_model_info['val_iou'] = epoch_iou
                    best_model_info['model_dict'] = copy.deepcopy(model.state_dict())
                    best_model_info['epoch'] = epoch + 1
                    best_model_info['outer_phase'] = outer_phase

                # Record and visualize training dynamics
                if inner_phase == 'val':
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
    print('\n* Best val IoU: %(val_iou)f at [%(outer_phase)s] epoch %(epoch)d\n' % best_model_info)

    return val_iou_history, loss_history


def validate_model(model, lr_candidates, wd_candidates, epochs, phase_2_lr_ratio=1/10, batch_size=8, verbose=True):
    print('Validating', repr(model), '...')

    # Move model to gpu if possible
    model = model.to(device)

    # Set up criterion and metric
    criterion = nn.CrossEntropyLoss()
    metric = IoUMetric()

    # Prepare dataset
    dataloaders, dataset_mean, dataset_std = create_dataloaders('../datasets/data0229', batch_size)

    # Validate model
    best_model_info = {'lr': 0., 'wd': 0., 'val_iou': 0., 'model_dict': None}
    since = time.time()
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
                    lr=lr * phase_2_lr_ratio,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=wd,
                    amsgrad=False
                )
            }
            # Train model
            val_iou_history, loss_history = train_model(model, criterion, metric, optimizers, dataloaders, epochs,
                                                        verbose=verbose, viz_info='%s lr=%g wd=%g' % (repr(model),lr, wd))
            # Save best model information
            this_val_iou = max(val_iou_history)
            if this_val_iou > best_model_info['val_iou']:
                best_model_info['lr'] = lr
                best_model_info['wd'] = wd
                best_model_info['val_iou'] = this_val_iou
                best_model_info['model_dict'] = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('* Validation takes %d min, %d sec' % (time_elapsed // 60, time_elapsed % 60))
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


def get_merged_tif(image_url, verbose=False):
    image_stack = Image.open(image_url)
    merged_image = np.zeros(image_stack.size, dtype=np.float32)
    print(image_stack.n_frames, 'frames in', image_url)
    for i in range(image_stack.n_frames):
        image_stack.seek(i)
        merged_image += np.fromstring(image_stack.tobytes(), dtype=np.uint16).reshape(*image_stack.size)
    merged_image /= 65536
    merged_image = merged_image.clip(0, 1)

    return np.concatenate([merged_image[:, :, np.newaxis]] * 3, axis=2)


def infer(model, folder_url):
    files = os.listdir(folder_url)
    total = len(files)
    fig = plt.figure(figsize=[8, 6 * total])
    model.eval()
    for idx, name in enumerate(files):
        image_url = os.path.join(folder_url, name)
        # float_image = get_merged_tif(image_url, verbose=True)
        image = (cv2.imread(image_url, cv2.IMREAD_ANYDEPTH) / 256).astype(np.uint8)
        image_enhanced = cv2.equalizeHist(image)
        float_image = image_enhanced.astype(np.float32) / 256
        float_image = np.concatenate([float_image[:, :, np.newaxis]] * 3, axis=2)
        inputs = torch.from_numpy(float_image.transpose((2, 0, 1)))
        mean = inputs.mean()
        std = inputs.std()
        print('mean:', mean, 'std:', std)
        inputs = inputs.sub(mean).div(std).unsqueeze(0).float()

        preds = model(inputs).argmax(dim=1)
        palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        pd = palette[preds.detach().cpu().numpy()[0]]

        ax = fig.add_subplot(total, 2, 2 * idx + 1)
        ax.set_title('Image')
        ax.imshow(float_image)
        ax = fig.add_subplot(total, 2, 2 * idx + 2)
        ax.set_title('Pred.')
        ax.imshow(pd)
        fig.savefig('result.jpg')


def visualize_model(model, loader):
    model.eval()
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    os.makedirs('tmp', exist_ok=True)
    counter = 1
    for idx, (sample, label) in enumerate(loader):
        preds = model(sample).argmax(dim=1)
        batch_size = sample.size()[0]
        fig = plt.figure(figsize=[19.2, 10.8])
        for i in range(batch_size):
            gd = palette[label.detach().cpu().numpy()[i]]
            pd = palette[preds.detach().cpu().numpy()[i]]
            img = (255 * sample.detach().cpu().numpy()[i].transpose((1, 2, 0))).clip(0, 255).astype(np.uint8)
            ax = fig.add_subplot(3, batch_size, i + 1)
            ax.set_title('img')
            ax.imshow(img)
            ax = fig.add_subplot(3, batch_size, i + 1 + batch_size)
            ax.set_title('gd')
            ax.imshow(gd)
            ax = fig.add_subplot(3, batch_size, i + 1 + 2*batch_size)
            ax.set_title('pd')
            ax.imshow(pd)
        plt.savefig('tmp/%d.jpg' % counter)
        counter += 1


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default='UNet-Vgg', type=str, choices=['Unet-Squeeze', 'UNet-Vgg', 'SegNet'])
    # args = parser.parse_args()

    model = UNetVgg()
    validate_model(
        model,
        # np.linspace(5e-6, 1e-4, 6),
        # np.linspace(1e-6, 1e-3, 6),
        [4e-5],
        [2e-5],
        {'Phase 1': 100, 'Phase 2': 100},
        phase_2_lr_ratio=1 / 8,
        batch_size=16
    )
    torch.save(model.state_dict(), os.path.abspath('../results/saved_models/UNetVgg.pt'))
    # model2 = SegNetVgg16()
    # validate_model(
    #     model2,
    #     np.linspace(5e-6, 1e-4, 6),
    #     np.linspace(1e-7, 1e-4, 6),
    #     {'Phase 1': 50, 'Phase 2': 50},
    #     phase_2_lr_ratio=1 / 5
    # )
    # torch.save(model.state_dict(), 'results/saved_models/SegNetVgg16.pt')

    print('\ndone')


if __name__ == '__main__':
    model = UNetVgg()
    model.load_state_dict(torch.load('../results/saved_models/UNetVgg.pt'))
    infer(model, '../datasets/segtest0426')
    # #
    # # loaders, _, _ = create_dataloaders('../datasets/data0229', 4)
    # # visualize_model(model, loaders['test'])
    # image = cv2.imread('../datasets/segtest0424/CZ22405 Z ASIX-WOUND-001_w0001.tif', )
    #
    # plt.imshow(image.astype(np.float) / 65536)
    # plt.show()
    # main()
