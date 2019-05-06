from utils.misc import *
from utils.ext_transforms import *
import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import copy
import torch.utils.model_zoo as model_zoo
from main.cell_segmentation import *
import time
from PIL import Image
from visdom import Visdom


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


def load_pretrained_weights(model):
    pretrained_model = None
    if isinstance(model, UNetVggVar):
        pretrained_model = model_zoo.load_url(model_urls['vgg11_bn'])
    elif isinstance(model, UNetVgg):
        pretrained_model = model_zoo.load_url(model_urls['vgg11_bn'])
    elif isinstance(model, SegNetVgg):
        pretrained_model = model_zoo.load_url(model_urls['vgg16_bn'])
    elif isinstance(model, LinkNetRes):
        pretrained_model = model_zoo.load_url(model_urls['resnet18'])
        pretrained_model.pop('fc.weight')
        pretrained_model.pop('fc.bias')
        model.features.load_state_dict(pretrained_model)
        return
    elif isinstance(model, LinkNetSqueeze):
        pretrained_model = model_zoo.load_url(model_urls['squeezenet1_0'])

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
    if isinstance(model, UNetVggVar):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2')  # and int(name.split('.')[0]) >= 8) # 15
    elif isinstance(model, UNetVgg):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2')  # and int(name.split('.')[0]) >= 8) # 15
    elif isinstance(model, SegNetVgg):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2')  # and int(name.split('.')[0]) >= 14) # 24
    elif isinstance(model, LinkNetRes):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2')
    elif isinstance(model, LinkNetSqueeze):
        for name, param in model.features.named_parameters():
            param.requires_grad = (phase == 'Phase 2')
    else:
        raise Exception('Model type is not supported.')


class SegmentationImageFolder(torch.utils.data.Dataset):
    def __init__(self, data_dir, anno_dir, ext_transforms=None, enable_check=True):
        self.data_dir = data_dir
        self.anno_dir = anno_dir
        self.ext_transforms = ext_transforms
        self.data_files = os.listdir(self.data_dir)
        self.anno_files = os.listdir(self.anno_dir)
        assert len(self.data_files) == len(self.anno_files), 'Invalid dataset for segmentation'
        self.data_files.sort()
        self.anno_files.sort()
        if enable_check: # extension is neglected
            for data, anno in zip(self.data_files, self.anno_files):
                assert data[:data.rfind('.')] == anno[:anno.rfind('.')], 'Invalid dataset for segmentation'

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        anno_path = os.path.join(self.anno_dir, self.anno_files[idx])
        with open(data_path, 'rb') as f:
            data = Image.open(f).convert('RGB')
        with open(anno_path, 'rb') as f:
            anno = Image.open(f).convert('RGB')
        return self.ext_transforms(data, anno)


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


def create_dataloaders(base_dir, batch_size, img_size=(400, 600)):
    """Create datasets and dataloaders.

    Args:
        base_dir: Top directory of dataset.
        batch_size: Batch size.
        img_size: Original image size.

    Returns:
        dataloaders: A dictionary that holds dataloaders for training, validating and testing.
        dataset_mean: Estimated mean of dataset.
        dataset_std: Estimated standard deviation of dataset.

    """

    scaled_size = get_scaled_size(img_size)
    print('* val and test images are scaled to', scaled_size)

    # Get dataset mean and std
    dataset_mean, dataset_std = estimate_dataset_mean_and_std([os.path.join(base_dir, 'data')])

    # Data augmentation and normalization for training
    # Just normalization for validation
    transforms = {
        'train': ExtCompose([
            ExtColorJitter(brightness=0.5, saturation=0.5),
            ExtRandomRotation(45, resample=Image.BILINEAR),
            ExtRandomCrop(224),
            ExtRandomHorizontalFlip(),
            ExtRandomVerticalFlip(),
            ExtToTensor(),
            ExtNormalize(dataset_mean, dataset_std)
        ]),
        'val': ExtCompose([
            ExtResize(scaled_size),
            ExtToTensor(),
            ExtNormalize(dataset_mean, dataset_std)
        ]),
        'test': ExtCompose([
            ExtResize(scaled_size),
            ExtToTensor(),
            ExtNormalize(dataset_mean, dataset_std)
        ])
    }

    # Create training and validation datasets
    image_datasets = {
        x: SegmentationImageFolder(
            os.path.join(base_dir, 'data', x),
            os.path.join(base_dir, 'anno', x),
            ext_transforms=transforms[x]
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
        inputs = inputs.to(using_device)
        labels = labels.to(using_device)
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


def train_model(model, criterion, metric, optimizers, dataloaders, epochs, verbose=True, viz_info=None, port=using_port):

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
                    inputs = inputs.to(using_device)
                    labels = labels.to(using_device)

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
                if inner_phase == 'val' and epoch_iou >= best_model_info['val_iou']:
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
        # Load best model weights in this phase
        model.load_state_dict(best_model_info['model_dict'])

    # Log
    print('\n* Best val IoU: %(val_iou)f at [%(outer_phase)s] epoch %(epoch)d\n' % best_model_info)

    return val_iou_history, loss_history


def validate_model(model, data_dir, lr_candidates, wd_candidates, epochs, phase_2_lr_ratio=1/10, batch_size=8, verbose=True):
    print('Validating', repr(model), '...')

    # Move model to gpu if possible
    model = model.to(using_device)

    # Set up criterion and metric
    criterion = nn.CrossEntropyLoss()
    metric = IoUMetric()

    # Prepare dataset
    dataloaders, dataset_mean, dataset_std = create_dataloaders(data_dir, batch_size)

    # Validate model
    best_model_info = {'lr': 0., 'wd': 0., 'val_iou': 0., 'model_dict': None, 'timestamp': None}
    since = time.time()
    for lr in lr_candidates:
        for wd in wd_candidates:
            start_time = time.ctime()
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
            viz_info = '{0} lr={1:e} wd={2:e} %{3}'.format(repr(model), lr, wd, start_time)
            val_iou_history, loss_history = train_model(model, criterion, metric, optimizers, dataloaders, epochs,
                                                        verbose=verbose,
                                                        viz_info=viz_info)
            # Save best model information
            this_val_iou = max(val_iou_history)
            if this_val_iou > best_model_info['val_iou']:
                best_model_info['lr'] = lr
                best_model_info['wd'] = wd
                best_model_info['val_iou'] = this_val_iou
                best_model_info['model_dict'] = copy.deepcopy(model.state_dict())
                best_model_info['timestamp'] = start_time
    time_elapsed = time.time() - since
    print('* Validation takes %d min, %d sec' % (time_elapsed // 60, time_elapsed % 60))
    print('* Found best model at lr=%(lr)g, wd=%(wd)g, val_iou=%(val_iou)g\n' % best_model_info)

    # Test model
    test_iou = test_model(model, dataloaders['test'])
    test_result = 'Test results:\n  iou_0: %f\n  iou_1: %f\n  iou_2: %f\n  iou_3: %f\n  iou_avg: %f' % test_iou
    print(test_result)
    Visdom(port=using_port).text(format_html_result(best_model_info['timestamp'], *test_iou))

    # Reload best model
    model.load_state_dict(best_model_info['model_dict'])
    return best_model_info['timestamp']


def main():
    model = LinkNetSqueeze()
    timestamp = validate_model(
        model,
        '../datasets/data0229_seg_enhanced',
        # np.linspace(5e-6, 1e-4, 6),
        # np.linspace(1e-6, 1e-3, 6),
        [4e-5],  # 4e-5
        [2e-5],
        {'Phase 1': 100, 'Phase 2': 150},
        phase_2_lr_ratio=1 / 8,  # 1/8
        batch_size=16
    )
    torch.save(model.state_dict(), os.path.abspath('../results/saved_models/%s %s.pt' % (repr(model), timestamp)))
    print('\ndone')


if __name__ == '__main__':
    main()
