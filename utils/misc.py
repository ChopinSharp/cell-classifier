import cv2
import torch
from torchvision import datasets, transforms
import os
from visdom import Visdom
import numpy as np

__all__ = ['opencv_loader', 'estimate_dataset_mean_and_std', 'using_device', 'using_port', 'VisdomBoard',
           'expand_subdir']


# Device to use in training
using_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Port on which Visdom run
using_port = 2337


def opencv_loader(path):
    return cv2.imread(path, cv2.IMREAD_ANYDEPTH)


def expand_subdir(folder):
    return [os.path.join(folder, name) for name in os.listdir(folder)]


def estimate_dataset_mean_and_std(data_dirs, input_size=None):
    """Calculate dataset mean and standard deviation.

    Args:
        data_dirs: List of image folders to estimate.
        input_size: Expected input size.

    Returns:
        dataset_mean: Dataset mean.
        dataset_std: Dataset standard deviation.

    """
    # Load all samples into a single dataset
    if input_size:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    dataset = torch.utils.data.ConcatDataset([
        datasets.ImageFolder(data_dir, transform=transform)
        for data_dir in data_dirs
    ])

    # Construct loader, trim off remainders
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=5, drop_last=True)
    num_batches = len(dataset) // 10

    # Estimate mean and std
    dataset_mean = torch.zeros(3)
    dataset_std = torch.zeros(3)
    for inputs, _ in iter(loader):
        dataset_mean += inputs.mean(dim=(0, 2, 3))
    dataset_mean /= num_batches
    for inputs, _ in iter(loader):
        dataset_std += torch.mean((inputs - dataset_mean.reshape((1, 3, 1, 1))) ** 2, dim=(0, 2, 3))
    dataset_std = torch.sqrt(dataset_std.div(num_batches))

    return dataset_mean.tolist(), dataset_std.tolist()


class VisdomBoard:
    def __init__(self, port=2333, info='', metric_label='metric', dummy=False):
        if dummy:
            return
        # Construct Visdom obj
        self.viz = Visdom(port=port)
        # Initialize plots
        self.metric_win = self.viz.line(
            X=[np.NaN],
            Y=[np.NaN],
            name='Phase 1',
            opts={
                'title': '[%s] %s' % (info, metric_label),
                'xlabel': 'epoch',
                'ylabel': metric_label,
                'showlegend': True,
                'height': 400,
                'width': 700
            }
        )
        self.loss_win = self.viz.line(
            X=[np.NaN],
            Y=[np.NaN],
            name='Phase 1',
            opts={
                'title': '[%s] Loss' % info,
                'xlabel': 'epoch',
                'ylabel': 'loss',
                'showlegend': True,
                'height': 400,
                'width': 700
            }
        )

    def _update(self, x, metric, loss, phase):
        self.viz.line(
            X=[x],
            Y=[metric],
            name=phase,
            win=self.metric_win,
            update='append'
        )
        self.viz.line(
            X=[x],
            Y=[loss],
            name=phase,
            win=self.loss_win,
            update='append'
        )

    def update_phase_1(self, x, metric, loss):
        self._update(x, metric, loss, 'Phase 1')

    def update_phase_2(self, x, metric, loss):
        self._update(x, metric, loss, 'Phase 2')

    def update_last_phase_1(self, x, metric, loss):
        self.update_phase_1(x, metric, loss)
        self.viz.line(
            X=[x],
            Y=[metric],
            win=self.metric_win,
            name='Phase 2',
            update='append'
        )
        self.viz.line(
            X=[x],
            Y=[loss],
            win=self.loss_win,
            name='Phase 2',
            update='append'
        )
