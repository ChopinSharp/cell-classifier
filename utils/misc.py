import cv2
import torch
from torchvision import datasets, transforms
import os
from utils import opencv_transforms


def opencv_loader(path):
    return cv2.imread(path, cv2.IMREAD_ANYDEPTH)


def estimate_dataset_mean_and_std(data_dir, input_size):
    """Calculate dataset mean and standard deviation.

    Args:
        data_dir: Top directory of data.
        input_size: Expected input size.

    Returns:
        dataset_mean: Dataset mean.
        dataset_std: Dataset standard deviation.

    """
    # Load all samples into a single dataset
    dataset = torch.utils.data.ConcatDataset([
        datasets.DatasetFolder(
            os.path.join(data_dir, x),
            opencv_loader,
            ['jpg', 'tif'],
            transform=transforms.Compose([
                opencv_transforms.Resize(input_size),
                opencv_transforms.ToTensor()
            ])
        )
        for x in ['train', 'val', 'test']
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
