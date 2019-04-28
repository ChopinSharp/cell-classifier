"""
Script to test the impact of depth of Squeeze Net 1.0 on model accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.utils.model_zoo as model_zoo
import warnings
from utils import *
from scripts.train_classifier import create_dataloaders, train_model
from visdom import Visdom
import numpy as np

def get_activation_size(model, verbose=False):
    activation_size = []
    x = torch.zeros(1, 3, 224, 224)
    for idx, layer in enumerate(model):
        x = layer(x)
        activation_size.append({'channel': x.size(1), 'size': x.size(2)})
        if verbose:
            print(idx, x.size())
    return activation_size


class SqueezeNetOfDepth(models.SqueezeNet):
    """ Squeeze Net of specific depth with pretrained weights. """

    def __init__(self, depth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(SqueezeNetOfDepth, self).__init__()

        self.depth = depth

        self.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/squeezenet1_0-a815701f.pth'))

        activation_size = get_activation_size(self.features)
        for param in self.parameters():
            param.requires_grad = False
        self.classifier[1] = nn.Conv2d(activation_size[depth]['channel'], 3, kernel_size=(1, 1), stride=(1, 1))
        self.classifier[3] = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = 3

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if idx > self.depth:
                break
            x = layer(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def main():

    # Create dataloaders
    print('Loading dataset ...\n')
    dataloaders, dataset_mean, dataset_std = create_dataloaders('../datasets/data0229', 224, 4)
    print('+ Dataset mean:', dataset_mean[0])
    print('+ Dataset standard deviation:', dataset_std[0])
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    print('+ %(train)d samples for training, %(val)d for validation, %(test)d for test.\n' % dataset_sizes)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    # Train Squeeze Net of different depths
    val_acc_of_depth = []
    for depth in range(13):
        print('At depth', depth)

        # Initialize the model for this run
        model = SqueezeNetOfDepth(depth)

        # Send the model to GPU
        model = model.to(device)

        # Setup optimizer
        optimizer = optim.Adam(
            model.classifier[1].parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5,
            amsgrad=False
        )

        # Train and evaluate
        model, val_acc_history, loss_history = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=30,
            is_inception=False
        )

        # Print out best val acc
        best_val_acc = max(val_acc_history)
        print('depth: %d, best val acc: %f' % (depth, best_val_acc))
        val_acc_of_depth.append(best_val_acc)

    # Visualize result
    viz = Visdom(port=2337, env='测试模型深度对精度的影响')
    viz.line(
        X=np.arange(1, 14),
        Y=val_acc_of_depth,
        name='best val acc',
        opts={
            'title': '模型深度对精度的影响',
            'xlabel': 'depth',
            'ylabel': 'acc',
            'showlegend': True,
            'height': 400,
            'width': 700
        }
    )


if __name__ == '__main__':
    main()
