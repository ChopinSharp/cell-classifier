import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from transfer_learning import create_dataloaders, device, train_model, test_model
import warnings


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
    def __init__(self, depth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(SqueezeNetOfDepth, self).__init__()
            standard_model = models.squeezenet1_0(True)

        self.depth = depth

        # self.state_dict().update(filter(lambda item: item[0] in self.state_dict(), standard_model.state_dict()))
        self.load_state_dict(torch.load('/home/mwb/.torch/models/squeezenet1_0-a815701f.pth'))

        activation_size = get_activation_size(standard_model.features)
        for param in self.parameters():
            param.requires_grad = False
        self.classifier[1] = nn.Conv2d(activation_size[depth]['channel'], 3, kernel_size=(1, 1), stride=(1, 1))
        self.classifier[3] = nn.AdaptiveAvgPool2d((1, 1))
        # nn.AvgPool2d(kernel_size=activation_size[depth]['size'], stride=1, padding=0)
        self.num_classes = 3

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if idx > self.depth:
                break
            x = layer(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def test_depth():
    # Create dataloaders
    print('Loading dataset ...\n')
    dataloaders, dataset_mean, dataset_std = \
        create_dataloaders('data0229', 224, 4)
    print('+ Dataset mean:', dataset_mean[0])
    print('+ Dataset standard deviation:', dataset_std[0])
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    print('+ %(train)d samples for training, %(val)d for validation, %(test)d for test.\n' % dataset_sizes)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

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

        # Print val acc for this set of hyper-parameters
        this_val_acc = max(val_acc_history)

        # Test model
        test_acc, _ = test_model(model, dataloaders['test'])
        print('\n* depth=%d, max_val_acc=%.4f, test_acc=%.4f\n\n' % (depth, this_val_acc, test_acc))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standard_model = models.squeezenet1_0(True)
        get_activation_size(standard_model.features, True)
