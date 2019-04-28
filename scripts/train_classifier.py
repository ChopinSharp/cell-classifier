from utils import *
from main.cell_classifier import *
import torch
from torch import nn, optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import copy
import time
import os
import numpy as np
from visdom import Visdom


def create_dataloaders(data_dir, input_size, batch_size):
    """Create datasets and dataloaders.

    Args:
        data_dir: Top directory of data.
        input_size: Expected input size.
        batch_size: Batch size.

    Returns:
        dataloaders: A dictionary that holds dataloaders for training, validating and testing.
        dataset_mean: Estimated mean of dataset.
        dataset_std: Estimated standard deviation of dataset.

    """
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


def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False, verbose=True):
    """Train and validate model with given dataloaders, loss and optimizer for some number of epochs.
    Log the training dynamics and return the model with best val acc.

    Args:
        model: Initialized model to be trained.
        dataloaders: Dataloaders to train and validate on.
        criterion: Loss function.
        optimizer: Optimizer to use with fixed hyper-parameters.
        num_epochs: Number of epochs to train.
        is_inception: Using Inception model or not.
        verbose: Print training info every epoch or not.

    Returns:
        model: Model with best val acc. Changed inplace, actually unnecessary to return.
        val_acc_history: History of val acc over epochs.
        loss_history: History of loss over epochs

    """
    # Record training stats
    # since = time.time()
    val_acc_history = []
    loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

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
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Record statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            if verbose:
                print('+ %s Loss: %.4f Acc: %.4f' % (phase, epoch_loss, epoch_acc))

            # Deep copy the best model so far
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, loss_history


def test_model(model, dataloader, is_inception=False):
    """Test model.

    Args:
        model: Model to test.
        dataloader: Dataloader of test set.
        is_inception: Using Inception model or not.

    Returns:
        test_acc: Test accuracy.
        avg_conf: Average confidence of the prediction.

    """
    running_corrects = 0
    sum_conf = 0.0

    # Iterate over data.
    for inputs, labels in iter(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward, track history if only in train
        with torch.no_grad():
            if is_inception:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)

        # statistics
        running_corrects += torch.sum(preds == labels.data).item()
        sum_conf += probs[torch.arange(probs.size()[0]), preds].sum().item()

    test_acc = running_corrects / len(dataloader.dataset)
    avg_conf = sum_conf / len(dataloader.dataset)

    return test_acc, avg_conf


def validate_model(data_dir, script_dir='../results/saved_scripts', model_dir='../results/saved_models',
                   model_name='squeezenet', num_classes=3, batch_size=8, num_epochs=50, feature_extract=True,
                   learning_rates=(1e-3,), weight_decays=(1e-5,), viz=None):
    """Train and validate chosen model with set(s) of hyper-parameters, plot the training process,
    save the model if required, and print out the test acc(s) in the end.

    Args:
        data_dir: Top level data directory.
        script_dir: Directory to save traced script.
        model_dir: Directory to save trained model.
        model_name: Model to use.
        num_classes: Total number of target classes.
        batch_size: Batch size for training.
        num_epochs: Number of epochs to train.
        feature_extract: Feature extracting or finetuning.
        learning_rates: Candidates of learning rates to use.
        weight_decays: Candidates of weight decays to use.
        viz: Visdom object to visualize training process.

    Returns:
        model: best model during validation.
        best_val_acc: best val acc.
        test_acc: test acc.

    """
    # Make sure model type is valid
    assert model_name in available_models_input_size.keys(), 'Unsupported model type ' + model_name
    print('Using %s ...\n' % model_name)

    # Record start time
    since = time.time()

    # Create dataloaders
    print('Loading dataset ...\n')
    dataloaders, dataset_mean, dataset_std = \
        create_dataloaders(data_dir, available_models_input_size[model_name], batch_size)
    print('+ Dataset mean:', dataset_mean[0])
    print('+ Dataset standard deviation:', dataset_std[0])
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    print('+ %(train)d samples for training, %(val)d for validation, %(test)d for test.\n' % dataset_sizes)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    # Keep record of best val acc and best model
    best_model = None
    best_val_acc = 0.0
    best_lr = None
    best_wd = None

    # Validating over hyper-parameters
    print('Validating hyper-parameters ...')
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            # Initialize the model for this run
            model, params_to_update = initialize_model(model_name, num_classes, feature_extract)

            # Send the model to GPU
            model = model.to(device)

            # Setup optimizer
            optimizer = optim.Adam(
                params_to_update,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=weight_decay,
                amsgrad=False
            )

            # Train and evaluate
            model, val_acc_history, loss_history = train_model(
                model=model,
                dataloaders=dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                is_inception=(model_name == "inception")
            )

            # Plot val acc and loss through learning process
            if viz:
                trace_name = '%s[lr=%g,wd=%g]' % (model_name, learning_rate, weight_decay)
                viz.line(
                    X=np.arange(1, 1 + len(val_acc_history)),
                    Y=val_acc_history,
                    name=trace_name,
                    win='Val Acc',
                    update='append',
                    opts={
                        'title': 'Val Acc',
                        'xlabel': 'epoch',
                        'ylabel': 'val_acc',
                        'showlegend': True,
                        'height': 400,
                        'width': 700
                    }
                )
                viz.line(
                    X=np.arange(1, 1 + len(loss_history)),
                    Y=loss_history,
                    name=trace_name,
                    win='Loss',
                    update='append',
                    opts={
                        'title': 'Loss',
                        'xlabel': 'epoch',
                        'ylabel': 'loss',
                        'showlegend': True,
                        'height': 400,
                        'width': 700
                    }
                )

            # Print val acc for this set of hyper-parameters
            this_val_acc = max(val_acc_history)
            print('\n* lr = %.4e, wd = %.4e, val_acc=%.4f\n' % (learning_rate, weight_decay, this_val_acc))

            # Update best model info
            if this_val_acc > best_val_acc:
                best_model = model
                best_val_acc = this_val_acc
                best_lr = learning_rate
                best_wd = weight_decay

    # Print out train result
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Model at lr=%e, wd=%e has the highest val acc: %.4f' % (best_lr, best_wd, best_val_acc))

    # Temperature scaling
    print('\nComputing temperature ...')
    temperature = compute_temperature(best_model, dataloaders['val'], verbose=True)

    # For dev purpose ...
    if model_dir:
        torch.save(best_model.state_dict(), os.path.abspath(os.path.join(model_dir, model_name + '.pt')))

    # Test model
    test_acc, _ = test_model(best_model, dataloaders['test'])
    print('\nTest Acc: %.4f' % test_acc)

    # Save the best model to disk
    if script_dir:
        # Convert to torch script for inference efficiency
        best_model = convert_to_torch_script(best_model, available_models_input_size[model_name])
        os.makedirs(script_dir, exist_ok=True)
        print('\nSaving model ...')
        info_list = [
            model_name,                         # model type
            time.ctime().replace(':', '#'),     # timestamp
            '%.3f' % temperature,               # temperature of the model
            '%.5f' % dataset_mean[0],
            '%.5f' % dataset_std[0]
        ]
        file_name = '%'.join(info_list) + '.pt'
        file_url = os.path.join(script_dir, file_name)
        best_model.save(file_url)
        print('\nModel saved to %s\n' % file_url)

    return best_model, best_val_acc, test_acc


def convert_to_torch_script(model, input_size):
    """Convert torch.nn.Module to torch.jit.ScriptModule via tracing.

    Args:
        model: trained torch.nn.Module instance.
        input_size: input size of the model.

    Returns:
        traced torch.jit.ScriptModule instance.

    """
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, input_size, input_size)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    return traced_script_module


def main():
    validate_model('../datasets/data0229', model_name='squeezenet', viz=None, num_epochs=30,
                   feature_extract=True)


if __name__ == '__main__':
    main()
