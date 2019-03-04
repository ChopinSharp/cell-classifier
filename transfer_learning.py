import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL
import warnings
import time
import os
import copy

# Device to use in training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Available models and their expected input size
available_models_input_size = {
    "resnet": 224,
    "alexnet": 224,
    "vgg": 224,
    "squeezenet": 224,
    "densenet": 224,
    "inception": 299
}


def set_parameter_requires_grad(model, feature_extract):
    """
    Set the require_grads attribute of model's parameters according to feature_extract flag.
    :param model: Model to operate on.
    :param feature_extract: Feature extracting or finetuning.
    """

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, verbose=False):
    """
    Initialize required model and set it up for feature extracting or finetuning.
    :param model_name: Type of model to initialize.
    :param num_classes: Total number of target classes.
    :param feature_extract: Feature extracting or finetuning.
    :param verbose: Print model info in the end or not.
    :return model_ft: Initialized model.
    :return params_to_update: List of parameters to be updated during training.
    """

    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        with warnings.catch_warnings():  # temporarily suppress warnings about deprecated functions
            warnings.simplefilter("ignore")
            model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:  # Unreachable
        exit()

    # Gather the parameters to be optimized
    params_to_update = list(filter(lambda p: p.requires_grad, model_ft.parameters()))

    # Print model info
    if verbose:
        print()
        print(model_ft)
        print()
        print("Params to learn:")
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print('\t', name)

    return model_ft, params_to_update


def estimate_dataset_mean_and_std(data_dir, input_size):
    """
    Calculate dataset mean and standard deviation.
    :param data_dir: Top directory of data.
    :param input_size: Expected input size.
    :return dataset_mean: Dataset mean.
    :return dataset_std: Dataset standard deviation.
    """

    # Load all samples into a single dataset
    dataset = torch.utils.data.ConcatDataset([
        datasets.ImageFolder(
            os.path.join(data_dir, x),
            transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor()
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
    dataset_std = torch.sqrt(dataset_std / num_batches)

    return dataset_mean, dataset_std


def create_dataloaders(data_dir, input_size, batch_size):
    """
    Create datasets and dataloaders.
    :param data_dir: Top directory of data.
    :param input_size: Expected input size.
    :param batch_size: Batch size.
    :return dataloaders: A dictionary that holds dataloaders for training, validating and testing.
    """

    # Get dataset mean and std
    dataset_mean, dataset_std = estimate_dataset_mean_and_std(data_dir, input_size)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45, resample=PIL.Image.BILINEAR),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])
    }

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    # Create training and validation dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
        for x in ['train', 'val', 'test']
    }

    return dataloaders, dataset_mean, dataset_std


def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False, verbose=True):
    """
    Train and validate model with given dataloaders, loss and optimizer for some number of epochs.
    Log the training dynamics and return the model with best val acc.
    :param model: Initialized model to be trained.
    :param dataloaders: Dataloaders to train and validate on.
    :param criterion: Loss function.
    :param optimizer: Optimizer to use with fixed hyper-parameters.
    :param num_epochs: Number of epochs to train.
    :param is_inception: Using Inception model or not.
    :param verbose: Print training info every epoch or not.
    :return model: Model with best val acc. Changed inplace, actually unnecessary to return.
    :return val_acc_history: History of val acc over epochs.
    :return loss_history: History of loss over epochs
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
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
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
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if verbose:
                print('+ %s Loss: %.4f Acc: %.4f' % (phase, epoch_loss, epoch_acc))

            # Deep copy the best model so far
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)

    # Print out best val acc
    # time_elapsed = time.time() - since
    # print('\nTraining complete in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, loss_history


def test_model(model, dataloaders, is_inception=False):
    """
    Test model.
    :param model: Model to test.
    :param dataloaders: Dataloader of test set.
    :param is_inception: Using Inception model or not.
    :return test_acc: Test accuracy.
    """

    model.eval()
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward, track history if only in train
        with torch.no_grad():
            if is_inception:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)

    return test_acc


def train(model_name='squeezenet', data_dir='data0229_dp', model_dir=None, plot_dir=None, num_classes=3,
          batch_size=4, num_epochs=20, feature_extract=True, learning_rates=(1e-3,), weight_decays=(0.0,)):
    """
    Train and validate chosen model with set(s) of hyper-parameters,
    plot the training process, save the model if required, and print out
    the test acc(s) in the end.
    :param model_name: Model to use.
    :param data_dir: Top level data directory.
    :param model_dir: Directory to save trained model.
    :param plot_dir: Directory to save training plots.
    :param num_classes: Total number of target classes.
    :param batch_size: Batch size for training.
    :param num_epochs: Number of epochs to train.
    :param feature_extract: Feature extracting or finetuning.
    :param learning_rates: Candidates of learning rates to use.
    :param weight_decays: Candidates of weight decays to use.
    """

    assert model_name in available_models_input_size.keys(), 'Unsupported model type ' + model_name
    print('Using %s ...\n' % model_name)

    # Create dataloaders
    print('Loading dataset ...\n')
    dataloaders, dataset_mean, dataset_std = \
        create_dataloaders(data_dir, available_models_input_size[model_name], batch_size)
    print('+ Dataset mean:', dataset_mean.tolist())
    print('+ Dataset standard deviation:', dataset_std.tolist())
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
            if plot_dir is not None:
                plt.title('Training Dynamics')
                plt.subplot(211)
                plt.xlabel('epoch')
                plt.ylabel('val_acc')
                plt.plot(val_acc_history)
                plt.subplot(212)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.plot(loss_history)
                plot_name = 'lr=%.4e_wd=%.4e.jpg' % (learning_rate, weight_decay)
                plt.savefig(os.path.join(plot_dir, plot_name))

            # Print val acc for this set of hyper-parameters
            this_val_acc = max(val_acc_history)
            print('\n* lr = %.4e, wd = %.4e, val_acc=%.4f\n' % (learning_rate, weight_decay, this_val_acc))

            # Update best model info
            if this_val_acc > best_val_acc:
                best_model = model
                best_val_acc = this_val_acc
                best_lr = learning_rate
                best_wd = weight_decay

        # Test model
        test_acc = test_model(best_model, dataloaders)
        print('Model at lr=%e, wd=%e has the highest val acc: %.4f' % (best_lr, best_wd, best_val_acc))
        print('Test Acc: %.4f' % test_acc)

        # Save the best model to disk
        if model_dir is not None:
            timestamp = time.ctime().split()
            info_list = [
                model_name,  # model type
                timestamp[-1], timestamp[1], timestamp[2], *(timestamp[3].split(':')),  # timestamp
                '%.4e-%.4e' % (best_lr, best_wd)  # hyper-parameters used
            ]
            file_name = '-'.join(info_list) + '.pt'
            torch.save(best_model.state_dict(), os.path.join(model_dir, file_name))


def compute_saliency_maps(model, inputs, labels):
    """
    Compute a class saliency map using the model for images X and labels y.
    :param model: Input images; Tensor of shape (N, 3, H, W).
    :param inputs: Labels for X; LongTensor of shape (N,).
    :param labels: A pretrained CNN that will be used to compute the saliency map.
    :return saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input images.
    """

    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    inputs.requires_grad_()

    # Compute the gradient of the correct class score with respect to each input image
    # Combine scores across a batch by summing
    sum_score = model(inputs).gather(1, labels.view(-1, 1)).squeeze().sum()
    sum_score.backward()
    saliency, _ = inputs.grad.abs().max(dim=1)

    return saliency


def show_saliency_maps(model, inputs, labels, images, class_names):
    """
    Show saliency map for trained model.
    :param model: Trained model to make predictions.
    :param inputs: Input data.
    :param labels: Labels of input data.
    :param images: Images for showing.
    :param class_names: Name of target classes.
    """

    # Compute saliency maps
    saliency = compute_saliency_maps(model, inputs, labels)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    acc_flags = preds == labels

    # Convert images and saliency maps from Torch Tensor to Numpy ndarray
    images = images.detach().numpy().transpose(0, 2, 3, 1)
    labels = labels.detach().numpy()
    saliency = saliency.numpy().clip(0, 1)

    # Show images and saliency maps together
    number_samples = images.shape[0]
    for i in range(number_samples):
        plt.subplot(2, number_samples, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(class_names[labels[i]] + '[T]' if acc_flags[i] else '[F]')
        plt.subplot(2, number_samples, number_samples + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


def visualize_model(model_dir='models', data_dir='data0229_dp', num_samples=5):
    """
    Visualize model via saliency maps.
    :param model_dir: Directory that holds saved model.
    :param data_dir: Directory that holds input data.
    :param num_samples: Number of samples to draw saliency maps.
    """

    # Load saved model
    model_file_name = os.listdir(model_dir)[0]  # load first model file by default
    model_name = model_file_name.split('-')[0]
    model_ft, _ = initialize_model(model_name, num_classes=3, feature_extract=True)
    model_ft.load_state_dict(torch.load(os.path.join(model_dir, model_file_name)))
    model_ft.eval()

    # Freeze model parameters
    for param in model_ft.parameters():
        param.requires_grad = False

    # Get dataset mean and std
    input_size = available_models_input_size[model_name]
    dataset_mean, dataset_std = estimate_dataset_mean_and_std(data_dir, input_size)

    # Resize and normalize for input, only resize for show.
    data_transforms = {
        'input': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'show': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
    }

    # Create datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms[x])
        for x in ['input', 'show']
    }
    class_names = image_datasets['input'].classes

    # Create dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=num_samples, shuffle=True, num_workers=4)
        for x in ['input', 'show']
    }

    # Sample a batch of data
    inputs, labels = next(iter(dataloaders['input']))
    images, _ = next(iter(dataloaders['show']))

    # Show saliency maps
    show_saliency_maps(model_ft, inputs, labels, images, class_names)


if __name__ == '__main__':
    train(plot_dir='figs', num_epochs=25, weight_decays=np.logspace(-8, -2, 5))
    # train()
    # visualize_model()
