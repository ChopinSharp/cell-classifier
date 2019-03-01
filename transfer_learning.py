import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import PIL
import warnings
import time
import os
import copy

# Device to use in training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, show_model=False):
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        with warnings.catch_warnings(): # temporarily suppress warnings about deprecated functions
            warnings.simplefilter("ignore")
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    if show_model:
        print(model_ft)

    return model_ft, input_size


def create_dataloaders(data_dir, input_size, batch_size):
    # Calculate dataset mean and standard deviation
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=5, drop_last=True)
    num_batches = len(dataset) // 10
    dataset_mean = torch.zeros(3)
    dataset_std = torch.zeros(3)
    for inputs, _ in iter(loader):
        dataset_mean += inputs.mean(dim=(0, 2, 3))
    dataset_mean /= num_batches
    for inputs, _ in iter(loader):
        dataset_std += torch.mean((inputs - dataset_mean.reshape((1, 3, 1, 1))) ** 2, dim=(0, 2, 3))
    dataset_std = torch.sqrt(dataset_std / num_batches)
    print('\nDataset mean:', dataset_mean.tolist())
    print('Dataset standard deviation:', dataset_std.tolist())

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

    # class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print('\n%(train)d samples for training, %(val)d for validation, %(test)d for test' % dataset_sizes)

    return dataloaders


def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False, verbose=True):
    since = time.time()
    val_acc_history = []
    loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if verbose:
            print('\nEpoch %d/%d' % (epoch + 1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # record statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if verbose:
                print('%s Loss: %.4f Acc: %.4f' % (phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: %4f' % best_acc)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, loss_history


def test_model(model, dataloaders, is_inception=False):
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


def train_main():
    # Top level data directory
    data_dir = 'data0229_dp'

    # Directory that holds trained models
    model_dir = 'models'

    # Models available
    model_names = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']

    # Number of classes in the dataset
    num_classes = 3

    # Batch size for training
    batch_size = 4

    # Number of epochs to train for
    num_epochs = 20

    # Flag for feature extracting
    feature_extract = True

    # Choose model
    model_name = 'squeezenet'
    # for model_name in model_names:
    #     print('Using', model_name)

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Create dataloaders
    dataloaders = create_dataloaders(data_dir, input_size, batch_size)

    # Gather the parameters to be optimized in this run
    print("\nParams to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print('\t', name)
    else:
        params_to_update = model_ft.parameters()
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print('\t', name)

    learning_rate = 1e-3
    weight_decay = 0
    optimizer = optim.Adam(
        params_to_update,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=False
    )

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, val_acc_history, loss_history = train_model(
        model=model_ft,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        is_inception=(model_name == "inception")
    )

    # Plot val acc and loss through learning process
    plt.title('Training Dynamics')
    plt.subplot(211)
    plt.xlabel('epoch')
    plt.ylabel('val_acc')
    plt.plot(val_acc_history)
    plt.subplot(212)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss_history)
    plt.show()

    # Save the best model to disk
    file_name = '-'.join([model_name, *(time.ctime().split()[1: -1])]) + '.pt'
    torch.save(model_ft.state_dict(), os.path.join(model_dir, file_name))

    # Test model
    test_acc = test_model(model_ft, dataloaders)
    print('Test Acc: %.4f' % test_acc)



def compute_saliency_maps(model, inputs, labels):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
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


def show_saliency_maps(model, inputs, labels, class_names):
    # Compute saliency maps for inputs
    saliency = compute_saliency_maps(model, inputs, labels)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    acc_flags = preds == labels
    # print('showing results ...')

    # Convert images and saliency maps from Torch Tensor to Numpy ndarray
    inputs = inputs.detach().numpy().transpose(0, 2, 3, 1).clip(0, 1)
    labels = labels.detach().numpy().clip(0, 1)
    saliency = saliency.numpy().clip(0, 1)

    # Show images and saliency maps together
    N = inputs.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(inputs[i])
        plt.axis('off')
        plt.title(class_names[labels[i]] + '[T]' if acc_flags[i] else '[F]')
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()





def visualize_model():
    model_path = 'models/squeezenet-Mar-1-18:05:52'
    data_dir = 'data0229_dp'
    num_samples = 5
    model_ft, input_size = initialize_model('squeezenet', num_classes=3, feature_extract=True, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_path))
    for param in model_ft.parameters():
        param.requires_grad = False
    model_ft.eval()
    dataloaders = create_dataloaders(data_dir, input_size, num_samples)
    class_names = dataloaders['test'].dataset.classes
    X, y = next(iter(dataloaders['test']))
    show_saliency_maps(model_ft, X, y, class_names)

if __name__ == '__main__':
    visualize_model()

