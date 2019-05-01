import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from main.cell_classifier import available_models_input_size, initialize_model
from utils.misc import estimate_dataset_mean_and_std, expand_subdir


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
    saliency, _ = inputs.grad.max(dim=1)
    # saliency = inputs.grad.mean(dim=1)

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
    # saliency = 1.5 * (saliency - saliency.mean()) / (saliency.max() - saliency.min()) + 0.5

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


def visualize_model(model_dir='../results/saved_models', data_dir='../datasets/data0229', num_samples=4):
    """
    Visualize model via saliency maps.
    :param model_dir: Directory that holds saved model.
    :param data_dir: Directory that holds input data.
    :param num_samples: Number of samples to draw saliency maps.
    """

    # Load saved model
    model_file_name = os.listdir(model_dir)[0]  # load first model file by default
    model_name = model_file_name.split('.')[0]
    model_ft, _ = initialize_model(model_name, num_classes=3, feature_extract=True)
    model_ft.load_state_dict(torch.load(os.path.join(model_dir, model_file_name)))
    model_ft.eval()

    # Freeze model parameters
    for param in model_ft.parameters():
        param.requires_grad = False

    # Get dataset mean and std
    input_size = available_models_input_size[model_name]
    dataset_mean, dataset_std = estimate_dataset_mean_and_std(expand_subdir(data_dir), input_size)

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

    # Show saliency maps
    for (inputs, labels), (images, _) in zip(dataloaders['input'], dataloaders['show']):
        show_saliency_maps(model_ft, inputs, labels, images, class_names)


if __name__ == '__main__':
    visualize_model()
