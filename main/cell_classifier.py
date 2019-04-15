import torch.nn as nn
from torchvision import models
import warnings


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





