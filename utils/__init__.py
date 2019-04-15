from .misc import *
from . import opencv_transforms
from .temperature_scaling import compute_temperature

# Device to use in training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
