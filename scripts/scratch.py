import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models


data_dir = '../datasets/data0229_svm/WT'


def show_threshold(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    thres, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    print('threshold:', thres)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_bin)
    plt.show()


def inspect_dataset():
    from utils.misc import estimate_dataset_mean_and_std, expand_subdir
    mean, std = estimate_dataset_mean_and_std(expand_subdir('../datasets/data0229'), 300)
    print(mean, std)
    mean, std = estimate_dataset_mean_and_std(expand_subdir('../datasets/data0318'), 300)
    print(mean, std)


def to_8_bit(image):
    return (image / 256).clip(0, 255).astype(np.uint8)


def inspect_resnet():
    model = models.resnet18()
    x = torch.zeros(1, 3, 224, 224)
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    print('after conv', x.size())
    x = model.maxpool(x)
    print('after pool', x.size())
    x = model.layer1(x)
    print('after layer1', x.size())
    x = model.layer2(x)
    print('after layer2', x.size())
    x = model.layer3(x)
    print('after layer3', x.size())
    x = model.layer4(x)
    print('after layer4', x.size())


def inspect_features(features):
    x = torch.zeros(1, 3, 224, 224)
    for idx, layer in enumerate(features):
        x = layer(x)
        print(idx, x.size())


if __name__ == '__main__':
    from main.cell_segmentation import UNetVggVar
    model = UNetVggVar()
    model.load_state_dict(torch.load('../results/saved_models/UNetVggVarCPU.pt'))
    inputs = torch.randn(1, 3, 224, 224)
    model_script = torch.jit.trace(model, inputs)
    print('model traced')
    o1 = model(inputs)
    o2 = model_script(inputs)
    print('distance {0:.8f}'.format(torch.pow(o1 - o2, 2).sum().item()))
    model_script.save('../results/saved_scripts/UNetVggVar.pt')
    m = torch.jit.load('../results/saved_scripts/UNetVggVar.pt')
    o3 = m(inputs)
    print('distance {0:.8f}'.format(torch.pow(o1 - o3, 2).sum().item()))












