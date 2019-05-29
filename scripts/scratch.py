import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models
from scripts.seg_utils import create_montage, adjust_contrast
from scripts.train_segmentation import create_dataloaders
from PIL import Image
from PIL import ImageFilter


data_dir = '../datasets/data0229_svm/WT'


def show_threshold():
    data = {
        'wt': '../datasets/data0229_seg/data/test/1 (2).png',
        'fg': '../datasets/data0229_seg/data/test/2(64).png',
        'hf': '../datasets/data0229_seg/data/test/3(2).png'
    }
    anno = {
        'wt': '../datasets/data0229_seg/anno/test/1 (2).png',
        'fg': '../datasets/data0229_seg/anno/test/2(64).png',
        'hf': '../datasets/data0229_seg/anno/test/3(2).png'
    }
    images = [[], [], [], []]
    for cls in ['wt', 'fg', 'hf']:
        img_ = cv2.imread(data[cls], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img = adjust_contrast(img_)
        lbl = cv2.imread(anno[cls], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        thres, img_bin = cv2.threshold(img[:, :, 0], 0, 255, cv2.THRESH_OTSU)
        img_bin = img_bin[:, :, np.newaxis]
        img_bin = np.concatenate((img_bin, img_bin, img_bin), axis=2)
        print('threshold:', thres)
        images[0].append(img_)
        images[1].append(img)
        images[2].append(img_bin)
        images[3].append(lbl)

    result = create_montage(images)
    cv2.imwrite('lalala.png', result)



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
    img = Image.open('../datasets/data0229_enhanced/train/WT/1 (4).jpg')
    img2 = img.filter(ImageFilter.GaussianBlur())
    img7 = img.filter(ImageFilter.GaussianBlur(1))
    plt.subplot(311)
    plt.imshow(img, cmap='gray')
    plt.subplot(312)
    plt.imshow(img2, cmap='gray')
    plt.subplot(313)
    plt.imshow(img7, cmap='gray')
    plt.show()













