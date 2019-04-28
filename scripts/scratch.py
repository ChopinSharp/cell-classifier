import cv2
import os
import matplotlib.pyplot as plt


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
    from utils.misc import estimate_dataset_mean_and_std
    mean, std = estimate_dataset_mean_and_std('../datasets/data0229', 300)
    print(mean, std)
    mean, std = estimate_dataset_mean_and_std('../datasets/data0318', 300)
    print(mean, std)


if __name__ == '__main__':
    inspect_dataset()



