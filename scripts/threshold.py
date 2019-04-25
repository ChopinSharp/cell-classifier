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


def main():
    show_threshold('../datasets/segtest0424/CZ22405 Z ASIX-WOUND-001_w0001.tif')
    # for name in os.listdir(data_dir):
    #     path = os.path.join(data_dir, name)


if __name__ == '__main__':
    main()



