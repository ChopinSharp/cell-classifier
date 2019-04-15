import cv2
import os
import matplotlib.pyplot as plt

data_dir = '../datasets/data0229_svm/WT'

for name in os.listdir(data_dir):
    path = os.path.join(data_dir, name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    thres, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    print(thres)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_bin)
    plt.show()


