import cv2
import numpy as np
import argparse
import os


# note that opencv uses BGR format when reading and writing images ...
palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='data directory')
    parser.add_argument('--anno-dir', type=str, required=True, help='annotation directory')
    args = parser.parse_args()
    os.makedirs(args.anno_dir, exist_ok=True)
    for folder in os.listdir(args.data_dir):
        data_folder_path = os.path.join(args.data_dir, folder)
        anno_folder_path = os.path.join(args.anno_dir, folder)
        os.makedirs(anno_folder_path, exist_ok=True)
        for name in os.listdir(data_folder_path):
            file_path = os.path.join(data_folder_path, name)
            anno_path = os.path.join(anno_folder_path, name)
            print('processing', file_path)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(image, 0, int(name[0]), cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            anno = palette[label]
            cv2.imwrite(anno_path, anno)


if __name__ == '__main__':
    main()