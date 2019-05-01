import cv2
import numpy as np
import argparse
import os


# Note that OpenCV uses BGR format when reading and writing images ...
palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='input directory')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory')
    args = parser.parse_args()
    for folder in os.listdir(args.input_dir):
        input_folder_path = os.path.join(args.input_dir, folder)
        data_folder_path = os.path.join(args.output_dir, 'data', folder)
        anno_folder_path = os.path.join(args.output_dir, 'anno', folder)
        os.makedirs(data_folder_path, exist_ok=True)
        os.makedirs(anno_folder_path, exist_ok=True)
        for name in os.listdir(input_folder_path):
            file_path = os.path.join(input_folder_path, name)
            new_name = name[:name.rfind('.')] + '.png'
            data_path = os.path.join(data_folder_path, new_name)
            anno_path = os.path.join(anno_folder_path, new_name)
            print('processing', file_path)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(data_path, image)
            _, label = cv2.threshold(image, 0, int(name[0]), cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            anno = palette[label]
            cv2.imwrite(anno_path, anno)


if __name__ == '__main__':
    main()
