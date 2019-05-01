import cv2
import numpy as np
import argparse
import os


# Note that OpenCV uses BGR format when reading and writing images ...
palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]])


def main1():
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


def adjust_contrast(image, saturation=0.35):
    image_min, image_max = image.min(), image.max()
    hist, _ = np.histogram(image, np.arange(image_min, image_max + 2), density=True)
    i, j = 0, hist.shape[0] - 1
    cur_sat = 0.
    while cur_sat < saturation / 100:
        if hist[i] < hist[j]:
            cur_sat += hist[i]
            i += 1
        else:
            cur_sat += hist[j]
            j -= 1
    low, high = i + image_min, j + image_min
    image_transformed = (image.astype(np.float32) - low) * (255 / (high - low))
    return image_transformed.clip(0, 255).astype(np.uint8)


def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='input directory')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory')
    args = parser.parse_args()
    for folder in os.listdir(args.input_dir):
        folder_path = os.path.join(args.input_dir, folder)
        output_folder = os.path.join(args.output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)
        for name in os.listdir(folder_path):
            image_url = os.path.join(folder_path, name)
            print('processing', image_url)
            image = cv2.imread(image_url, cv2.IMREAD_ANYDEPTH)
            image_enhanced = adjust_contrast(image, 0.35)
            output_url = os.path.join(output_folder, name)
            cv2.imwrite(output_url, image_enhanced)


if __name__ == '__main__':
    main2()
