import cv2
import numpy as np
import argparse
import os
import random
from utils.misc import create_montage
import copy
from tqdm import tqdm
import time


# Note that OpenCV uses BGR format when reading and writing images ...
palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]])


def create_segmentation_dataset(input_dir, output_dir):
    for folder in os.listdir(input_dir):
        input_folder_path = os.path.join(input_dir, folder)
        data_folder_path = os.path.join(output_dir, 'data', folder)
        anno_folder_path = os.path.join(output_dir, 'anno', folder)
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


def enhance_dataset():
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


def group_list_and_drop_remainder(L, group_size):
    return [L[i * group_size: (i + 1) * group_size] for i in range(len(L) // group_size)]


# Create montage data set using a segmentation dataset
def create_montage_dataset(input_dir, output_dir, num_walks=6, dice_size=128, dice_counts=(4, 4), ratio_threshold=0.2,
                           dilation=1):
    top_anno_dir = os.path.join(input_dir, 'anno')
    top_data_dir = os.path.join(input_dir, 'data')
    top_output_anno_dir = os.path.join(output_dir, 'anno')
    top_output_data_dir = os.path.join(output_dir, 'data')
    for t in ['train', 'val', 'test']:
        anno_dir = os.path.join(top_anno_dir, t)
        data_dir = os.path.join(top_data_dir, t)
        output_anno_dir = os.path.join(top_output_anno_dir, t)
        output_data_dir = os.path.join(top_output_data_dir, t)
        os.makedirs(output_anno_dir, exist_ok=True)
        os.makedirs(output_data_dir, exist_ok=True)
        names = os.listdir(data_dir)
        # counting ....
        dices_per_montage = dice_counts[0] * dice_counts[1]
        montages_per_walk = len(names) * dilation // dices_per_montage
        print('generating {} ...'.format(t))
        print(' - {} dices per montage'.format(dices_per_montage))
        print(' - {} montages per walk'.format(montages_per_walk))
        print(' - {} samples in total'.format(num_walks * montages_per_walk))

        time.sleep(1)
        for num_walk in tqdm(range(num_walks), desc='Progress'):
            random.shuffle(names)
            # group names into montages
            name_groups = group_list_and_drop_remainder(names * dilation, dices_per_montage)
            # one group, one montage
            for idx, name_group in enumerate(name_groups):
                anno_montage_list = []
                data_montage_list = []
                # fill in the montage list
                for name in name_group:
                    anno = cv2.imread(os.path.join(anno_dir, name))
                    data = cv2.imread(os.path.join(data_dir, name))
                    # random crop
                    while True:
                        x = random.randint(0, 400 - dice_size)
                        y = random.randint(0, 600 - dice_size)
                        anno_roi = anno[x: x + dice_size, y: y + dice_size]
                        ratio = np.sum(np.any(anno_roi > 128, axis=2)) / (dice_size * dice_size)
                        if ratio > ratio_threshold:
                            break
                    anno_montage_list.append(anno_roi)
                    data_montage_list.append(data[x: x + dice_size, y: y + dice_size])
                # create montages
                anno_montage = create_montage(group_list_and_drop_remainder(anno_montage_list, dice_counts[0]))
                data_montage = create_montage(group_list_and_drop_remainder(data_montage_list, dice_counts[0]))
                output_name = '{0:03d}.png'.format(num_walk * montages_per_walk + idx + 1)
                cv2.imwrite(os.path.join(output_anno_dir, output_name), anno_montage)
                cv2.imwrite(os.path.join(output_data_dir, output_name), data_montage)
            time.sleep(1)
        print('done generating {}'.format(t))


def _test_create_montage_dataset():
    # create_montage_dataset('../datasets/data0531_seg_enhanced/data/train')
    data_folder = '../datasets/data0531_seg_enhanced/data/train'
    anno_folder = '../datasets/data0531_seg_enhanced/anno/train'
    file_names = os.listdir(data_folder)
    random.shuffle(file_names)
    data_full_url = lambda name: os.path.join(data_folder, name)
    anno_full_url = lambda name: os.path.join(anno_folder, name)
    data = cv2.imread(data_full_url(file_names[0]), cv2.IMREAD_COLOR)
    anno = cv2.imread(anno_full_url(file_names[0]), cv2.IMREAD_COLOR)
    montage = create_montage([[data], [anno]], 2)
    cv2.imshow('lalala', montage)
    cv2.waitKey(0)
    edge_length = 128
    while True:
        ix = random.randint(0, 400 - edge_length)
        iy = random.randint(0, 600 - edge_length)
        anno_roi = anno[ix: ix+edge_length, iy: iy+edge_length]
        data_roi = data[ix: ix+edge_length, iy: iy+edge_length]
        ratio = np.sum(np.sum(anno_roi > 0, axis=2) > 0) / (edge_length * edge_length)
        if ratio < 0.2:
            continue
        print('ratio: {}'.format(ratio))
        montage = create_montage([[data_roi, anno_roi]], 2)
        cv2.imshow('lalala', montage)
        cv2.waitKey(0)


if __name__ == '__main__':
    create_montage_dataset(
        '../datasets/data0531_seg_enhanced',
        '../datasets/montage_16_16_32_data0531_seg_enhanced',
        dice_size=32,
        dice_counts=(16, 16),
        num_walks=60,
        dilation=10
    )
    pass
