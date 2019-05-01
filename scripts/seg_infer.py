from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
from main.cell_segmentation import *


def merge_tif(image_url, verbose=False):
    image_stack = Image.open(image_url)
    merged_image = np.zeros(image_stack.size, dtype=np.float32)
    print(image_stack.n_frames, 'frames in', image_url)
    for i in range(image_stack.n_frames):
        image_stack.seek(i)
        merged_image += np.fromstring(image_stack.tobytes(), dtype=np.uint16).reshape(*image_stack.size)
    merged_image /= 256
    return merged_image.clip(0, 255).astype(np.uint8)


def adjust_contrast(image, saturation=0.35):
    # path = '../datasets/segtest0426/CZ22405-1.tif'
    # image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    image_min, image_max = image.min(), image.max()
    # print(image_min, image_max)
    hist, _ = np.histogram(image, np.arange(image_min, image_max + 2), density=True)
    # print(hist.shape)
    # print(hist.sum())
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
    # print(cur_sat, low, high)
    image_transformed = (image.astype(np.float32) - low) * (255 / (high - low))
    return image_transformed.clip(0, 255).astype(np.uint8)
    # plt.title('origin')
    # plt.imshow(to_8_bit(image), cmap='gray')
    # plt.show()
    # plt.title('equalized')
    # plt.imshow(image_equalized, cmap='gray')
    # plt.show()
    # plt.title('hist. trans.')
    # plt.imshow(image_transformed, cmap='gray')
    # plt.show()


def pad_channels(image):
    return np.concatenate([image[:, :, np.newaxis]] * 3, axis=2)


def montage(images):
    rows = []
    for image_row in images:
        row = np.concatenate(image_row, axis=1)
        rows.append(row)
        rows.append(np.ones((2, row.shape[1], 3), dtype=np.uint8) * 255)
    rows.pop()
    return np.concatenate(rows, axis=0)


def infer(model, folder_url):
    files = os.listdir(folder_url)
    total = len(files)
    # fig = plt.figure(figsize=[10, 5 * total])
    model.eval()
    images = []
    for idx, name in enumerate(files):
        image_url = os.path.join(folder_url, name)
        image = cv2.imread(image_url, cv2.IMREAD_ANYDEPTH)
        ori_image = (image / 256).astype(np.uint8)
        image_enhanced = adjust_contrast(image, 0.35)  # merge_tif(image_url)
        float_image = ori_image.astype(np.float32) / 256
        float_image = pad_channels(float_image)
        inputs = torch.from_numpy(float_image.transpose((2, 0, 1)))
        mean = inputs.mean()
        std = inputs.std()
        print('mean:', mean.item(), 'std:', std.item())
        inputs = inputs.sub(mean).div(std).unsqueeze(0).float()

        preds = model(inputs).argmax(dim=1)
        palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        pd = palette[preds.detach().cpu().numpy()[0]]

        images.append([pad_channels(ori_image), pad_channels(image_enhanced), pd])
        # ax = fig.add_subplot(total, 3, 3 * idx + 1)
        # ax.set_title('Original')
        # ax.imshow(ori_image, cmap='gray')
        # ax = fig.add_subplot(total, 3, 3 * idx + 2)
        # ax.set_title('Enhanced')
        # ax.imshow(image_enhanced, cmap='gray')
        # ax = fig.add_subplot(total, 3, 3 * idx + 3)
        # ax.set_title('Pred.')
        # ax.imshow(pd)

    save_path = folder_url.split('/')[-1] + '.jpg'
    print('Saving results to', save_path)
    result = montage(images)
    cv2.imwrite(save_path, result)
    # fig.savefig(folder_url.split('/')[-1] + '.jpg')


def visualize_model(model, loader):
    model.eval()
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    os.makedirs('tmp', exist_ok=True)
    counter = 1
    for idx, (sample, label) in enumerate(loader):
        preds = model(sample).argmax(dim=1)
        batch_size = sample.size()[0]
        fig = plt.figure(figsize=[19.2, 10.8])
        for i in range(batch_size):
            gd = palette[label.detach().cpu().numpy()[i]]
            pd = palette[preds.detach().cpu().numpy()[i]]
            img = (255 * sample.detach().cpu().numpy()[i].transpose((1, 2, 0))).clip(0, 255).astype(np.uint8)
            ax = fig.add_subplot(3, batch_size, i + 1)
            ax.set_title('img')
            ax.imshow(img)
            ax = fig.add_subplot(3, batch_size, i + 1 + batch_size)
            ax.set_title('gd')
            ax.imshow(gd)
            ax = fig.add_subplot(3, batch_size, i + 1 + 2*batch_size)
            ax.set_title('pd')
            ax.imshow(pd)
        plt.savefig('tmp/%d.jpg' % counter)
        counter += 1


if __name__ == '__main__':
    _model = UNetVgg()
    # _model.to('cuda:0')
    _model.load_state_dict(torch.load('../results/saved_models/UNetVgg Wed May  1 00:26:55 2019.pt'))
    infer(_model, '../datasets/segtest0424')
