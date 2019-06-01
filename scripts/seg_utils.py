from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
from main.cell_segmentation import *
from scripts.train_segmentation import get_scaled_size, create_dataloaders
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    using_tqdm = False
else:
    using_tqdm = True


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


def pad_channels(image):
    return np.concatenate([image[:, :, np.newaxis]] * 3, axis=2)


def create_montage(images, row_border_width=2):
    rows = []
    for image_row in images:
        image_row_with_border = [
            np.concatenate((image, np.ones((image.shape[0], row_border_width, 3), dtype=np.uint8) * 255), axis=1)
            for image in image_row[:-1]
        ]
        image_row_with_border.append(image_row[-1])
        row = np.concatenate(image_row_with_border, axis=1)
        rows.append(row)
        rows.append(np.ones((row_border_width, row.shape[1], 3), dtype=np.uint8) * 255)
    rows.pop()
    return np.concatenate(rows, axis=0)


def infer(model, model2, folder_url):
    files = os.listdir(folder_url)
    total = len(files)
    # fig = plt.figure(figsize=[10, 5 * total])
    model.eval()
    model2.eval()
    images = []
    for idx, name in enumerate(files):
        image_url = os.path.join(folder_url, name)
        image = cv2.imread(image_url, cv2.IMREAD_ANYDEPTH)
        image = cv2.resize(image, get_scaled_size((600, 400)), interpolation=cv2.INTER_LINEAR)
        ori_image = (image / 256).astype(np.uint8)
        image_enhanced = adjust_contrast(image, 0.35)
        float_image = image_enhanced.astype(np.float32) / 256
        float_image = pad_channels(float_image)
        inputs = torch.from_numpy(float_image.transpose((2, 0, 1)))
        mean = inputs.mean()
        std = inputs.std()
        print('mean:', mean.item(), 'std:', std.item())
        inputs = inputs.sub(mean).div(std).unsqueeze(0).float()

        o1 = model(inputs)
        o2 = model2(inputs)
        print('distance {0:.8f}'.format(torch.pow(o1 - o2, 2).sum().item()))

        preds = o1.argmax(dim=1)
        preds2 = o2.argmax(dim=1)
        palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        pd = palette[preds.detach().cpu().numpy()[0]]
        pd2 = palette[preds2.detach().cpu().numpy()[0]]

        images.append([pad_channels(ori_image), pad_channels(image_enhanced), pd, pd2])

    save_path = folder_url.split('/')[-1] + '.jpg'
    print('Saving results to', save_path)
    result = create_montage(images)
    cv2.imwrite(save_path, result)


def visualize_model(model, data_dir, save_path='visualization.jpg'):
    model.eval()
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    loaders, *_ = create_dataloaders(data_dir, 1, _batch_size=1, verbose=False)
    montage = []
    if using_tqdm:
        loader = tqdm(loaders['test'], desc='Visualize Model:')
    else:
        loader = loaders['test']
    for sample, label in loader:
        pred = model(sample).argmax(dim=1).squeeze()
        pd = palette[pred.detach().cpu().numpy()]
        gt = palette[label.squeeze().cpu().numpy()]
        montage.append([gt, pd])
    print('Saving results to', save_path, '...')
    result = create_montage(montage)
    cv2.imwrite(save_path, result)


def test_tracing(model, script, data_dir=None, loader=None, save_path='test_tracing.jpg'):
    assert data_dir is not None or loader is not None, 'either data_dir or loader should not be None'
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    if loader is None:
        loaders, *_ = create_dataloaders(data_dir, 1, _batch_size=1, verbose=False)
        loader = loaders['test']
    montage = []
    if using_tqdm:
        loader = tqdm(loader, desc='Test Tracing: ')
    for sample, label in loader:
        pred_m = model(sample).argmax(dim=1).squeeze()
        pd_m = palette[pred_m.detach().cpu().numpy()]
        pred_s = script(sample).argmax(dim=1).squeeze()
        pd_s = palette[pred_s.detach().cpu().numpy()]
        gt = palette[label.squeeze().cpu().numpy()]
        montage.append([gt, pd_m, pd_s])
    print('\nSaving results to', save_path, '...')
    result = create_montage(montage)
    cv2.imwrite(save_path, result)


def convert_to_torch_script(model, name):
    # model = UNetVggVar()
    # model.load_state_dict(torch.load('../results/saved_models/UNetVggVar Fri May 31 17:57:22 2019.pt'))
    model.eval()  # NOTE: ALWAYS remember to put model in EVAL mode before tracing !!!
    inputs = torch.randn(1, 3, 224, 224)

    script = torch.jit.trace(model, inputs)  # , check_inputs=check_inputs)
    script_url = os.path.join('../results/saved_scripts', name)
    torch.jit.save(script, script_url)

    loaders, *_ = create_dataloaders('../datasets/data0229_seg_enhanced', 1, _batch_size=1, verbose=False)
    test_tracing(model, torch.jit.load(script_url), loader=loaders['test'])


def main():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, required=True, help='path to model')
    # parser.add_argument('--input', type=str, required=True, help='input directory')
    # args = parser.parse_args()

    # model = UNetVggVar()
    # model.load_state_dict(torch.load(os.path.join('../results/saved_models', 'UNetVggVarCPU.pt')))
    # visualize_model(model, '../datasets/data0229_seg_enhanced')

    convert_to_torch_script()

    pass


if __name__ == '__main__':
    convert_to_torch_script()
