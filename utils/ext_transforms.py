import random
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, ToPILImage
import numbers
from PIL import Image
from collections.abc import Iterable
import numpy as np
import torch


__all__ = ['ExtCompose', 'ExtColorJitter', 'ExtResize', 'ExtNormalize', 'ExtRandomCrop', 'ExtRandomHorizontalFlip',
           'ExtRandomRotation', 'ExtRandomVerticalFlip', 'ExtToTensor']


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class ExtCompose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtRandomHorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtRandomVerticalFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtColorJitter:

    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.):
        self.transform = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, lbl):
        return self.transform(img), lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.transform.brightness)
        format_string += ', contrast={0}'.format(self.transform.contrast)
        format_string += ', saturation={0}'.format(self.transform.saturation)
        format_string += ', hue={0})'.format(self.transform.hue)
        return format_string


class ExtRandomRotation:

    # noinspection PyTypeChecker
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, lbl):

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center), \
            F.rotate(lbl, angle, Image.NEAREST, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class ExtToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    @staticmethod
    def _lbl_to_tensor(lbl):
        lbl_array = np.array(lbl, dtype=np.uint8, copy=False)
        padding = np.ones((lbl_array.shape[0], lbl_array.shape[1], 1), dtype=np.uint8) * 128
        lbl_padded = np.concatenate((padding, lbl_array), axis=2)
        true_label = lbl_padded.argmax(axis=2)
        return torch.from_numpy(true_label).type(dtype=torch.long)

    def __call__(self, pic, lbl):
        return F.to_tensor(pic), self._lbl_to_tensor(lbl)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ExtNormalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std), lbl

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ExtResize:
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    # noinspection PyTypeChecker
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ExtRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl):
        assert img.size[0] > self.size and img.size[1] > self.size, 'No room for randomness'
        hi = random.randint(0, img.size[1] - self.size)
        wi = random.randint(0, img.size[0] - self.size)
        crop_cfg = (wi, hi, wi + self.size, hi + self.size)
        return img.crop(crop_cfg), lbl.crop(crop_cfg)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ExtToPILImage:
    def __init__(self):
        self.transform = ToPILImage()
        self.palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])

    def __call__(self, img, lbl):
        palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]])
        lbl_img = self.palette[lbl.detach().numpy()].astype(np.uint8)
        return self.transform(img), Image.fromarray(lbl_img, mode='RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scripts.train_segmentation import SegmentationImageFolder
    # dataset = SegmentationImageFolder(
    #     '../datasets/data0229_seg/data/val',
    #     '../datasets/data0229_seg/anno/val',
    #     ext_transforms=ExtCompose([
    #         ExtColorJitter(0.5, 0.5),
    #         ExtToTensor()
    #     ])
    # )
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    # t = ExtToPILImage()
    # for idx, (imgs, lbls) in enumerate(loader):
    #     img, lbl = t(imgs[0], lbls[0])
    #     plt.subplot(121)
    #     plt.imshow(img)
    #     plt.subplot(122)
    #     plt.imshow(lbl)
    #     plt.show()
    #     if idx == 10:
    #         break
    t = ExtColorJitter(0.9, 0.9)
    img = Image.open('../datasets/data0229_seg/data/test/1 (2).png')
    lbl = Image.open('../datasets/data0229_seg/anno/test/1 (2).png')
    plt.subplot(121)
    plt.imshow(img)
    img_t, _ = t(img, lbl)
    plt.subplot(122)
    plt.imshow(img_t)
    plt.show()
