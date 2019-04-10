import cv2
import random
import math
import numpy as np
import torch


class ToNumpy:
    """
    Convert a PIL.Image object to 3-dimensional numpy.ndarray.
    """

    def __call__(self, img):
        """
        :param img: PIL image to convert.
        :return: Numpy array.
        """
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            channels = img_array.shape[2]
            assert channels == 3, 'Images with %d channels are not supported' % channels
            return img_array
        # Expand grayscale image via broadcasting
        expanded_img = np.zeros((*img_array.shape, 3), dtype=img_array.dtype)
        expanded_img[:, :, :] = img_array[:, :, np.newaxis]
        return expanded_img


class Resize:

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        """
        :param size: Of type int. Expected size.
        :param interpolation: Sampling method to use.
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        :param img: Of type numpy.ndarray. Image to be scaled.
        :return: Rescaled image.
        """
        return cv2.resize(img, (self.size, self.size), interpolation=self.interpolation)


class RandomResizedCrop:
    """
    Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    """

    def __init__(self, size, scale=(0.1, 1.0), ratio=(5. / 6., 6. / 5.), interpolation=cv2.INTER_LINEAR):
        """
        :param size: expected output size of each edge
        :param scale: range of size of the origin size cropped
        :param ratio: range of aspect ratio of the origin aspect ratio cropped
        :param interpolation: Default: cv2.INTER_LINEAR
        """
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """
        Get parameters for crop for a random sized crop.
        :param img: (numpy.ndarray) Image to be cropped.
        :param scale: (tuple) range of size of the origin size cropped.
        :param ratio: (tuple) range of aspect ratio of the origin aspect ratio cropped.
        :returns tuple: params (i, j, h, w) to be passed to crop for a random sized crop.
        """
        area = img.shape[0] * img.shape[1]
        init_ratio = img.shape[1] / img.shape[0]
        ratio = list(map(lambda x: x * init_ratio, ratio))

        # print(ratio)

        w, h = None, None  # dummy line to suppress Pycharm warning ...

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # I don't really understand this if ...
            # if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
            #     w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)

                # print(' - Resized crop done')

                return i, j, h, w

        # Fallback

        # print(' - Resized crop failed with w =', w, 'h =', h)

        w = min(img.shape[0], img.shape[1])
        i = (img.shape[1] - w) // 2
        j = (img.shape[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        :param img: (numpy.ndarray) Image to be cropped and resized.
        :return: (numpy.ndarray) Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return cv2.resize(img[i:i+h, j:j+w], self.size, interpolation=cv2.INTER_LINEAR)


class RandomHorizontalFlip:
    """
    Horizontally flip the given PIL Image randomly with a given probability.
    """

    def __init__(self, p=0.5):
        """
        :param p: (float) probability of the image being flipped. Default value is 0.5.
        """
        self.p = p

    def __call__(self, img):
        """
        :param img: Image to be flipped.
        :return: Randomly flipped image.
        """
        if random.random() < self.p:
            return img[:, ::-1]
        return img


class RandomVerticalFlip:
    """
    Vertically flip the given PIL Image randomly with a given probability.
    """

    def __init__(self, p=0.5):
        """
        :param p: (float) probability of the image being flipped. Default value is 0.5.
        """
        self.p = p

    def __call__(self, img):
        """
        :param img: Image to be flipped.
        :return: Randomly flipped image.
        """
        if random.random() < self.p:
            return img[::-1]
        return img


class RandomRotation:
    """
    Rotate the image by angle.
    """

    def __init__(self, degrees, interpolation=cv2.INTER_LINEAR):
        """
        :param degrees: Rotate for random degrees randomly chosen from (-degrees, +degrees).
        :param interpolation: Interpolation method.
        """
        self.degrees = (-degrees, degrees)
        self.interpolation = interpolation

    def __call__(self, img):
        rows, cols, *_ = img.shape
        angle = random.uniform(*self.degrees)
        m = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
        return cv2.warpAffine(img, m, (cols, rows), flags=self.interpolation)


class ToTensor:
    """
    Convert a numpy array to 3-dimensional torch tensor.
    """

    def __call__(self, img_array):
        if img_array.dtype == np.uint8:
            max_value = 2 ** 8
        elif img_array.dtype == np.uint16:
            max_value = 2 ** 16
        else:
            assert False, 'In opencv_transforms.ToTensor: unsupported dtype: ' + str(img_array.dtype)

        # Expand grayscale image to 3-dim array
        if len(img_array.shape) == 3:
            channels = img_array.shape[2]
            assert channels == 3, 'Images with %d channels are not supported' % channels
            img_expanded = img_array
        else:
            img_expanded = np.zeros((*img_array.shape, 3), dtype=img_array.dtype)
            img_expanded[:, :, :] = img_array[:, :, np.newaxis]

        img = torch.from_numpy(img_expanded.transpose((2, 0, 1)).astype(np.float32, casting='safe'))
        return img.div(max_value)


if __name__ == '__main__':
    """
    Test module.
    """

    def test():

        from PIL import Image
        import matplotlib.pyplot as plt
        img = Image.open('data0318/test/hyperfused/(1).tif')
        print('Show PIL.Image via matplotlib ...')
        plt.imshow(img)
        plt.show()

        print('\nTest ToNumpy ...')
        print('Image of mode:', img.mode)
        img_array = ToNumpy()(img)
        print('Convert to ndarray of dtype:', img_array.dtype, 'size: ', img_array.shape)
        print('Show ndarray via OpenCV ...')
        cv2.imshow('image', img_array)
        print('[ PRESS ENTER TO CONTINUE ! ]')
        cv2.waitKey(0)

        print('\nTest Resize ...')
        print('Resize to (300, 300) ...')
        img_resized = Resize(300)(img_array)
        cv2.imshow('image', img_resized)
        print('[ PRESS ENTER TO CONTINUE ! ]')
        cv2.waitKey(0)

        print('\nTest RandomResizedCrop ...')
        print('Ten rounds in total ...')
        trans = RandomResizedCrop(300)
        for i in range(10):
            print(i)
            img_cropped = trans(img_array)
            cv2.imshow('image', img_cropped)
            print('[ PRESS ENTER TO CONTINUE ! ]')
            cv2.waitKey(0)

        print('\nTest RandomHorizontalFlip ...')
        print('Ten rounds in total ...')
        trans = RandomHorizontalFlip(0.5)
        for i in range(10):
            print(i)
            img_flipped = trans(img_array)
            cv2.imshow('image', img_flipped)
            print('[ PRESS ENTER TO CONTINUE ! ]')
            cv2.waitKey(0)

        print('\nTest RandomVerticalFlip ...')
        print('Ten rounds in total ...')
        trans = RandomVerticalFlip(0.5)
        for i in range(10):
            print(i)
            img_flipped = trans(img_array)
            cv2.imshow('image', img_flipped)
            print('[ PRESS ENTER TO CONTINUE ! ]')
            cv2.waitKey(0)

        print('\nTest RandomRotation ...')
        trans = RandomRotation(45)
        print('Ten rounds in total ...')
        for i in range(10):
            print(i)
            img_rot = trans(img_array)
            cv2.imshow('image', img_rot)
            print('[ PRESS ENTER TO CONTINUE ! ]')
            cv2.waitKey(0)

        print('\nTest ToTensor ...')
        img_tensor = ToTensor()(img_array)
        print('Tensor of shape:', img_tensor.size(), 'type:', img_tensor.dtype)
        cv2.destroyAllWindows()

    test()
