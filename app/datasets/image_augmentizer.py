"""Image Augmentizing Wrapper for KMILT
Copyright 2018 Michael (Tao-Yi) Lee

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

from app.datasets.AugmentConfig import AugmentConfig


class ImageAugmentizer:
    def __init__(self, aug_config):
        """

        :param nrm_config:
        :type nrm_config: NormalizeConfig
        """
        ia.seed(1)
        if not isinstance(aug_config, AugmentConfig):
            raise TypeError(f"{aug_config} is not a AugmentConfig")
        else:
            self.aug_config = aug_config

        aug_seq = []
        if self.aug_config.random_horz_flip:
            aug_seq.append(iaa.Fliplr(self.aug_config.flip_prob, name="fliplr0"))

        if self.aug_config.random_vert_flip:
            aug_seq.append(iaa.Flipud(self.aug_config.flip_prob, name="flipud0"))

        # print(f"augmenting sequence = {aug_seq}")
        self.seq = iaa.Sequential(aug_seq, random_order=False)  # apply augmenters in random order

    def augmentize(self, images: np.ndarray):
        """

        :param images:
        :return:
        """
        if images.ndim < 3:
            raise ValueError(
                f"Received a {images.ndim} array, this method does not support image with less than 3 dimensions")

        if images.ndim > 4:
            raise ValueError(
                f"Received a {images.ndim} array, this method does not support image array with more than 4 dimensions")

        if images.ndim == 3:
            images_aug = np.array(self.seq.augment_image(images))
        else:
            images_aug = np.array(self.seq.augment_images(images))
        return images_aug
