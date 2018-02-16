from unittest import TestCase

import cv2
import numpy as np

from app.datasets import ImageAugmentizer, NormalizeConfig, AugmentConfig


class TestImageAugmentizer(TestCase):
    def setUp(self):
        self.conf = AugmentConfig("augment_config.ini")
        self.conf_prob = AugmentConfig("augment_config_flip_prob.ini")
        self.conf_ud = AugmentConfig("augment_config_ud.ini")
        self.conf_lr = AugmentConfig("augment_config_lr.ini")
        self.conf_noaug = AugmentConfig("augment_config_noaug.ini")
        self.image0 = np.array(np.arange(0, 27, 1).reshape(3, 3, 3))
        self.image1 = np.array(np.arange(50, 77, 1).reshape(3, 3, 3))
        self.image0_1 = np.stack([self.image0, self.image1])
        self.imagecxr0 = np.array(cv2.imread("00000001_000.png"))
        self.imagecxr1 = np.array(cv2.imread("00000001_000.png"))
        self.imagecxr0_1 = np.stack([self.imagecxr0, self.imagecxr1])

    def test_ImageAugmentizer(self):
        ia0 = ImageAugmentizer(self.conf)
        self.assertTrue(isinstance(ia0, ImageAugmentizer))

    def test_ImageAugmentizer_NoArgument(self):
        with self.assertRaises(TypeError):
            ImageAugmentizer()

    def test_ImageAugmentizer_BadParameter(self):
        with self.assertRaises(TypeError):
            ImageAugmentizer("pxr_config.ini")
        with self.assertRaises(TypeError):
            ImageAugmentizer(5566)
        with self.assertRaises(TypeError):
            ImageAugmentizer(NormalizeConfig("augment_config_noaug.ini"))

    def test_wrong_dimension(self):
        with self.assertRaises(ValueError):
            ia0 = ImageAugmentizer(self.conf)
            ia0.augmentize(np.random.rand(1, 1))

        with self.assertRaises(ValueError):
            ia0 = ImageAugmentizer(self.conf)
            ia0.augmentize(np.random.rand(1))

    def test_ImageAugmentizer_NoAug(self):
        ia0 = ImageAugmentizer(self.conf_noaug)
        image0_aug = ia0.augmentize(self.image0)
        self.assertTrue((self.image0 == image0_aug).all())

    def test_ImageAugmentizer_Aug(self):
        ia0 = ImageAugmentizer(self.conf)
        image0_aug = ia0.augmentize(self.image0)
        diff = self.image0 - image0_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

    def test_ImageAugmentizer_Aug_Prob(self):
        ia0 = ImageAugmentizer(self.conf_prob)
        flipped_times = 0
        test_runs = 10000
        for i in range(test_runs):
            image0_aug = ia0.augmentize(self.image0)
            diff = self.image0 - image0_aug
            if not (diff == np.zeros_like(diff)).all():
                flipped_times += 1
        self.assertTrue(abs(flipped_times / test_runs - self.conf_prob.flip_prob) < 0.01)

    def test_ImageAugmentizer_Aug_LR(self):
        ia0 = ImageAugmentizer(self.conf_lr)
        image0_aug = ia0.augmentize(self.image0)
        diff = self.image0 - image0_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

        image0_aug = ia0.augmentize(self.imagecxr0)
        diff = self.imagecxr0 - image0_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

    def test_ImageAugmentizer_Aug_LR_4D(self):
        ia0 = ImageAugmentizer(self.conf_lr)
        image_aug = ia0.augmentize(self.image0_1)
        diff = self.image0_1 - image_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

        image_aug = ia0.augmentize(self.imagecxr0_1)
        diff = self.imagecxr0_1 - image_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

    def test_ImageAugmentizer_Aug_UD(self):
        ia0 = ImageAugmentizer(self.conf_ud)
        image0_aug = ia0.augmentize(self.image0)
        diff = self.image0 - image0_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

        image0_aug = ia0.augmentize(self.imagecxr0)
        diff = self.imagecxr0 - image0_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

    def test_ImageAugmentizer_Aug_UD_4D(self):
        ia0 = ImageAugmentizer(self.conf_ud)
        image_aug = ia0.augmentize(self.image0_1)
        diff = self.image0_1 - image_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())

        image_aug = ia0.augmentize(self.imagecxr0_1)
        diff = self.imagecxr0_1 - image_aug
        self.assertFalse((diff == np.zeros_like(diff)).all())
