import math
import os

import cv2
import numpy as np
from keras.utils import Sequence

from app.datasets.image_normalizer import ImageNormalizer
from app.imagetoolbox.ImageConfig import ImageConfig


class DataSequence(Sequence):
    def __init__(self, batch, image_config, set_name="train", verbosity=0):
        """

        :param batch:
        :param image_config:
        :type image_config: ImageConfig
        :param set_name:
        :param verbosity:
        """
        self.verbosity = verbosity
        self.set_name = set_name
        self.batch = batch
        self.image_config = image_config
        self.batch_size = image_config.batch_size

    def __len__(self):
        return math.ceil(self.batch.shape[0] / self.image_config.batch_size)

    def targets(self):
        return self.batch["One_Hot_Labels"].tolist()

    def inputs(self, index, mode="train"):
        return image_generator(self.batch["Image Index"].iloc[[index]], self.image_config, mode=mode,
                               verbosity=self.verbosity)

    def __getitem__(self, idx):
        slice0 = idx * self.batch_size
        slice1 = (idx + 1) * self.batch_size
        batchi = self.batch.iloc[slice0:slice1]
        if self.verbosity > 0:
            print(f'** now yielding {self.set_name} batch = {batchi["Patient ID"].tolist()}')
            print(f'** images are = {batchi["Image Index"].tolist()}')

        return batch_generator(batchi["Image Index"],
                               batchi["One_Hot_Labels"].tolist(), mode=self.set_name,
                               image_config=self.image_config,
                               verbosity=self.verbosity)


def pos_count(subset_series, class_names):
    ret_dict = dict()
    one_hot_labels = np.array(subset_series["One_Hot_Labels"].tolist())
    for i, c in enumerate(class_names):
        ret_dict[c] = np.int(np.sum(one_hot_labels[:, i]))
    print(f"{ret_dict}")
    return ret_dict


def image_generator(image_filenames, image_config, mode="train", verbosity=0):
    """

    :param image_filenames:
    :param image_config:
    :type image_config: ImageConfig
    :param verbosity:
    :param mode:
    :return:
    """
    if image_config.color_mode == 'grayscale':
        inputs = np.array(
            image_filenames.apply(
                lambda x: load_image(x, image_config, mode=mode, verbosity=verbosity)).tolist())[:, :, :, np.newaxis]
    else:
        inputs = np.array(
            image_filenames.apply(
                lambda x: load_image(x, image_config, mode=mode, verbosity=verbosity)).tolist())
    return inputs


def batch_generator(image_filenames, labels, image_config, mode="train", verbosity=0):
    """
    :param image_filenames:
    :param labels:
    :param image_config:
    :type image_config: ImageConfig
    :param verbosity:
    :return:
    """
    inputs = image_generator(image_filenames=image_filenames, image_config=image_config, mode=mode, verbosity=verbosity)

    targets = np.array(labels)

    if image_config.class_mode == 'multibinary':
        targets = np.swapaxes(labels, 0, 1)
        targets = [np.array(targets[i, :]) for i in range(np.shape(targets)[0])]

    if verbosity > 0:
        print(f"(input, targets) = ({np.shape(inputs)}, {np.shape(targets)})")

    return inputs, targets


def load_image(image_name, image_config, verbosity=0, mode="train"):
    """

    :param image_name:
    :param image_config:
    :type image_config: ImageConfig
    :param verbosity:
    :param mode: Generating mode: train, dev, test and raw
    :return:
    """
    image_file = image_config.image_dir + "/" + image_name
    if not os.path.isfile(image_file):
        raise Exception(f"{image_file} not found")
    if verbosity > 1:
        print(f"Load image from {image_file}")
    if image_config.color_mode == 'grayscale':
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    if mode != "raw":
        if image_config.img_dim is not None:
            image = cv2.resize(image, (image_config.img_dim, image_config.img_dim))
        if image_config.scale is not None:
            image = image * image_config.scale
        normalizer = ImageNormalizer(image_config.NormalizeConfig)
        normalizer.normalize(image)

    return image


def label2vec(label, class_names):
    vec = np.zeros(len(class_names))
    if label == "No Finding":
        return vec
    labels = label.split("|")
    for l in labels:
        vec[class_names.index(l)] = 1
    return vec


def get_class_weights_multibinary(total_counts, class_positive_counts, multiply, use_class_balancing):
    """
    Calculate class_weight used in training

    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean

    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """

    def get_single_class_weight(pos_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        # print(f"Total counts = {total_counts}, Positive counts = {pos_counts}")
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    def balancing(class_weights, label_counts, multiply=10):
        """
        Normalize the class_weights so that each class has the same impact to backprop

        ex: label_counts: [1, 2, 3] -> factor: [1, 1/2, 1/3] * len(label_counts) / (1+1/2+1/3)
        """
        balanced = {}
        # compute factor
        reciprocal = np.reciprocal(label_counts.astype(float))
        factor = reciprocal * len(label_counts) * multiply / np.sum(reciprocal)

        # multiply by factor
        i = 0
        for c, w in class_weights.items():
            balanced[c] = {
                0: w[0] * factor[i],
                1: w[1] * factor[i],
            }
            i += 1
        return balanced

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = {}
    for i, class_name in enumerate(class_names):
        class_weights[class_name] = get_single_class_weight(label_counts[i])

    if use_class_balancing:
        class_weights = balancing(class_weights, label_counts)
    return class_weights
