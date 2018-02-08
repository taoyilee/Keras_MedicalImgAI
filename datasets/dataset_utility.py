import math
import os

import cv2
import numpy as np
from keras.utils import Sequence


class DataSequence(Sequence):
    def __init__(self, batch, image_dir, set_name, batch_size=16, verbosity=0, scale=1. / 255, img_dim=256,
                 color_mode='grayscale', class_mode='multiclass'):
        self.batch_size = batch_size
        self.batch = batch
        self.image_dir = image_dir
        self.verbosity = verbosity
        self.scale = scale
        self.img_dim = img_dim
        self.set_name = set_name
        self.color_mode = color_mode
        self.class_mode = class_mode

    def __len__(self):
        return math.ceil(self.batch.shape[0] / self.batch_size)

    def targets(self):
        return self.batch["One_Hot_Labels"].tolist()

    def inputs(self, cam=False):
        return image_generator(image_filenames=self.batch["Image Index"], image_dir=self.image_dir,
                               img_dim=self.img_dim, scale=self.scale,
                               colormode=self.color_mode)

    def __getitem__(self, idx):
        return common_generator(self.batch.iloc[idx * self.batch_size:(idx + 1) * self.batch_size],
                                image_dir=self.image_dir, set_name=self.set_name,
                                verbosity=self.verbosity, scale=self.scale, img_dim=self.img_dim,
                                class_mode=self.class_mode)


def common_generator(batch, image_dir, set_name, verbosity=0, scale=1. / 255, img_dim=256,
                     class_mode='multiclass'):
    if verbosity > 0:
        print(
            f'** now yielding {set_name} batch = {batch["Patient ID"].tolist()}\nimages are = {batch["Image Index"].tolist()}')

    return batch_generator(batch["Image Index"],
                           batch["One_Hot_Labels"].tolist(), image_dir=image_dir,
                           class_mode=class_mode,
                           img_dim=img_dim, scale=scale, verbosity=verbosity)


def pos_count(subset_series, class_names):
    ret_dict = dict()
    one_hot_labels = np.array(subset_series["One_Hot_Labels"].tolist())
    for i, c in enumerate(class_names):
        ret_dict[c] = np.int(np.sum(one_hot_labels[:, i]))
    print(f"{ret_dict}")
    return ret_dict


def image_generator(image_filenames, image_dir, img_dim=256, scale=1. / 255, colormode='grayscale'):
    if colormode == 'grayscale':
        inputs = np.array(
            image_filenames.apply(lambda x: load_image(x, image_dir, img_dim=img_dim, scale=scale)).tolist())[:, :, :,
                 np.newaxis]
    else:
        inputs = np.array(
            image_filenames.apply(lambda x: load_image(x, image_dir, img_dim=img_dim, scale=scale)).tolist())
    return inputs


def batch_generator(image_filenames, labels, image_dir, img_dim=256, scale=1. / 255, colormode='grayscale',
                    class_mode='multiclass', verbosity=0):
    inputs = image_generator(image_filenames=image_filenames, image_dir=image_dir, img_dim=img_dim, scale=scale,
                             colormode=colormode)

    if class_mode == 'multiclass':
        targets = np.array(labels)
    elif class_mode == 'multibinary':
        targets = np.swapaxes(labels, 0, 1)
        targets = [np.array(targets[i, :]) for i in range(np.shape(targets)[0])]

    if verbosity > 1:
        print(f"targets({np.shape(targets)}) = {targets}")
    return inputs, targets


def load_image(image_name, image_dir, img_dim=256, scale=1. / 255, colormode='grayscale', verbosity=0):
    image_file = image_dir + "/" + image_name;
    if not os.path.isfile(image_file):
        raise Exception(f"{image_file} not found")
    if verbosity > 1:
        print(f"Load image from {image_file}")
    if colormode == 'grayscale':
        image = cv2.imread(image_file, 0)[:, :, np.newaxis]
    else:
        image = cv2.imread(image_file, 1)
    image = cv2.resize(image, (img_dim, img_dim))
    return image * scale


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
