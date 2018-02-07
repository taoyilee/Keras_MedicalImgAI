import math
import os

import cv2
import numpy as np
from keras.utils import Sequence


class DataSequence(Sequence):
    def __init__(self, batch, image_dir, batch_size=16, verbosity=0, scale=1. / 255, img_dim=256):
        self.batch_size = batch_size
        self.batch = batch
        self.image_dir = image_dir
        self.verbosity = verbosity
        self.scale = scale
        self.img_dim = img_dim

    def __len__(self):
        return math.ceil(self.batch.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        return common_generator(self.batch.iloc[idx * self.batch_size:(idx + 1) * self.batch_size],
                                image_dir=self.image_dir,
                                verbosity=self.verbosity, scale=self.scale, img_dim=self.img_dim)


def common_generator(batch, image_dir, verbosity=0, scale=1. / 255, img_dim=256):
    if verbosity > 0:
        print(f'** now yielding batch = {batch["Patient ID"].tolist()}\nimages are = {batch["Image Index"].tolist()}')

    return batch_generator(batch["Image Index"],
                           batch["One_Hot_Labels"].tolist(), image_dir=image_dir,
                           img_dim=img_dim, scale=scale, verbosity=verbosity)


def pos_count(subset_series, class_names):
    ret_dict = dict()
    one_hot_labels = np.array(subset_series["One_Hot_Labels"].tolist())
    for i, c in enumerate(class_names):
        ret_dict[c] = np.int(np.sum(one_hot_labels[:, i]))
    print(f"{ret_dict}")
    return ret_dict


def batch_generator(image_filenames, labels, image_dir, img_dim=256, scale=1. / 255, colormode='grayscale',
                    verbosity=0):
    if colormode == 'grayscale':
        inputs = np.array(
            image_filenames.apply(lambda x: load_image(x, image_dir, img_dim=img_dim, scale=scale)).tolist())[:, :, :,
                 np.newaxis]
    else:
        inputs = np.array(
            image_filenames.apply(lambda x: load_image(x, image_dir, img_dim=img_dim, scale=scale)).tolist())
    targets = np.swapaxes(labels, 0, 1)
    targets = [np.array(targets[i, :]) for i in range(np.shape(targets)[0])]
    if verbosity > 1:
        print(f"targets = {targets}")
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
