import os
import random

import pandas as pd

from datasets.dataset_utility import label2vec, pos_count, DataSequence


class DataSet:
    def __init__(self, image_dir, data_entry, class_names, output_dir, random_state=0,
                 train_ratio=70, dev_ratio=10, batch_size=16, img_dim=256, scale=1. / 255):
        """Loads Chexnet dataset.
        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        self.image_dir = image_dir
        self.data_entry = data_entry
        self.class_names = class_names
        self.output_dir = output_dir
        self.random_state = random_state
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.scale = scale

        os.makedirs(self.output_dir, exist_ok=True)
        e = pd.read_csv(self.data_entry)

        # one hot encode
        e["One_Hot_Labels"] = e["Finding Labels"].apply(lambda x: label2vec(x, self.class_names))

        # shuffle and split
        pid = list(e["Patient ID"].unique())
        total_patients = len(pid)
        train_patient_count = int(total_patients * train_ratio / 100)
        dev_patient_count = int(total_patients * dev_ratio / 100)
        test_patient_count = total_patients - train_patient_count - dev_patient_count

        random.seed(self.random_state)
        random.shuffle(pid)
        self.train = e[e["Patient ID"].isin(pid[:train_patient_count])]
        self.dev = e[e["Patient ID"].isin(pid[train_patient_count:train_patient_count + dev_patient_count])]
        self.test = e[e["Patient ID"].isin(pid[train_patient_count + dev_patient_count:])]
        total_images = len(e)
        self.train_count = len(self.train)
        self.dev_count = len(self.dev)
        self.test_count = len(self.test)
        self.train_pos_count = pos_count(self.train, self.class_names)
        self.dev_pos_count = pos_count(self.dev, self.class_names)
        self.test_pos_count = pos_count(self.test, self.class_names)

        print(
            f"Total patients = {total_patients} in train/dev/test {train_patient_count}/{dev_patient_count}/{test_patient_count}")
        print(f"Total images = {total_images} in train/dev/test {self.train_count}/{self.dev_count}/{self.test_count}")

        # export csv
        output_fields = ["Image Index", "Patient ID", "Finding Labels", "One_Hot_Labels"]
        self.train[output_fields].to_csv(os.path.join(output_dir, "train.csv"), index=False)
        self.dev[output_fields].to_csv(os.path.join(output_dir, "dev.csv"), index=False)
        self.test[output_fields].to_csv(os.path.join(output_dir, "test.csv"), index=False)

    def train_generator(self, verbosity=0):
        batch = self.train.sample(frac=1)  # shuffle
        return DataSequence(batch, image_dir=self.image_dir, set_name='train',
                            img_dim=self.img_dim, scale=self.scale, verbosity=verbosity)

    def dev_generator(self, verbosity=0):
        batch = self.dev.sample(frac=1)  # shuffle
        return DataSequence(batch, image_dir=self.image_dir, set_name='dev',
                            img_dim=self.img_dim, scale=self.scale, verbosity=verbosity)

    def test_generator(self, verbosity=0):
        batch = self.test.sample(frac=1)  # shuffle
        return DataSequence(batch, image_dir=self.image_dir, set_name='test',
                            img_dim=self.img_dim, scale=self.scale, verbosity=verbosity)


class DataSetTest:
    def __init__(self, image_dir, data_entry, class_names, batch_size=16, img_dim=256, scale=1. / 255):
        """Loads Chexnet dataset.
        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        self.image_dir = image_dir
        self.data_entry = data_entry
        self.class_names = class_names
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.scale = scale

        self.test = pd.read_csv(self.data_entry)

        # one hot encode
        self.test["One_Hot_Labels"] = self.test["Finding Labels"].apply(lambda x: label2vec(x, self.class_names))
        pid = list(self.test["Patient ID"].unique())
        self.test_patient_count = len(pid)
        self.test_count = len(self.test)

        print(
            f"Total patients for test = {self.test_patient_count}")
        print(f"Total images  for test = {self.test_count}")

    def test_generator(self, verbosity=0):
        batch = self.test.sample(frac=1)  # shuffle
        return DataSequence(batch, image_dir=self.image_dir, set_name='test',
                            img_dim=self.img_dim, scale=self.scale, verbosity=verbosity)
