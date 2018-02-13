import os
import random
import shutil

import pandas as pd
import sklearn as sk

from app.datasets.DatasetConfig import DatasetConfig
from app.datasets.dataset_utility import label2vec, pos_count, DataSequence, get_class_weights_multibinary


class DataSet:
    total_patients = 0
    total_images = 0
    train = []
    dev = []
    test = []

    def __init__(self, dsconfig):
        """
        :param dsconfig:
        :type dsconfig: DatasetConfig
        """
        self.dsconfig = dsconfig

        self.train_csv = os.path.join(dsconfig.output_dir, "train.csv")
        self.dev_csv = os.path.join(dsconfig.output_dir, "dev.csv")
        self.test_csv = os.path.join(dsconfig.output_dir, "test.csv")

        if dsconfig.use_default_split:  # split train/dev/test
            datasets = ["train", "dev", "test"]
            for d in datasets:
                shutil.copy(f"{self.DSConfig.data_entry_dir}/{d}.csv", self.conf.output_dir)

        if self.force_resplit or not os.path.isfile(self.train_csv) or not os.path.isfile(
                self.dev_csv) or not os.path.isfile(self.test_csv):
            self.split_dataset()
        else:
            self.reload_dataset()

        self.train_count = len(self.train)
        self.dev_count = len(self.dev)
        self.test_count = len(self.test)
        self.train_pos_count = pos_count(self.train, self.class_names)
        self.dev_pos_count = pos_count(self.dev, self.class_names)
        self.test_pos_count = pos_count(self.test, self.class_names)
        self.print_summary()
        # export csv
        output_fields = ["Image Index", "Patient ID", "Finding Labels"]
        self.train[output_fields].to_csv(self.train_csv, index=False)
        self.dev[output_fields].to_csv(self.dev_csv, index=False)
        self.test[output_fields].to_csv(self.test_csv, index=False)

    def print_summary(self):
        print(
            f"Total patients = {self.total_patients} in train/dev/test {train_patient_count}/{dev_patient_count}/{test_patient_count}")
        print(
            f"Total images = {self.total_images} in train/dev/test {self.train_count}/{self.dev_count}/{self.test_count}")

    def reload_dataset(self):
        print(f"Reloading splitted datasets from {self.train_csv}")
        self.train = pd.read_csv(self.train_csv)
        print(f"Reloading splitted datasets from {self.dev_csv}")
        self.dev = pd.read_csv(self.dev_csv)
        print(f"Reloading splitted datasets from {self.test_csv}")
        self.test = pd.read_csv(self.test_csv)
        self.train["One_Hot_Labels"] = self.train["Finding Labels"].apply(lambda x: label2vec(x, self.class_names))
        self.dev["One_Hot_Labels"] = self.dev["Finding Labels"].apply(lambda x: label2vec(x, self.class_names))
        self.test["One_Hot_Labels"] = self.test["Finding Labels"].apply(lambda x: label2vec(x, self.class_names))

        pid_train = list(self.train["Patient ID"].unique())
        pid_dev = list(self.dev["Patient ID"].unique())
        pid_test = list(self.test["Patient ID"].unique())
        train_patient_count = len(pid_train)
        dev_patient_count = len(pid_dev)
        test_patient_count = len(pid_test)

        self.total_patients = train_patient_count + dev_patient_count + test_patient_count
        self.total_images = len(self.train.index) + len(self.dev.index) + len(self.test.index)

    def split_dataset(self):
        print(f"Splitting dataset for {self.class_mode}")
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

    def class_weights(self):
        print(f"** {self.class_mode} class_weights **")
        if self.class_mode == 'multiclass':
            class_id = range(len(self.class_names))
            class_weight_sk = sk.utils.class_weight.compute_class_weight('balanced', self.class_names,
                                                                         self.train["Finding Labels"].tolist())
            class_weight = dict(zip(class_id, class_weight_sk))
            for c, w in class_weight.items():
                print(f"  {c}: {w}")
            return class_weight

        elif self.class_mode == 'multibinary':
            class_weight = get_class_weights_multibinary(
                self.train_count,
                self.train_pos_count,
                multiply=self.positive_weights_multiply,
                use_class_balancing=self.use_class_balancing)
            for c, w in class_weight.items():
                print(f"  {c}: {w}")

            return class_weight

    def train_generator(self, verbosity=0):
        batch = self.train.sample(frac=1)  # shuffle
        return DataSequence(batch, image_dir=self.image_dir, set_name='train',
                            img_dim=self.img_dim, scale=self.scale, class_mode=self.class_mode, verbosity=verbosity)

    def dev_generator(self, verbosity=0):
        batch = self.dev.sample(frac=1)  # shuffle
        return DataSequence(batch, image_dir=self.image_dir, set_name='dev',
                            img_dim=self.img_dim, scale=self.scale, class_mode=self.class_mode, verbosity=verbosity)


class DataSetTest:
    def __init__(self, image_dir, data_entry, class_names, batch_size=16, img_dim=256, scale=1. / 255,
                 class_mode="multiclass"):
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
        self.class_mode = class_mode

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
                            img_dim=self.img_dim, scale=self.scale, class_mode=self.class_mode, verbosity=verbosity)
