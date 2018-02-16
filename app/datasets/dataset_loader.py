import os
import random
import shutil

import numpy as np
import pandas as pd
import sklearn as sk

from app.datasets.DatasetConfig import DatasetConfig
from app.datasets.dataset_utility import label2vec, pos_count, DataSequence, get_class_weights_multibinary


class DataSet:
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

        if self.dsconfig.force_resplit or not os.path.isfile(self.train_csv) or not os.path.isfile(
                self.dev_csv) or not os.path.isfile(self.test_csv):
            self.split_dataset()
        else:
            self.reload_dataset()

        output_fields = ["Image Index", "Patient ID", "Finding Labels"]
        self.train[output_fields].to_csv(self.train_csv, index=False)
        self.dev[output_fields].to_csv(self.dev_csv, index=False)
        self.test[output_fields].to_csv(self.test_csv, index=False)
        self.print_summary()

    def img_count(self, subdataset="all"):
        return_dict = {"train": len(self.train), "dev": len(self.dev),
                       "test": len(self.test),
                       "all": len(self.train) + len(self.dev) + len(self.test)}
        return return_dict[subdataset.lower()]

    def img_pos_count(self, subdataset="train"):
        return_dict = {"train": pos_count(self.train, self.dsconfig.class_names),
                       "dev": pos_count(self.dev, self.dsconfig.class_names),
                       "test": pos_count(self.test, self.dsconfig.class_names)}
        return return_dict[subdataset.lower()]

    def pat_count(self, subdataset="all"):
        return_dict = {"train": len(list(self.train["Patient ID"].unique())),
                       "dev": len(list(self.dev["Patient ID"].unique())),
                       "test": len(list(self.test["Patient ID"].unique())),
                       "all": len(list(self.train["Patient ID"].unique())) + len(
                           list(self.dev["Patient ID"].unique())) + len(list(self.test["Patient ID"].unique()))}
        return return_dict[subdataset.lower()]

    def print_summary(self):
        print("Total patients = {} ".format(self.pat_count(subdataset="all")), end="")
        print('in train/dev/test ', end="")
        print("{}/".format(self.pat_count(subdataset="train")), end="")
        print("{}/".format(self.pat_count(subdataset="dev")), end="")
        print("{}".format(self.pat_count(subdataset="test")))
        print("Total images = {} ".format(self.img_count(subdataset="all")), end="")
        print("in train/dev/test ", end="")
        print("{}/".format(self.img_count(subdataset="train")), end="")
        print("{}/".format(self.img_count(subdataset="dev")), end="")
        print("{}".format(self.img_count(subdataset="test")))

    def reload_dataset(self):
        print(f"Reloading splitted datasets from {self.train_csv}")
        self.train = pd.read_csv(self.train_csv)
        print(f"Reloading splitted datasets from {self.dev_csv}")
        self.dev = pd.read_csv(self.dev_csv)
        print(f"Reloading splitted datasets from {self.test_csv}")
        self.test = pd.read_csv(self.test_csv)
        self.train["One_Hot_Labels"] = self.train["Finding Labels"].apply(
            lambda x: label2vec(x, self.dsconfig.class_names))
        self.dev["One_Hot_Labels"] = self.dev["Finding Labels"].apply(lambda x: label2vec(x, self.dsconfig.class_names))
        self.test["One_Hot_Labels"] = self.test["Finding Labels"].apply(
            lambda x: label2vec(x, self.dsconfig.class_names))

    def split_dataset(self):
        print(f"Splitting dataset for {self.dsconfig.class_mode}")
        os.makedirs(self.dsconfig.output_dir, exist_ok=True)
        e = pd.read_csv(self.dsconfig.data_entry)

        # one hot encode
        e["One_Hot_Labels"] = e["Finding Labels"].apply(lambda x: label2vec(x, self.dsconfig.class_names))

        # shuffle and split
        pid = list(e["Patient ID"].unique())
        total_patients = len(pid)
        train_patient_count = int(total_patients * self.dsconfig.train_ratio / 100)
        dev_patient_count = int(total_patients * self.dsconfig.dev_ratio / 100)

        random.seed(self.dsconfig.random_state)
        random.shuffle(pid)

        self.train = e[e["Patient ID"].isin(pid[:train_patient_count])]
        self.dev = e[e["Patient ID"].isin(pid[train_patient_count:train_patient_count + dev_patient_count])]
        self.test = e[e["Patient ID"].isin(pid[train_patient_count + dev_patient_count:])]

    def class_weights(self):
        print(f"** {self.dsconfig.class_mode} class_weights **")
        if self.dsconfig.class_mode == 'multiclass':
            class_id = range(len(self.dsconfig.class_names))
            class_weight_sk = sk.utils.class_weight.compute_class_weight('balanced', self.dsconfig.class_names,
                                                                         self.train["Finding Labels"].tolist())
            class_weight = dict(zip(class_id, class_weight_sk))
            for c, w in class_weight.items():
                print(f"  {c} [{self.dsconfig.class_names[c]}]: {w}")
        elif self.dsconfig.class_mode == 'multibinary':
            class_weight = get_class_weights_multibinary(
                self.img_count(subdataset="train"),
                self.img_pos_count(subdataset="train"),
                multiply=self.dsconfig.positive_weights_multiply,
                use_class_balancing=self.dsconfig.use_class_balancing)
            for c, w in class_weight.items():
                print(f" {c}: ", end="")
                for wk, wv in w.items():
                    print(f"{wk}: {np.round(wv,2)} ", end="")
                print("")

        return class_weight

    def train_generator(self, verbosity=0):
        batch = self.train.sample(frac=1)  # shuffle
        return DataSequence(batch, image_config=self.dsconfig.ImageConfig, set_name='train',
                            verbosity=verbosity)

    def dev_generator(self, verbosity=0):
        batch = self.dev.sample(frac=1)  # shuffle
        return DataSequence(batch, image_config=self.dsconfig.ImageConfig, set_name='dev',
                            verbosity=verbosity)


class DataSetTest:
    def __init__(self, dsconfig):
        """

        :param dsconfig:
        :type dsconfig: DatasetConfig
        """
        self.dsconfig = dsconfig
        self.test = pd.read_csv(self.dsconfig.data_entry)

        # one hot encode
        self.test["One_Hot_Labels"] = self.test["Finding Labels"].apply(
            lambda x: label2vec(x, self.dsconfig.class_names))
        pid = list(self.test["Patient ID"].unique())
        self.test_patient_count = len(pid)
        self.test_count = len(self.test)

        print(f"Total patients for test = {self.test_patient_count}")
        print(f"Total images  for test = {self.test_count}")

    def test_generator(self, verbosity=0):
        batch = self.test.sample(frac=1)  # shuffle
        return DataSequence(batch, image_config=self.dsconfig.ImageConfig, verbosity=verbosity)
