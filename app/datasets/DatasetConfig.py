import os

from app.imagetoolbox.ImageConfig import ImageConfig
from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assignIfNotNull, returnPropertIfNotNull


class DatasetConfig(ConfigBase):
    _random_state = 0
    _train_ratio = 70
    _dev_ratio = 10
    _use_class_balancing = True
    _use_default_split = True
    _positive_weights_multiply = 1

    @property
    def ImageConfig(self):
        return ImageConfig(self.cp)

    @property
    def data_entry(self):
        return returnPropertIfNotNull(self.cp["DATASET"].get("data_entry_file"))

    @property
    def data_entry_dir(self):
        return os.path.dirname(self.data_entry)

    @property
    def class_names(self):
        return returnPropertIfNotNull(self.cp["DATASET"].get("class_names").split(","))

    @property
    def positive_weights_multiply(self):
        return assignIfNotNull(self.cp["DATASET"].getfloat("positive_weights_multiply"),
                               self._positive_weights_multiply)

    @property
    def train_ratio(self):
        return assignIfNotNull(self.cp["DATASET"].getfloat("train_patient_ratio"), self._train_ratio)

    @property
    def dev_ratio(self):
        return assignIfNotNull(self.cp["DATASET"].getfloat("dev_patient_ratio"), self._dev_ratio)

    @property
    def use_class_balancing(self):
        return assignIfNotNull(self.cp["DATASET"].getboolean("use_class_balancing"), self._use_class_balancing)

    @property
    def use_default_split(self):
        return assignIfNotNull(self.cp["DATASET"].getboolean("use_default_split"), self._use_default_split)

    @property
    def random_state(self):
        return assignIfNotNull(self.cp["DATASET"].getint("split_dataset_random_state"), self._random_state)
