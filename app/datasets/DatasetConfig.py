from app.imagetoolbox.ImageConfig import ImageConfig
from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assignIfNotNull, returnPropertIfNotNull
import os

class DatasetConfig(ConfigBase):
    _random_state = 0
    _train_ratio = 70
    _dev_ratio = 10
    _use_class_balancing = True
    _use_default_split = True
    _positive_weights_multiply = 1
    _force_resplit = False


    @property
    def ImageConfig(self):
        return ImageConfig(self.cp)

    @property
    def force_resplit(self):
        return assignIfNotNull(self.cp["DATASET"].getboolean("force_resplit"), self._force_resplit)

    @property
    def image_dir(self):
        return returnPropertIfNotNull(self.cp["IMAGE"].get("image_dir"))

    @property
    def data_entry(self):
        self.data_entry = returnPropertIfNotNull(self.cp["DATASET"].get("data_entry"))

    @property
    def data_entry_dir(self):
        os.path.dirname(self.data_entry)

    @property
    def class_names(self):
        self.class_names = returnPropertIfNotNull(self.cp["DATASET"].get("class_names"))

    @property
    def output_dir(self):
        self.output_dir = returnPropertIfNotNull(self.cp["DATASET"].get("output_dir"))

    @property
    def positive_weights_multiply(self):
        return assignIfNotNull(self.cp["DATASET"].getfloat("positive_weights_multiply"), self._positive_weights_multiply)

    @property
    def train_ratio(self):
        return assignIfNotNull(self.cp["DATASET"].getfloat("train_ratio"), self._train_ratio)

    @property
    def dev_ratio(self):
        return assignIfNotNull(self.cp["DATASET"].getfloat("dev_ratio"), self._dev_ratio)

    @property
    def class_mode(self):
        return returnPropertIfNotNull(self.cp["DATASET"].getfloat("class_mode"))

    @property
    def use_class_balancing(self):
        return assignIfNotNull(self.cp["DATASET"].getboolean("use_class_balancing"), self._use_class_balancing)

    @property
    def use_default_split(self):
        return assignIfNotNull(self.cp["DATASET"].getboolean("use_default_split"), self._use_default_split)