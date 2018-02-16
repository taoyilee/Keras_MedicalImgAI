import configparser
import os

from app.datasets.DatasetConfig import DatasetConfig
from app.models.ModelConfig import ModelConfig
from app.utilities.util_config import returnPropertIfNotNull, assignIfNotNull


class ConfigBase:
    _batch_size = 32

    def __init__(self, cp):
        """

        :param cp: File name of config.ini or a ConfigParser
        """

        if os.path.isfile(cp):
            print(f"** Reading config file {cp}")
            self.cp = configparser.ConfigParser()
            self.cp.read(cp)
        else:
            self.cp = cp

    @property
    def output_dir(self):
        return returnPropertIfNotNull(self.cp["DEFAULT"].get("output_dir"))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    @property
    def class_mode(self):
        return returnPropertIfNotNull(self.cp["DATASET"].get("class_mode"))

    @property
    def batch_size(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("batch_size"), self._batch_size)

    @property
    def train_stats_file(self):
        return os.path.join(self.output_dir, "training_stats.json")

    @property
    def DatasetConfig(self):
        return DatasetConfig(self.cp)

    @property
    def ModelConfig(self):
        return ModelConfig(self.cp)

    @property
    def isResumeMode(self):
        return not self.DatasetConfig.force_resplit and self.ModelConfig.use_trained_model_weights
