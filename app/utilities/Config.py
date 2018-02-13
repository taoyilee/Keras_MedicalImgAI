import configparser
import os

from app.datasets.DatasetConfig import DatasetConfig
from app.imagetoolbox.ImageConfig import ImageConfig
from app.models.ModelConfig import ModelConfig
from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assignIfNotNull


class Config(ConfigBase):
    _epochs = 20
    _verbosity = 0
    _progress_verbosity = 1
    _batch_size = 32
    _initial_learning_rate = 0.0001

    def __init__(self, cp, config_file=None):
        """

        :param cp: Config Parser
        :type cp: configparser.ConfigParser
        """
        if config_file is not None:
            print(f"** Reading config {config_file}")
            self.cp = configparser.ConfigParser(config_file=config_file)
        if cp is not None:
            super().__init__(cp)

    @property
    def initial_learning_rate(self):
        return assignIfNotNull(self.cp["TRAIN"].getfloat("initial_learning_rate"), self._initial_learning_rate)

    @initial_learning_rate.setter
    def initial_learning_rate(self, value):
        self._initial_learning_rate = value

    @property
    def train_stats_file(self):
        return os.path.join(self.output_dir, ".training_stats.json")

    @property
    def isResumeMode(self):
        return not self.DatasetConfig.force_resplit and self.ModelConfig.use_trained_model_weights

    @property
    def ImageConfig(self):
        return ImageConfig(self.cp)

    @property
    def DatasetConfig(self):
        return DatasetConfig(self.cp)

    @property
    def ModelConfig(self):
        return ModelConfig(self.cp)


    @property
    def verbosity(self):
        return assignIfNotNull(self.cp["DEFAULT"].getint("verbosity"), self._verbosity)

    @property
    def epochs(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("epochs"), self._epochs)

    @property
    def batch_size(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("batch_size"), self._batch_size)

    @property
    def progress_verbosity(self, phase="train"):
        """

        :param phase:
        :type phase: str
        :return:
        """
        return_dict = {"train": assignIfNotNull(self.cp["TRAIN"].get("progress_verbosity"), self._progress_verbosity),
                       "test": assignIfNotNull(self.cp["TEST"].get("progress_verbosity"), self._progress_verbosity)}
        return return_dict[phase.lower()]
