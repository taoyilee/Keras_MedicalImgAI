import configparser

from app.datasets.DatasetConfig import DatasetConfig
from app.imagetoolbox.ImageConfig import ImageConfig
from app.models.ModelConfig import ModelConfig
from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assignIfNotNull


class Config(ConfigBase):
    _epochs = 20
    _verbosity = 0
    _progress_verbosity = 1
    _initial_learning_rate = 0.0001
    _train_steps = "auto"
    _validation_steps = "auto"
    _patience_reduce_lr = 1
    _gpu = 1

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
    def ImageConfig(self):
        return ImageConfig(self.cp)

    @property
    def DatasetConfig(self):
        return DatasetConfig(self.cp)

    @property
    def ModelConfig(self):
        return ModelConfig(self.cp)

    @property
    def patience_reduce_lr(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("patience_reduce_lr"), self._patience_reduce_lr)

    @property
    def gpu(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("gpu"), self._gpu)

    @property
    def verbosity(self):
        return assignIfNotNull(self.cp["DEFAULT"].getint("verbosity"), self._verbosity)

    @property
    def epochs(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("epochs"), self._epochs)

    @property
    def train_steps(self):
        train_steps = assignIfNotNull(self.cp["TRAIN"].get("train_steps"), self._train_steps)
        if train_steps != "auto":
            try:
                train_steps = int(self.cp["TRAIN"].get("train_steps"))
            except ValueError:
                raise ValueError("** train_steps: {} is invalid,please use 'auto' or integer.".format(
                    self.cp["TRAIN"].get("train_steps")))

            self._train_steps = assignIfNotNull(train_steps, self._train_steps)
        return self._train_steps

    @train_steps.setter
    def train_steps(self, value):
        self._train_steps = value

    @property
    def validation_steps(self):
        validation_steps = assignIfNotNull(self.cp["TRAIN"].get("validation_steps"), self._validation_steps)
        if validation_steps != "auto":
            try:
                validation_steps = int(self.cp["TRAIN"].get("validation_steps"))
            except ValueError:
                raise ValueError("** validation_steps: {} is invalid,please use 'auto' or integer.".format(
                    self.cp["TRAIN"].get("validation_steps")))

            self._validation_steps = assignIfNotNull(validation_steps, self._validation_steps)
        return self._validation_steps

    @validation_steps.setter
    def validation_steps(self, value):
        self._validation_steps = value

    @property
    def progress_train_verbosity(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("progress_verbosity"), self._progress_verbosity)

    @property
    def progress_test_verbosity(self):
        return assignIfNotNull(self.cp["TEST"].getint("progress_verbosity"), self._progress_verbosity)
