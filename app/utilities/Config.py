from app.datasets import DatasetConfig
from app.imagetoolbox.ImageConfig import ImageConfig
from app.models.ModelConfig import ModelConfig
from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assign_fallback, assign_raise


class Config(ConfigBase):
    _epochs = 20
    _verbosity = 0
    _progress_verbosity = 1
    _initial_learning_rate = 0.0001
    _train_steps = None
    _test_steps = None
    _validation_steps = "auto"
    _patience_reduce_lr = 1
    _gpu = 1
    _enable_grad_cam = True

    def __init__(self, cp):
        super().__init__(cp=cp)
        self._initial_learning_rate = assign_fallback(self.cp["TRAIN"].getfloat("initial_learning_rate"),
                                                      self._initial_learning_rate)
        config_train_steps = self.cp["TRAIN"].get("train_steps")
        if config_train_steps != "auto" and config_train_steps is not None:
            try:
                config_train_steps = int(config_train_steps)
            except ValueError:
                raise ValueError(
                    "** train_steps: {} is invalid,please use 'auto' or integer.".format(config_train_steps))

        config_validation_steps = assign_fallback(self.cp["TRAIN"].get("validation_steps"), self._validation_steps)
        if config_validation_steps != "auto" and config_validation_steps is not None:
            try:
                config_validation_steps = int(config_validation_steps)
            except ValueError:
                raise ValueError(
                    "** validation_steps: {} is invalid,please use 'auto' or integer.".format(config_validation_steps))

        config_test_steps = assign_fallback(self.cp["TEST"].get("steps"), self._test_steps)
        if config_test_steps != "auto" and config_test_steps is not None:
            try:
                config_test_steps = int(config_test_steps)
            except ValueError:
                raise ValueError("** test_steps: {} is invalid,please use 'auto' or integer.".format(config_test_steps))

        self._train_steps = assign_fallback(config_train_steps, self._train_steps)
        self._validation_steps = assign_fallback(config_validation_steps, self._validation_steps)
        self._test_steps = assign_fallback(config_test_steps, self._test_steps)

    @property
    def initial_learning_rate(self):
        return self._initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, value):
        self._initial_learning_rate = value

    @property
    def ImageConfig(self):
        return ImageConfig(self.cp)

    @property
    def patience_reduce_lr(self):
        return assign_fallback(self.cp["TRAIN"].getint("patience_reduce_lr"), self._patience_reduce_lr)

    @property
    def gpu(self):
        return assign_fallback(self.cp["TRAIN"].getint("gpu"), self._gpu)

    @property
    def verbosity(self):
        return assign_fallback(self.cp["DEFAULT"].getint("verbosity"), self._verbosity)

    @property
    def epochs(self):
        return assign_fallback(self.cp["TRAIN"].getint("epochs"), self._epochs)

    @property
    def grad_cam_outputdir(self):
        return assign_raise(self.cp["TEST"].get("grad_cam_outputdir"))

    @property
    def train_steps(self):
        return self._train_steps

    @train_steps.setter
    def train_steps(self, value):
        self._train_steps = value

    @property
    def test_steps(self):
        return self._test_steps

    @test_steps.setter
    def test_steps(self, value):
        self._test_steps = value

    @property
    def validation_steps(self):
        return self._validation_steps

    @validation_steps.setter
    def validation_steps(self, value):
        self._validation_steps = value

    @property
    def progress_train_verbosity(self):
        return assign_fallback(self.cp["TRAIN"].getint("progress_verbosity"), self._progress_verbosity)

    @property
    def progress_test_verbosity(self):
        return assign_fallback(self.cp["TEST"].getint("progress_verbosity"), self._progress_verbosity)

    @property
    def DatasetConfig(self):
        return DatasetConfig(self.cp)

    @property
    def ModelConfig(self):
        return ModelConfig(self.cp)

    @property
    def enable_grad_cam(self):
        return assign_fallback(self.cp["TEST"].getboolean("enable_grad_cam"), self._enable_grad_cam)
