import numpy as np

from app.imagetoolbox.ImageConfig import ImageConfig
from app.utilities.util_config import returnPropertIfNotNull, assignIfNotNull


class NormalizeConfig(ImageConfig):
    SECTION = "IMAGE-PREPROCESSING"
    _img_dim = 256
    _scale = 1. / 255
    _color_mode = "grayscale"
    _normalize_mean = np.array([0.485, 0.456, 0.406])
    _normalize_stdev = np.array([0.229, 0.224, 0.225])
    _normalize_samplewise = False

    @property
    def normalize_by_mean_var(self):
        return returnPropertIfNotNull(self.cp[self.SECTION].getboolean("normalize_by_mean_var"))

    @property
    def normalize_samplewise(self):
        return assignIfNotNull(self.cp[self.SECTION].getboolean("normalize_samplewise"), self._normalize_samplewise)

    @property
    def norm_mean(self):
        if self.color_mode == "rgb" or self.color_mode == "hsv":
            mean_chan1 = assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_mean_chan1"),
                                         self._normalize_mean[0])
            mean_chan2 = assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_mean_chan2"),
                                         self._normalize_mean[1])
            mean_chan3 = assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_mean_chan3"),
                                         self._normalize_mean[2])
            return np.array([mean_chan1, mean_chan2, mean_chan3])
        elif self.color_mode == "grayscale":
            grayscale_mean = np.mean(self._normalize_mean)
            return assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_mean"), grayscale_mean)

    @property
    def norm_stdev(self):
        if self.color_mode == "rgb" or self.color_mode == "hsv":
            stdev_chan1 = assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_stdev_chan1"),
                                          self._normalize_stdev[0])
            stdev_chan2 = assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_stdev_chan2"),
                                          self._normalize_stdev[1])
            stdev_chan3 = assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_stdev_chan3"),
                                          self._normalize_stdev[2])
            return np.array([stdev_chan1, stdev_chan2, stdev_chan3])
        elif self.color_mode == "grayscale":
            grayscale_stdev = np.mean(self._normalize_stdev)
            return assignIfNotNull(self.cp[self.SECTION].getfloat("normalize_stdev"), grayscale_stdev)
