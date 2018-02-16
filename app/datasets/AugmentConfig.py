import configparser

from app.imagetoolbox.ImageConfigBase import ImageConfigBase
from app.utilities.util_config import assign_fallback


class AugmentConfig(ImageConfigBase):
    SECTION = "IMAGE-AUGMENATION"

    _train_augmentation = True
    _dev_augmentation = False
    _random_horz_flip = False
    _random_vert_flip = False
    _flip_prob = 0.5

    def __init__(self, cp: configparser.ConfigParser):
        super(AugmentConfig, self).__init__(cp)
        if self.SECTION not in self.cp.sections():
            self.cp.read_dict({self.SECTION: {}})

    @property
    def train_augmentation(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("train_augmentation"), self._train_augmentation)

    @property
    def dev_augmentation(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("dev_augmentation"), self._dev_augmentation)

    @property
    def flip_prob(self):
        return assign_fallback(self.cp[self.SECTION].getfloat("flip_prob"), self._flip_prob)

    @property
    def random_horz_flip(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("random_horz_flip"), self._random_horz_flip)

    @property
    def random_vert_flip(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("random_vert_flip"), self._random_vert_flip)
