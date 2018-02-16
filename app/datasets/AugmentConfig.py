from app.imagetoolbox.ImageConfigBase import ImageConfigBase
from app.utilities.util_config import returnPropertIfNotNull


class AugmentConfig(ImageConfigBase):
    SECTION = "IMAGE-AUGMENATION"

    _train_augmentation = True
    _dev_augmentation = False

    @property
    def train_augmentation(self):
        return returnPropertIfNotNull(self.cp[self.SECTION].getboolean("train_augmentation"))

    @property
    def dev_augmentation(self):
        return returnPropertIfNotNull(self.cp[self.SECTION].getboolean("dev_augmentation"))
