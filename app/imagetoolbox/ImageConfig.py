import app.datasets
from app.imagetoolbox.ImageConfigBase import ImageConfigBase
from app.utilities.util_config import assignIfNotNull, returnPropertIfNotNull


class ImageConfig(ImageConfigBase):
    _img_dim = 256
    _scale = 1. / 255

    @property
    def image_dir(self):
        return returnPropertIfNotNull(self.cp["IMAGE"].get("image_source_dir"))

    @property
    def img_dim(self):
        return assignIfNotNull(self.cp["IMAGE"].getint("image_dimension"), self._img_dim)

    @property
    def scale(self):
        return assignIfNotNull(self.cp["IMAGE"].getfloat("scale"), self._scale)

    @property
    def NormalizeConfig(self):
        return app.datasets.NormalizeConfig(self.cp)

    @property
    def AugmentConfig(self):
        return app.datasets.AugmentConfig(self.cp)
