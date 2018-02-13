from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assignIfNotNull, returnPropertIfNotNull
from app.datasets.NormalizeConfig import NormalizeConfig


class ImageConfig(ConfigBase):
    _img_dim = 256
    _scale = 1. / 255
    _color_mode = "grayscale"

    @property
    def image_dir(self):
        return returnPropertIfNotNull(self.cp["IMAGE"].get("image_source_dir"))

    @property
    def color_mode(self):
        return assignIfNotNull(self.cp["IMAGE"].get("color_mode"), self._color_mode)

    @property
    def img_dim(self):
        return assignIfNotNull(self.cp["IMAGE"].getint("image_dimension"), self._img_dim)

    @property
    def scale(self):
        return assignIfNotNull(self.cp["IMAGE"].getfloat("scale"), self._scale)

    @property
    def NormalizeConfig(self):
        return NormalizeConfig(self.cp)
