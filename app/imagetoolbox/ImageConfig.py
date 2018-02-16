import app.datasets
from app.imagetoolbox.ImageConfigBase import ImageConfigBase
from app.utilities.util_config import assign_fallback, assign_raise


class ImageConfig(ImageConfigBase):
    _img_dim = 256
    _scale = 1. / 255

    @property
    def image_dir(self):
        return assign_raise(self.cp["IMAGE"].get("image_source_dir"))

    @property
    def img_dim(self):
        return assign_fallback(self.cp["IMAGE"].getint("image_dimension"), self._img_dim)

    @property
    def scale(self):
        return assign_fallback(self.cp["IMAGE"].getfloat("scale"), self._scale)

    @property
    def NormalizeConfig(self):
        return app.datasets.NormalizeConfig(self.cp)

    @property
    def AugmentConfig(self):
        return app.datasets.AugmentConfig(self.cp)
