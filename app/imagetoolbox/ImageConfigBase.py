from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assignIfNotNull


class ImageConfigBase(ConfigBase):
    _color_mode = "grayscale"

    @property
    def color_mode(self):
        return assignIfNotNull(self.cp["IMAGE"].get("color_mode"), self._color_mode)
