from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import returnPropertIfNotNull, assignIfNotNull


class ModelConfig(ConfigBase):
    SECTION = "MODEL"

    _model_name = "densenet121"
    _use_base_model_weights = True
    _use_trained_model_weights = True
    _output_weights_name = ""
    _show_model_summary = True
    _use_best_weights = False

    @property
    def model_name(self):
        return assignIfNotNull(self.cp[self.SECTION].get("model_name"), self._model_name)

    @property
    def use_base_model_weights(self):
        return assignIfNotNull(self.cp[self.SECTION].getboolean("use_base_model_weights"), self._use_base_model_weights)

    @property
    def use_trained_model_weights(self):
        return assignIfNotNull(self.cp[self.SECTION].getboolean("use_trained_model_weights"),
                               self._use_trained_model_weights)

    @property
    def base_model_weights_file(self):
        if self.use_base_model_weights:
            return returnPropertIfNotNull(self.cp[self.SECTION].get("base_model_weights_file"))
        else:
            return None

    @property
    def use_best_weights(self):
        return assignIfNotNull(self.cp[self.SECTION].getboolean("use_best_weights"), self._use_trained_model_weights)

    @property
    def output_weights_name(self):
        return returnPropertIfNotNull(self.cp[self.SECTION].get("output_weights_name"))

    @property
    def show_model_summary(self):
        return assignIfNotNull(self.cp[self.SECTION].get("show_model_summary"), self._show_model_summary)
