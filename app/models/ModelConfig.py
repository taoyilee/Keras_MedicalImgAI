import os

from app.utilities.ConfigBase import ConfigBase
from app.utilities.util_config import assign_raise, assign_fallback


class ModelConfig(ConfigBase):
    SECTION = "MODEL"

    _model_name = "densenet121"
    _use_base_model_weights = True
    _use_ext_base_model_weights = False
    _use_trained_model_weights = True
    _output_weights_name = ""
    _show_model_summary = False
    _use_best_weights = False

    def __init__(self, cp):
        super().__init__(cp)
        self._use_trained_model_weights = assign_fallback(self.cp[self.SECTION].getboolean("use_trained_model_weights"),
                                                          self._use_trained_model_weights)

    @property
    def model_name(self):
        return assign_fallback(self.cp[self.SECTION].get("model_name"), self._model_name)

    @property
    def use_ext_base_model_weights(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("use_ext_base_model_weights"),
                               self._use_ext_base_model_weights)

    @property
    def use_base_model_weights(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("use_base_model_weights"), self._use_base_model_weights)

    @property
    def use_trained_model_weights(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("use_trained_model_weights"),
                               self._use_trained_model_weights)

    @property
    def is_resume_mode(self):
        return not self.force_resplit and self.use_trained_model_weights

    @property
    def trained_model_weights(self):
        trained_model = None
        if self._use_trained_model_weights:
            if self.use_best_weights:
                model_weights_file = f"best_{self.output_weights_name}"
            else:
                model_weights_file = f"{self.output_weights_name}"
            trained_model = os.path.join(self.output_dir, model_weights_file)
        return trained_model

    @property
    def base_model_weights_file(self):
        if self.use_base_model_weights:
            if self.use_ext_base_model_weights:
                return assign_raise(self.cp[self.SECTION].get("base_model_weights_file"))
            else:
                return "imagenet"
        else:
            return None

    @property
    def use_best_weights(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("use_best_weights"), self._use_trained_model_weights)

    @property
    def output_weights_name(self):
        return assign_raise(self.cp[self.SECTION].get("output_weights_name"))

    @property
    def show_model_summary(self):
        return assign_fallback(self.cp[self.SECTION].getboolean("show_model_summary"), self._show_model_summary)
