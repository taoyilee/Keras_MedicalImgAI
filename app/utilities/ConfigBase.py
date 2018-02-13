import configparser

from app.utilities.util_config import returnPropertIfNotNull, assignIfNotNull


class ConfigBase:
    _batch_size = 32

    def __init__(self, cp):
        """

        :param cp: Config Parser
        :type cp: configparser.ConfigParser
        """
        self.cp = cp

    @property
    def output_dir(self):
        return returnPropertIfNotNull(self.cp["DEFAULT"].get("output_dir"))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    @property
    def class_mode(self):
        return returnPropertIfNotNull(self.cp["DATASET"].get("class_mode"))

    @property
    def batch_size(self):
        return assignIfNotNull(self.cp["TRAIN"].getint("batch_size"), self._batch_size)
