import configparser
import os

from app.utilities.util_config import assign_raise, assign_fallback


class ConfigBase:
    _batch_size = 32
    _force_resplit = False
    cp: configparser.ConfigParser = None

    def __init__(self, cp: configparser.ConfigParser):
        """

        :param cp: File name of config.ini or a ConfigParser
        """

        if isinstance(cp, str):
            if not os.path.isfile(cp):
                raise FileNotFoundError(f"Config file {cp} is not found")
            print(f"** Reading config file {cp}")
            self.cp = configparser.ConfigParser()
            self.cp.read(cp)
        else:
            self.cp = cp

    @property
    def output_dir(self):
        return assign_raise(self.cp["DEFAULT"].get("output_dir"))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    @property
    def class_mode(self):
        return assign_raise(self.cp["DATASET"].get("class_mode"))

    @property
    def batch_size(self):
        return assign_fallback(self.cp["TRAIN"].getint("batch_size"), self._batch_size)

    @property
    def train_stats_file(self):
        return os.path.join(self.output_dir, ".training_stats.json")

    @property
    def train_lock_file(self):
        return os.path.join(self.output_dir, ".training.lock")

    @property
    def test_lock_file(self):
        return os.path.join(self.output_dir, ".testing.lock")

    @property
    def force_resplit(self):
        return assign_fallback(self.cp["DATASET"].getboolean("force_resplit"), self._force_resplit)

    @property
    def dataset_dilation(self):
        return assign_raise(self.cp["DATASET"].getfloat("dataset_dilation"))
