import configparser
import unittest

from app.datasets.DatasetConfig import DatasetConfig
from app.imagetoolbox.ImageConfig import ImageConfig


class TestDatasetConfig(unittest.TestCase):
    def setUp(self):
        self.cp = configparser.ConfigParser()
        self.cp['IMAGE'] = {"image_dir": "image/", "img_dim": 256, "scale": 1. / 255, "class_mode": "multiclass",
                            "use_class_balancing": True, "color_mode": "grayscale"}

        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": "",
                              "random_state": 0, "train_ratio": 70, "dev_ratio": 10, "batch_size": 32,
                              "class_mode": "multiclass", "use_class_balancing": True, "positive_weights_multiply": 1,
                              "force_resplit": False}

    def test_ImageConfig(self):
        dc = DatasetConfig(self.cp)
        if "ImageConfig" not in dir(dc):
            self.fail("ImageConfig property method does not exist")

    def test_ImageConfigEqual(self):
        ic = ImageConfig(self.cp)
        dc = DatasetConfig(self.cp)
        self.assertEqual(dc.ImageConfig, ic)


if __name__ == '__main__':
    unittest.main()
