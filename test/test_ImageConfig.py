import configparser
import unittest

from app.imagetoolbox.ImageConfig import ImageConfig


class TestImageConfig(unittest.TestCase):
    def setUp(self):
        self.cp = configparser.ConfigParser()
        self.cp['IMAGE'] = {"image_dir": "image/", "img_dim": 256, "scale": 1. / 255, "class_mode": "multiclass",
                            "use_class_balancing": True, "color_mode": "grayscale"}

        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": "",
                              "random_state": 0, "train_ratio": 70, "dev_ratio": 10, "batch_size": 32,
                              "class_mode": "multiclass", "use_class_balancing": True, "positive_weights_multiply": 1,
                              "force_resplit": False}

    def test_img_dim_default(self):
        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": ""}
        self.cp['IMAGE'] = {"image_dir": "image/"}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.img_dim, 256)

    def test_scale_default(self):
        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": ""}
        self.cp['IMAGE'] = {"image_dir": "image/"}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.scale, 1. / 255)

    def test_color_mode_default(self):
        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": ""}
        self.cp['IMAGE'] = {"image_dir": "image/"}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.color_mode, "grayscale")

    def test_img_dim_setvalue(self):
        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": ""}
        self.cp['IMAGE'] = {"image_dir": "image/", "img_dim": 1024}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.img_dim, 1024)
        self.cp['IMAGE'] = {"image_dir": "image/", "img_dim": 512}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.img_dim, 512)
        self.cp['IMAGE'] = {"image_dir": "image/", "img_dim": 256}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.img_dim, 256)

    def test_scale_setvalue(self):
        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": ""}
        self.cp['IMAGE'] = {"image_dir": "image/", "scale": 1.0}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.scale, 1.0)
        self.cp['IMAGE'] = {"image_dir": "image/", "scale": 0.5}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.scale, 0.5)
        self.cp['IMAGE'] = {"image_dir": "image/", "scale": 0.1}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.scale, 0.1)

    def test_color_mode_setvalue(self):
        self.cp['DATASET'] = {"data_entry": "image/data_entry.csv", "class_names": "", "output_dir": ""}
        self.cp['IMAGE'] = {"image_dir": "image/", "color_mode": "rgb"}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.color_mode, "rgb")
        self.cp['IMAGE'] = {"image_dir": "image/", "color_mode": "grayscale"}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.color_mode, "grayscale")
        self.cp['IMAGE'] = {"image_dir": "image/", "color_mode": "hsv"}
        ic = ImageConfig(self.cp)
        self.assertEqual(ic.color_mode, "hsv")

    def test_img_dim_type(self):
        ic = ImageConfig(self.cp)
        self.assertIsInstance(ic.img_dim, int)

    def test_scale_type(self):
        ic = ImageConfig(self.cp)
        self.assertIsInstance(ic.scale, float)

    def test_color_mode_type(self):
        ic = ImageConfig(self.cp)
        self.assertIsInstance(ic.color_mode, str)


if __name__ == '__main__':
    unittest.main()
