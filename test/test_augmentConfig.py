from unittest import TestCase

from app.datasets import AugmentConfig


class TestAugmentConfig(TestCase):
    def setUp(self):
        self.conf = AugmentConfig("augment_config_flip_prob.ini")
        self.conf_ud = AugmentConfig("augment_config_ud.ini")
        self.conf_lr = AugmentConfig("augment_config_lr.ini")
        self.conf_empty = AugmentConfig("empty.ini")

    def test_nonexistent_config(self):
        with self.assertRaises(FileNotFoundError):
            self.conf = AugmentConfig("nonexistent.ini")

    def test_has_flip_prob(self):
        self.assertTrue(hasattr(self.conf, "flip_prob"))

    def test_has_flip_lr(self):
        self.assertTrue(hasattr(self.conf, "random_horz_flip"))

    def test_has_flip_ud(self):
        self.assertTrue(hasattr(self.conf, "random_vert_flip"))

    def test_flip_lr_def(self):
        self.assertFalse(self.conf_empty.random_horz_flip)

    def test_flip_prob_def(self):
        self.assertEqual(self.conf_empty.flip_prob, 0.5)

    def test_flip_prob_set(self):
        self.assertEqual(self.conf.flip_prob, 0.4)

    def test_flip_ud_def(self):
        self.assertFalse(self.conf_empty.random_vert_flip)

    def test_flip_lr_set(self):
        self.assertTrue(self.conf.random_horz_flip)
        self.assertTrue(self.conf_lr.random_horz_flip)

    def test_flip_ud_set(self):
        self.assertTrue(self.conf.random_vert_flip)
        self.assertTrue(self.conf_ud.random_vert_flip)

