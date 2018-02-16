from unittest import TestCase

from app.datasets import NormalizeConfig


class TestNormalizeConfig(TestCase):
    def test_nonexistent_config(self):
        with self.assertRaises(FileNotFoundError):
            self.conf = NormalizeConfig("nonexistent.ini")
