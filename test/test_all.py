import unittest

from test.test_DatasetConfig import TestDatasetConfig
from test.test_ImageConfig import TestImageConfig


def create_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestDatasetConfig())
    test_suite.addTest(TestImageConfig())
    return test_suite


if __name__ == '__main__':
    suite = create_suite()

    runner = unittest.TextTestRunner()
    runner.run(suite)
