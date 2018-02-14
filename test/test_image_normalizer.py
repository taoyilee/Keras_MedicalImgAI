import configparser

import cv2
import numpy as np

from app.datasets.NormalizeConfig import NormalizeConfig
from app.datasets.image_normalizer import ImageNormalizer

cp = configparser.ConfigParser()
cp.read("chexnet_config.ini")
nrm_conf = NormalizeConfig(cp)
nm0 = ImageNormalizer(nrm_conf)
img = cv2.imread("datasets/chexnet_ds/images/00000001_000.png")
print(f"mean/std = {np.mean(img)}{np.std(img)}")
img = img / 255
print(f"mean/std = {np.mean(img)}/{np.std(img)}")
img = nm0.normalize(img)
print(f"mean/std = {np.mean(img)}/{np.std(img)}")
