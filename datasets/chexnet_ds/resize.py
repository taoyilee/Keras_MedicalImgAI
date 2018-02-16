import argparse
import csv
import fnmatch
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np

from resize_img import resize_img

parser = argparse.ArgumentParser(description='Generate data entry from image dir')
parser.add_argument('input_dir', metavar='output_dir', type=str, help='the output directory')
parser.add_argument('output_dir', metavar='output_dir', type=str, help='the input directory')
parser.add_argument('output_size', metavar='output_size', type=int, help='output_size')

args = parser.parse_args()
input_dir = str(args.input_dir)
output_dir = str(args.output_dir)
output_size = args.output_size
print(f"Resizing all images to {output_size}")

os.makedirs(output_dir, exist_ok=True)
with ThreadPoolExecutor(max_workers=64) as executor:
    for images in os.listdir(path=input_dir):
        image_path = os.path.join(input_dir, images)
        args = (image_path, output_dir, output_size)
        executor.submit(resize_img, *args)
