import argparse
import csv
import fnmatch
import os
import threading

import cv2 as cv
import numpy as np

from .resize_img import resize_img

parser = argparse.ArgumentParser(description='Generate data entry from image dir')
parser.add_argument('input_dir', metavar='output_dir', type=str, help='the output directory')
parser.add_argument('output_dir', metavar='output_dir', type=str, help='the input directory')
parser.add_argument('output_size', metavar='output_size', type=int, help='output_size')

args = parser.parse_args()
output_dir = str(args.output_dir)
output_size = args.output_size
prefix_str = ['FNFX', 'ITFX', 'NM']
patient_ID = 1
print(f"Resizing all images to {output_size}")

os.makedirs(output_dir, exist_ok=True)
with open('image_data_entry.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['Image Index', 'Finding Labels', 'Patient ID', 'OriginalImage Width', 'OriginalImage Height'])

    for images in os.listdir(path=v):
        image_path = os.path.join(v, images)
        threading.Thread(target=resize_img, args=(csvwriter, image_path, output_dir, output_size)).start()
        patient_ID = patient_ID + 1
