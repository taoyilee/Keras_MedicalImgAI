import argparse
import csv
import fnmatch
import os

import cv2 as cv
import numpy as np

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
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Image Index', 'Finding Labels', 'Patient ID', 'OriginalImage Width', 'OriginalImage Height'])

        for images in os.listdir(path=v):
            image_path = os.path.join(v, images)
            print(f"Processing {image_path} {os.path.isfile(image_path)}")
            img = cv.imread(image_path)
            _, fn = os.path.split(images)
            fn_root, _ = os.path.splitext(fn)
            w = int(img.shape[0])
            h = int(img.shape[1])
            landscape = w > h
            portrait = h > w
            starting_index = int(abs(w - h) / 2)
            if landscape:
                img = img[starting_index:starting_index + h, :]
            if portrait:
                img = img[:, starting_index:starting_index + w]
            img = cv.resize(img, (output_size, output_size))
            image_out_base = f"{prefix_str[i]}_{fn_root}.png"
            image_out_path = os.path.join(output_dir, image_out_base)
            print(f"Writing resized to {image_out_path} {img.shape[0]}X{img.shape[1]}")
            cv.imwrite(image_out_path, img)
            spamwriter.writerow([image_out_base, diag_str[i], patient_ID, img.shape[0], img.shape[1]])
            patient_ID = patient_ID + 1
