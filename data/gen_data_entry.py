import os 
import csv
import fnmatch
import argparse
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(description='Generate data entry from image dir')
parser.add_argument('imgdir', metavar='imgdir', type=str, help='the image directory')

args = parser.parse_args()
imgdir = str(args.imgdir)

patient_ID = 1

with open('image_data_entry.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Image Index', 'Finding Labels', 'Patient ID', 'OriginalImage Width', 'OriginalImage Height'])

    for images in  os.listdir(path=imgdir):
        img = cv.imread(imgdir + images )
        if fnmatch.fnmatch(images, 'ITFX_*.jpg'):
            print ('Intertrochanter fracture ' + images)
            spamwriter.writerow([images, 'Intertrochanteric_FX', patient_ID, img.shape[0], img.shape[1]])
        if fnmatch.fnmatch(images, 'FX_*.jpg'):
            print ('Femoral neck fracture ' + images)
            spamwriter.writerow([images, 'Femoral_Neck_FX', patient_ID, img.shape[0], img.shape[1]])
        if fnmatch.fnmatch(images, 'NM_*.jpg'):
            print ('Normal ' + images)
            spamwriter.writerow([images, 'Normal', patient_ID, img.shape[0], img.shape[1]])
        patient_ID = patient_ID + 1
