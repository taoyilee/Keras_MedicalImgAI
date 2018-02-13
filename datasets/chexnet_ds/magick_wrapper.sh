#!/bin/bash
if [ ! -f images/$1 ]; then
	echo "Resizing images_1024/$1 to images/$1"
	magick images_1024/$1 -resize '224x224!' images/$1
else
	echo "Skip images/$1"
fi
