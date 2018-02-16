#!/bin/bash
suffix=\..*
prefix=".*\/"
find images_1024/ -type f -name "*.png" | sed -e "s/^$prefix//" | xargs -n 1 -P 8 -I {} ./magick_wrapper.sh {}
