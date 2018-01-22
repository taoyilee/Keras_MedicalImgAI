#!/bin/bash

for a in `find $1 -name "*.png" -type f`;
do 
	prefix_dir="$(cut -d'/' -f1 <<< $a)"
	filename="$(cut -d'/' -f2 <<< $a)"	
	rename1=$prefix_dir/$filename
	rename2=$prefix_dir/$2_$filename
	echo "Renaming $prefix_dir/$filename to "$prefix_dir/$2_$filename
        mv $rename1 $rename2
done
