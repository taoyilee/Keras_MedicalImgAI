import os
import random

import cv2
import numpy as np
import pandas as pd


def load_data(image_dir, data_entry, class_names, output_dir, random_state=0,
              train_ratio=70, dev_ratio=10, batch_size=16, img_dim=256, scale=1. / 255):
    """Loads Chexnet dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    os.makedirs( output_dir, exist_ok=True )
    e = pd.read_csv( data_entry )

    # one hot encode
    e["One_Hot_Labels"] = e["Finding Labels"].apply( lambda x: label2vec( x, class_names ) )

    # shuffle and split
    pid = list( e["Patient ID"].unique() )
    total_patients = len( pid )
    train_patient_count = int( total_patients * train_ratio / 100 )
    dev_patient_count = int( total_patients * dev_ratio / 100 )
    test_patient_count = total_patients - train_patient_count - dev_patient_count

    random.seed( random_state )
    random.shuffle( pid )
    train = e[e["Patient ID"].isin( pid[:train_patient_count] )]
    dev = e[e["Patient ID"].isin( pid[train_patient_count:train_patient_count + dev_patient_count] )]
    test = e[e["Patient ID"].isin( pid[train_patient_count + dev_patient_count:] )]
    total_images = len( e )
    train_count = len( train )
    dev_count = len( dev )
    test_count = len( test )
    print(
        f"Total patients = {total_patients} in train/dev/test {train_patient_count}/{dev_patient_count}/{test_patient_count}" )
    print( f"Total images = {total_images} in train/dev/test {train_count}/{dev_count}/{test_count}" )

    # export csv
    output_fields = ["Image Index", "Patient ID", "Finding Labels", "One_Hot_Labels"]
    train[output_fields].to_csv( os.path.join( output_dir, "train.csv" ), index=False )
    dev[output_fields].to_csv( os.path.join( output_dir, "dev.csv" ), index=False )
    test[output_fields].to_csv( os.path.join( output_dir, "test.csv" ), index=False )
    i_train = 0
    i_dev = 0
    i_test = 0
    while True:
        train_gen = batch_generator( train["Image Index"].iloc[i_train:i_train + batch_size],
                                     train["One_Hot_Labels"].tolist()[i_train: i_train + batch_size], image_dir,
                                     img_dim=img_dim, scale=scale )
        dev_gen = batch_generator( dev["Image Index"].iloc[i_train:i_train + batch_size],
                                   dev["One_Hot_Labels"].tolist()[i_train:i_train + batch_size], image_dir,
                                   img_dim=img_dim, scale=scale )
        test_gen = batch_generator( test["Image Index"].iloc[i_train:i_train + batch_size],
                                    test["One_Hot_Labels"].tolist()[i_train:i_train + batch_size], image_dir,
                                    img_dim=img_dim, scale=scale )
        yield train_gen, dev_gen, test_gen
        i_train += batch_size
        i_train %= train_count
        i_dev += batch_size
        i_dev %= dev_count
        i_test += batch_size
        i_test %= test_count


def batch_generator(image_filenames, labels, image_dir, img_dim=256, scale=1. / 255):
    return {'data': np.array(
        image_filenames.apply( lambda x: load_image( x, image_dir, img_dim=img_dim, scale=scale ) ).tolist() ),
        'label': np.array( labels )}


def load_image(image_name, image_dir, img_dim=256, scale=1. / 255):
    image_file = image_dir + "/" + image_name;
    if not os.path.isfile( image_file ):
        raise Exception( f"{image_file} not found" )
    image = cv2.imread( image_file, 0 )[:, :, np.newaxis]
    image = cv2.resize( image, (img_dim, img_dim) )
    return image * scale


def label2vec(label, class_names):
    vec = np.zeros( len( class_names ) )
    if label == "No Finding":
        return vec
    labels = label.split( "|" )
    for l in labels:
        vec[class_names.index( l )] = 1
    return vec
