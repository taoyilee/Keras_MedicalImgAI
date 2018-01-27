import numpy as np
import os
from callback import load_generator_data
from configparser import ConfigParser
from generator import custom_image_generator
from models.densenet121 import get_model
from utility import get_sample_counts
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import cv2
import csv
import sys
import grad_cam as gc


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    class_names = cp["DEFAULT"].get("class_names").split(",")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "test", class_names)

    symlink_dir_name = "image_links"
    #test_dir_name = sys.argv[1]
    test_data_path = f"{output_dir}/{symlink_dir_name}/test/"
    # test_data_path = f"{test_dir_name}/"

    step_test = int(test_counts / batch_size)
    print("** load test generator **")
    print(f"** loading from {test_data_path} **")
    test_generator = custom_image_generator(
        ImageDataGenerator(horizontal_flip=True, rescale=1./255),
        test_data_path,
        batch_size=batch_size,
        class_names=class_names,
    )
    x, y, x_orig = load_generator_data(test_generator, step_test, len(class_names))
    xshape = np.shape(x)
    print(f"x = {xshape}")
    #x0 = x[0:1,:,:,:]
    #x0_orig = x_orig[0:1,:,:,:]
    #x0shape = np.shape(x0)
    #print(f"x0 = {x0shape}")

    print("** load model **")
    model = get_model(class_names)
    if use_best_weights:
        print("** use best weights **")
        model.load_weights(best_weights_path)
    else:
        print("** use last weights **")
        model.load_weights(weights_path)

    print("** make prediction **")
    y_hat = model.predict(x)

    print("** perform grad cam **")
    print("y_hat = {}".format(y_hat))
    print("shape y_hat = {}".format(np.shape(y_hat)))
    fx_prob = y_hat[0][:, 0]
    nm_prob = y_hat[1][:, 0]
    print("fx_prob = {}".format(np.shape(fx_prob)))
    with open('predicted_class.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['ID', 'FX Probability', 'Normal Probability'])
        for i in range(len(fx_prob)):
            csvwriter.writerow([i, fx_prob[i], nm_prob[i]])
    #print(predicted_class)
    #for l in model.layers:
    #    print("layer name = {} {}".format(l.name, l.__class__.__name__))
            xi_orig = 255*x_orig[i, :]
            #print("image = {}".format(x0_orig))
            #print("image dimension = {}".format(np.shape(xi_orig)))
            #print("image pixel class = {}".format(x0_orig[0,0,0].__class__.__name__))
            cv2.imwrite(f"imgdir/orig_image_{i}.jpg", np.uint8(xi_orig))
            predicted_class = 0 if fx_prob[i] > nm_prob[i] else 1
            cam, heatmap = gc.grad_cam(model, x[np.newaxis, i, :], predicted_class, "conv5_blk_scale")
            cv2.imwrite(f"imgdir/gradcam.jpg_{i}", cam)
            cv2.imwrite(f"imgdir/concat_gradcam_{i}.jpg", np.concatenate((np.uint8(xi_orig), cam), axis=1))
    #print(f"yhat = {y_hat}")

if __name__ == "__main__":
    main()
