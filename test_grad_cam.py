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
    image_dimension = cp["DEFAULT"].getint("image_dimension")

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
        target_size=(image_dimension, image_dimension),
        cam=True
    )
    x, y, x_orig = load_generator_data(test_generator, step_test, len(class_names), cam=True)
    xshape = np.shape(x)
    print(f"x = {xshape}")
    #x0 = x[0:1,:,:,:]
    #x0_orig = x_orig[0:1,:,:,:]
    #x0shape = np.shape(x0)
    #print(f"x0 = {x0shape}")

    print("** load model **")
    model = get_model(class_names, image_dimension=image_dimension)
    if use_best_weights:
        print("** use best weights **")
        model.load_weights(best_weights_path)
    else:
        print("** use last weights **")
        model.load_weights(weights_path)

    print("** make prediction **")
    y_hat = np.array(model.predict(x)).squeeze()
    y_hat = y_hat.swapaxes(0,1)
    print("** perform grad cam **")
    with open('predicted_class.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_header = ['ID', 'Most probable diagnosis']
        for i, v in enumerate(class_names):
            csv_header.append(f"{v}_Prob")
        csvwriter.writerow(csv_header)
        for i, v in enumerate(y_hat):
            predicted_class = np.argmax(v)
            print(f"** y_hat[{i+1}] = {v.round(3)} Prediction: {class_names[predicted_class]}")
            csv_row = [str(i+1), f"{class_names[predicted_class]}"] + [str(vi.round(3)) for vi in v] 
            csvwriter.writerow(csv_row)
            xi_orig = 255*x_orig[i, :]
            cv2.imwrite(f"imgdir/orig_image_{i}.jpg", np.uint8(xi_orig))
            cam, heatmap = gc.grad_cam(model, x[np.newaxis, i, :], predicted_class, "conv5_blk_scale", image_dimension=image_dimension)
            font = cv2.FONT_HERSHEY_SIMPLEX
            gradcam_img = 0.4*cam + 0.6*np.uint8(xi_orig)
            cv2.putText(gradcam_img, f"Predicted as:{class_names[predicted_class]}", (5, 20), font, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(f"imgdir/gradcam_{i}.jpg", gradcam_img)

if __name__ == "__main__":
    main()
