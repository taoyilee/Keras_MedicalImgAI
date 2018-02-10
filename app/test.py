import argparse
import csv
import os
from configparser import ConfigParser

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from app import grad_cam as gc
from app.datasets import dataset_loader as dsload
from app.models.densenet121 import get_model


def main(config_file):
    # parser config
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_dimension = cp["DEFAULT"].getint("image_dimension")
    model_name = cp["DEFAULT"].get("nn_model")
    class_mode = cp["DEFAULT"].get("class_mode")

    verbosity = cp["DEFAULT"].getint("verbosity")
    progress_verbosity = cp["TEST"].getint("progress_verbosity")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")
    enable_grad_cam = cp["TEST"].getboolean("enable_grad_cam")
    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # get test sample count
    data_entry_file = f"{output_dir}/test.csv"
    print(f"**Reading test set from {data_entry_file}")
    dataset0 = dsload.DataSetTest(image_dir=image_source_dir, data_entry=data_entry_file, batch_size=batch_size,
                                  img_dim=256, class_names=class_names, class_mode=class_mode)

    print("** load test generator **")
    test_generator = dataset0.test_generator(verbosity=verbosity)
    step_test = test_generator.__len__()
    print("** load model **")
    model = get_model(class_names, image_dimension=image_dimension, class_mode=class_mode)
    if use_best_weights:
        print("** use best weights **")
        model.load_weights(best_weights_path)
    else:
        print("** use last weights **")
        model.load_weights(weights_path)

    print("** make predictions **")
    y = np.array(test_generator.targets()).squeeze()
    y_hat = np.array(model.predict_generator(generator=test_generator, steps=step_test, verbose=progress_verbosity))
    test_log_path = os.path.join(output_dir, "test.log")
    print(f"** write log to {test_log_path} **")
    aurocs = []
    print(f"y = {np.shape(y)}")
    print(f"y_hat = {np.shape(y_hat)}")

    with open(test_log_path, "w") as f:
        if class_mode == "multibinary":
            y = y.squeeze().swapaxes(0, 1)
        aurocs = roc_auc_score(y, y_hat, average=None)

        if len(class_names) != len(aurocs):
            raise Exception(f"Wrong shape in either y or y_hat {len(class_names)} != {len(current_auroc)}")

        for i, v in enumerate(class_names):
            print(f" {i+1}. {v} AUC = {np.around(aurocs[i], 2)}")
            f.write(f"{class_names[i]}: {aurocs[i]}\n")

        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write(f"mean AUC: {mean_auroc}\n")
        print("-------------------------")
        print(f"mean AUC: {mean_auroc}")

    if enable_grad_cam:
        print("** perform grad cam **")
        os.makedirs("imgdir", exist_ok=True)
        with open('predicted_class.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_header = ['ID', 'Most probable diagnosis']
            for i, v in enumerate(class_names):
                csv_header.append(f"{v}_Prob")
            csvwriter.writerow(csv_header)
            for i, v in enumerate(y_hat):
                predicted_class = np.argmax(v)
                labeled_class = np.argmax(y[i])
                print(
                    f"** y_hat[{i}] = {v.round(3)} Label/Prediction: {class_names[labeled_class]}/{class_names[predicted_class]}")
                csv_row = [str(i + 1), f"{class_names[predicted_class]}"] + [str(vi.round(3)) for vi in v]
                csvwriter.writerow(csv_row)
                x_orig = test_generator.orig_input(i).squeeze()
                x_orig = cv2.cvtColor(x_orig, cv2.COLOR_GRAY2RGB)
                x = test_generator.model_input(i)
                cam = gc.grad_cam(model, x, x_orig, predicted_class, "conv5_blk_scale", class_names)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(x_orig, f"Labeled as:{class_names[labeled_class]}", (5, 20), font, 1,
                            (255, 255, 255),
                            2, cv2.LINE_AA)

                cv2.putText(cam, f"Predicted as:{class_names[predicted_class]}", (5, 20), font, 1,
                            (255, 255, 255),
                            2, cv2.LINE_AA)

                print(f"Writing cam file to imgdir/gradcam_{i}.jpg")

                cv2.imwrite(f"imgdir/gradcam_{i}.jpg", np.concatenate((x_orig, cam), axis=1))


if __name__ == "__main__":
    # use argparse to accept command line variables (config.ini)
    parser = argparse.ArgumentParser(description='Train a specfic dataset')
    parser.add_argument('config', metavar='config', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    main(config_file=args.config)