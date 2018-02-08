import argparse
import os
from configparser import ConfigParser

import numpy as np
from sklearn.metrics import roc_auc_score

from datasets import dataset_loader as dsload
from models.densenet121 import get_model


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


if __name__ == "__main__":
    # use argparse to accept command line variables (config.ini)
    parser = argparse.ArgumentParser(description='Train a specfic dataset')
    parser.add_argument('config', metavar='config', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    main(config_file=args.config)
