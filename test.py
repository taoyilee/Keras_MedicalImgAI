import argparse
import importlib
import os
from configparser import ConfigParser

import numpy as np
from sklearn.metrics import roc_auc_score

from models.densenet121 import get_model


def main(config_file):
    # parser config
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    train_patient_ratio = cp["DEFAULT"].getint("train_patient_ratio")
    dev_patient_ratio = cp["DEFAULT"].getint("dev_patient_ratio")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_dimension = cp["DEFAULT"].getint("image_dimension")
    model_name = cp["DEFAULT"].get("nn_model")
    dataset_name = cp["DEFAULT"].get("dataset_name")

    dataset_spec = importlib.util.spec_from_file_location(dataset_name, f"./datasets/{dataset_name}.py")
    if dataset_spec is None:
        print(f"can't find the {dataset_name} module")
    else:
        # If you chose to perform the actual import ...
        dataset_pkg = importlib.util.module_from_spec(dataset_spec)
        dataset_spec.loader.exec_module(dataset_pkg)

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # get test sample count
    dataset0 = dataset_pkg.DataSet(image_dir=image_source_dir, data_entry=data_entry_file,
                                   train_ratio=train_patient_ratio,
                                   dev_ratio=dev_patient_ratio,
                                   output_dir=output_dir, img_dim=256, class_names=class_names,
                                   random_state=split_dataset_random_state)

    step_test = int(dataset0.test_count / batch_size)
    print("** load test generator **")
    test_generator = dataset0.test_generator(verbosity=verbosity)

    print("** load model **")
    model = get_model(class_names, image_dimension=image_dimension)
    if use_best_weights:
        print("** use best weights **")
        model.load_weights(best_weights_path)
    else:
        print("** use last weights **")
        model.load_weights(weights_path)

    print("** make prediction **")
    y_hat = model.predict_generator(generator=test_generator, steps=step_test, verbose=1)

    test_log_path = os.path.join(output_dir, "test.log")
    print(f"** write log to {test_log_path} **")
    aurocs = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[i], y_hat[i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write(f"{class_names[i]}: {score}\n")
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}\n")


if __name__ == "__main__":
    # use argparse to accept command line variables (config.ini)
    parser = argparse.ArgumentParser(description='Train a specfic dataset')
    parser.add_argument('config', metavar='config', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    main(config_file=args.config)
