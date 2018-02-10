import argparse
import json
import os
import pickle
import shutil
from configparser import ConfigParser

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from callback import MultipleClassAUROC, MultiGPUModelCheckpoint, SaveBaseModel
from app.datasets import dataset_loader as dsload
from app.models.densenet121 import get_model


def main(config_file):
    # parser config
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    model_name = cp["DEFAULT"].get("nn_model")
    class_mode = cp["DEFAULT"].get("class_mode")
    train_patient_ratio = cp["DEFAULT"].getint("train_patient_ratio")
    dev_patient_ratio = cp["DEFAULT"].getint("dev_patient_ratio")
    data_entry_file = cp["DEFAULT"].get("data_entry_file")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_dimension = cp["DEFAULT"].getint("image_dimension")
    verbosity = cp["DEFAULT"].getint("verbosity")
    progress_verbosity = cp["TRAIN"].getint("progress_verbosity")

    color_mode = cp["DEFAULT"].get("color_mode")

    # train config
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    train_steps = cp["TRAIN"].get("train_steps")
    patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
    validation_steps = cp["TRAIN"].get("validation_steps")
    positive_weights_multiply = cp["TRAIN"].getfloat("positive_weights_multiply")
    use_class_balancing = cp["TRAIN"].getboolean("use_class_balancing")
    use_default_split = cp["TRAIN"].getboolean("use_default_split")
    force_resplit = cp["TRAIN"].getboolean("force_resplit")

    # if previously trained weights is used, never re-split
    training_stats = {}
    if not force_resplit and use_trained_model_weights:
        # resuming mode
        print("** attempting to use trained model weights **")
        # load training status for resuming
        training_stats_file = os.path.join(output_dir, ".training_stats.json")
        if os.path.isfile(training_stats_file):
            training_stats = json.load(open(training_stats_file))
            initial_learning_rate = training_stats["lr"]
            print(f"** learning rate is set to previous final {initial_learning_rate} **")
        else:
            print("** trained model weights not found, starting over **")
            use_trained_model_weights = False

    split_dataset_random_state = cp["TRAIN"].getint("split_dataset_random_state")
    show_model_summary = cp["TRAIN"].getboolean("show_model_summary")
    # end parser config

    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    running_flag_file = os.path.join(output_dir, ".training.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()
    try:
        print(f"backup config file to {output_dir}")
        shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

        # split train/dev/test
        if use_default_split:
            datasets = ["train", "dev", "test"]
            for d in datasets:
                shutil.copy(f"./data/default_split/{d}.csv", output_dir)

        data_set = dsload.DataSet(image_dir=image_source_dir, data_entry=data_entry_file,
                                  train_ratio=train_patient_ratio,
                                  dev_ratio=dev_patient_ratio,
                                  output_dir=output_dir, img_dim=256, class_names=class_names,
                                  random_state=split_dataset_random_state, class_mode=class_mode,
                                  use_class_balancing=use_class_balancing,
                                  positive_weights_multiply=positive_weights_multiply, force_resplit=force_resplit)
        print("** create image generators **")
        train_generator = data_set.train_generator(verbosity=verbosity)
        dev_generator = data_set.dev_generator(verbosity=verbosity)

        # compute steps
        if train_steps == "auto":
            train_steps = train_generator.__len__()
        else:
            try:
                train_steps = int(train_steps)
            except ValueError:
                raise ValueError(f"""
                train_steps: {train_steps} is invalid,
                please use 'auto' or integer.
                """)
        print(f"** train_steps: {train_steps} **")

        if validation_steps == "auto":
            validation_steps = dev_generator.__len__()
        else:
            try:
                validation_steps = int(validation_steps)
            except ValueError:
                raise ValueError(f"""
                validation_steps: {validation_steps} is invalid,
                please use 'auto' or integer.
                """)
        print(f"** validation_steps: {validation_steps} **")

        # compute class weights
        print("** compute class weights from training data **")
        class_weights = data_set.class_weights()

        print("** load model **")
        if use_base_model_weights:
            base_model_weights_file = cp["TRAIN"].get("base_model_weights_file")
            print(f"** loading base model weight from {base_model_weights_file} **")
        else:
            base_model_weights_file = None
        if use_trained_model_weights:
            if use_best_weights:
                model_weights_file = os.path.join(output_dir, f"best_{output_weights_name}")
                print(f"** loading best model weight from {model_weights_file} **")
            else:
                model_weights_file = os.path.join(output_dir, output_weights_name)
                print(f"** loading final model weight from {model_weights_file} **")
        else:
            model_weights_file = None
        model = get_model(class_names, base_model_weights_file, model_weights_file, image_dimension=image_dimension,
                          color_mode=color_mode, class_mode=class_mode)
        if show_model_summary:
            print(model.summary())

        output_weights_path = os.path.join(output_dir, output_weights_name)
        print(f"** set output weights path to: {output_weights_path} **")

        print("** check multiple gpu availability **")
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            model_train = multi_gpu_model(model, gpus)
            # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
            checkpoint = MultiGPUModelCheckpoint(
                filepath=output_weights_path,
                base_model=model,
            )
        else:
            model_train = model
            checkpoint = ModelCheckpoint(output_weights_path)

        print("** compile model with class weights **")
        optimizer = Adam(lr=initial_learning_rate)
        model_train.compile(optimizer=optimizer, loss="binary_crossentropy")
        auroc = MultipleClassAUROC(generator=dev_generator, steps=validation_steps, class_names=class_names,
                                   class_mode=class_mode, weights_path=output_weights_path, stats=training_stats)
        callbacks = [
            checkpoint,
            TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr, verbose=1),
            auroc,
            SaveBaseModel(filepath=cp["TRAIN"].get("base_model_weights_file"), save_weights_only=False)
        ]

        print("** training start **")
        print(f"** training with: {epochs} epochs @ {train_steps} steps/epoch **")

        history = model_train.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            verbose=progress_verbosity,
            validation_data=dev_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            max_queue_size=4, workers=4, use_multiprocessing=True
        )

        # dump history
        print("** dump history **")
        with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
            pickle.dump({
                "history": history.history,
                "auroc": auroc.aurocs,
            }, f)
        print("** done! **")

    finally:
        os.remove(running_flag_file)


if __name__ == "__main__":
    '''
    Entry Point
    '''

    # use argparse to accept command line variables (config.ini)
    parser = argparse.ArgumentParser(description='Train a specfic dataset')
    parser.add_argument('config', metavar='config', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    main(config_file=args.config)