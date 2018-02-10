import json
import os
import pickle
import shutil
from configparser import ConfigParser

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from app.callback import MultipleClassAUROC, MultiGPUModelCheckpoint, SaveBaseModel
from app.datasets import dataset_loader as dsload
from app.datasets.DatasetConfig import DatasetConfig
from app.models.densenet121 import get_model


class Trainer:
    output_dir = ""
    image_source_dir = ""
    model_name = "densenet121"
    class_mode = "multiclass"
    train_patient_ratio = 70
    dev_patient_ratio = 20
    data_entry_file = ""
    class_names = ""
    image_dimension = 256
    verbosity = 0
    progress_verbosity = 1
    color_mode = "grayscale"
    # model config
    base_model_weights_file = None
    # train config
    use_base_model_weights = True
    use_trained_model_weights = True
    use_best_weights = False
    output_weights_name = ""
    epochs = 20
    batch_size = 32
    initial_learning_rate = 0.001
    train_steps = "auto"
    patience_reduce_lr = 1
    validation_steps = "auto"
    positive_weights_multiply = 1
    use_class_balancing = True
    use_default_split = False
    force_resplit = False
    training_stats = {}
    split_dataset_random_state = 0
    show_model_summary = False
    running_flag_file = ""
    DSConfig = None

    # Runtime stuffs
    history = None
    auroc = None
    model = None
    model_train = None
    checkpoint = None

    def __init__(self, config_file):
        self.cp = ConfigParser()
        self.cp.read(config_file)
        self.parse_config()

    def parse_config(self):
        self.output_dir = self.cp["DEFAULT"].get("output_dir")
        self.image_source_dir = self.cp["DEFAULT"].get("image_source_dir")
        self.model_name = self.cp["DEFAULT"].get("nn_model")
        self.class_mode = self.cp["DEFAULT"].get("class_mode")
        self.train_patient_ratio = self.cp["DEFAULT"].getint("train_patient_ratio")
        self.dev_patient_ratio = self.cp["DEFAULT"].getint("dev_patient_ratio")
        self.data_entry_file = self.cp["DEFAULT"].get("data_entry_file")
        self.class_names = self.cp["DEFAULT"].get("class_names").split(",")
        self.image_dimension = self.cp["DEFAULT"].getint("image_dimension")
        self.verbosity = self.cp["DEFAULT"].getint("verbosity")
        self.progress_verbosity = self.cp["TRAIN"].getint("progress_verbosity")
        self.color_mode = self.cp["DEFAULT"].get("color_mode")
        # model config
        self.base_model_weights_file = self.cp["TRAIN"].get("base_model_weights_file")

        # train config
        self.use_base_model_weights = self.cp["TRAIN"].getboolean("use_base_model_weights")
        self.use_trained_model_weights = self.cp["TRAIN"].getboolean("use_trained_model_weights")
        self.use_best_weights = self.cp["TRAIN"].getboolean("use_best_weights")
        self.output_weights_name = self.cp["TRAIN"].get("output_weights_name")
        self.epochs = self.cp["TRAIN"].getint("epochs")
        self.batch_size = self.cp["TRAIN"].getint("batch_size")
        self.initial_learning_rate = self.cp["TRAIN"].getfloat("initial_learning_rate")
        self.train_steps = self.cp["TRAIN"].get("train_steps")
        self.patience_reduce_lr = self.cp["TRAIN"].getint("patience_reduce_lr")
        self.validation_steps = self.cp["TRAIN"].get("validation_steps")
        self.positive_weights_multiply = self.cp["TRAIN"].getfloat("positive_weights_multiply")
        self.use_class_balancing = self.cp["TRAIN"].getboolean("use_class_balancing")
        self.use_default_split = self.cp["TRAIN"].getboolean("use_default_split")
        self.force_resplit = self.cp["TRAIN"].getboolean("force_resplit")
        self.split_dataset_random_state = self.cp["TRAIN"].getint("split_dataset_random_state")
        self.show_model_summary = self.cp["TRAIN"].getboolean("show_model_summary")

        # DatasetConfig
        self.DSConfig = DatasetConfig(self.cp)
        self.running_flag_file = os.path.join(self.output_dir, ".training.lock")

    def check_training_lock(self):
        if os.path.isfile(self.running_flag_file):
            raise RuntimeError("A process is running in this directory!!!")
        else:
            open(self.running_flag_file, "a").close()

    def dump_history(self):
        # dump history
        print("** dump history **")
        with open(os.path.join(self.output_dir, "history.pkl"), "wb") as f:
            pickle.dump({"history": self.history.history, "auroc": self.auroc.aurocs, }, f)
        print("** done! **")

    def check_gpu_availability(self):
        self.model_train = self.model
        self.checkpoint = ModelCheckpoint(self.output_weights_path)
        print("** check multiple gpu availability **")
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            self.model_train = multi_gpu_model(self.model, gpus)
            # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
            self.checkpoint = MultiGPUModelCheckpoint(
                filepath=self.output_weights_path,
                base_model=self.model,
            )

    def prepare_datasets(self):
        if not self.force_resplit and self.use_trained_model_weights:
            # resuming mode
            print("** attempting to use trained model weights **")
            # load training status for resuming
            training_stats_file = os.path.join(self.output_dir, ".training_stats.json")
            if os.path.isfile(training_stats_file):
                training_stats = json.load(open(training_stats_file))
                initial_learning_rate = training_stats["lr"]
                print(f"** learning rate is set to previous final {initial_learning_rate} **")
            else:
                print("** trained model weights not found, starting over **")
                self.use_trained_model_weights = False

        print(f"backup config file to {output_dir}")
        shutil.copy(self.config_file, os.path.join(self.output_dir, os.path.split(self.config_file)[1]))

        if self.use_default_split:  # split train/dev/test
            datasets = ["train", "dev", "test"]
            for d in datasets:
                shutil.copy(f"./data/default_split/{d}.csv", self.output_dir)

        data_set = dsload.DataSet(image_dir=image_source_dir, data_entry=data_entry_file,
                                  train_ratio=train_patient_ratio,
                                  dev_ratio=dev_patient_ratio,
                                  output_dir=output_dir, img_dim=256, class_names=class_names,
                                  random_state=split_dataset_random_state, class_mode=class_mode,
                                  use_class_balancing=use_class_balancing,
                                  positive_weights_multiply=positive_weights_multiply,
                                  force_resplit=force_resplit)
        print("** create image generators **")
        train_generator = data_set.train_generator(verbosity=verbosity)
        dev_generator = data_set.dev_generator(verbosity=verbosity)

        # compute steps
        if self.train_steps == "auto":
            train_steps = train_generator.__len__()
        else:
            try:
                train_steps = int(self.train_steps)
            except ValueError:
                raise ValueError(f"""
                      train_steps: {train_steps} is invalid,
                      please use 'auto' or integer.
                      """)
        print(f"** train_steps: {train_steps} **")

        if self.validation_steps == "auto":
            validation_steps = dev_generator.__len__()
        else:
            try:
                validation_steps = int(self.validation_steps)
            except ValueError:
                raise ValueError(f"""
                      validation_steps: {validation_steps} is invalid,
                      please use 'auto' or integer.
                      """)
        print(f"** validation_steps: {validation_steps} **")

        # compute class weights
        print("** compute class weights from training data **")
        self.class_weights = data_set.class_weights()

    def prepare_model(self):
        print("** load model **")
        if not self.use_base_model_weights:
            self.base_model_weights_file = None
            print(f"** retrain without base model weight **")
        else:
            print(f"** loading base model weight from {base_model_weights_file} **")

        if self.use_trained_model_weights:
            if self.use_best_weights:
                model_weights_file = os.path.join(self.output_dir, f"best_{output_weights_name}")
                print(f"** loading best model weight from {model_weights_file} **")
            else:
                model_weights_file = os.path.join(self.output_dir, self.output_weights_name)
                print(f"** loading final model weight from {model_weights_file} **")
        else:
            model_weights_file = None

        self.model = get_model(self.class_names, self.base_model_weights_file, model_weights_file,
                               image_dimension=self.image_dimension, color_mode=self.color_mode,
                               class_mode=self.class_mode)
        if self.show_model_summary:
            print(self.model.summary())

        output_weights_path = os.path.join(self.output_dir, self.output_weights_name)
        print(f"** set output weights path to: {output_weights_path} **")
        self.check_gpu_availability()

        print("** compile model with class weights **")
        optimizer = Adam(lr=self.initial_learning_rate)
        self.model_train.compile(optimizer=optimizer, loss="binary_crossentropy")
        self.auroc = MultipleClassAUROC(generator=dev_generator, steps=validation_steps,
                                        class_names=self.class_names,
                                        class_mode=self.class_mode, weights_path=output_weights_path,
                                        stats=training_stats)

    def train(self):
        self.check_training_lock()
        os.makedirs(self.output_dir, exist_ok=True)  # check output_dir, create it if not exists
        self.prepare_datasets()
        self.prepare_model()

        callbacks = [
            self.checkpoint,
            TensorBoard(log_dir=os.path.join(self.output_dir, "logs"), batch_size=self.batch_size),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience_reduce_lr, verbose=1),
            self.auroc,
            SaveBaseModel(filepath=self.base_model_weights_file, save_weights_only=False)
        ]

        try:
            print("** training start **")
            print(f"** training with: {epochs} epochs @ {train_steps} steps/epoch **")
            self.history = self.model_train.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_steps,
                epochs=self.epochs,
                verbose=self.progress_verbosity,
                validation_data=dev_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                class_weight=class_weights,
                max_queue_size=4, workers=4, use_multiprocessing=True
            )
            self.dump_history()
        finally:
            os.remove(self.running_flag_file)
