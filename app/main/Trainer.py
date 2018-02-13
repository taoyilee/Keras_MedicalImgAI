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
from app.models.densenet121 import get_model
from app.utilities.Config import Config


class Trainer:
    _DSConfig = None
    _IMConfig = None
    _MDConfig = None

    # Runtime stuffs
    history = None
    auroc = None
    model = None
    model_train = None
    checkpoint = None

    conf = None

    def __init__(self, config_file):
        if not os.path.isfile(config_file):
            raise FileExistsError(f"Configuration file {config_file} not found")

        cp = ConfigParser()
        cp.read(config_file)
        self.config_file = config_file
        self.conf = Config(cp=cp)
        self.fitter_kwargs = {"verbose": self.conf.progress_verbosity, "max_queue_size": 4, "workers": 4,
                              "use_multiprocessing": True}
        self.parse_config()
        self.running_flag_file = os.path.join(self.conf.output_dir, ".training.lock")
        self.check_training_lock()
        os.makedirs(self.conf.output_dir, exist_ok=True)  # check output_dir, create it if not exists

    def parse_config(self):
        self.DSConfig = self.conf.DatasetConfig
        self.IMConfig = self.conf.ImageConfig
        self.MDConfig = self.conf.ModelConfig

    def check_training_lock(self):
        if os.path.isfile(self.running_flag_file):
            raise RuntimeError(f"A process is running in this directory {self.running_flag_file} !!!")
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
        if self.conf.isResumeMode:
            print("** attempting to use trained model weights **")
            # load training status for resuming
            if os.path.isfile(self.conf.train_stats_file):
                training_stats = json.load(open(self.conf.train_stats_file))
                self.conf.initial_learning_rate = training_stats["lr"]
                print(f"** learning rate is set to previous final {self.conf.initial_learning_rate} **")
            else:
                print("** trained model weights not found, starting over **")
                self.MDConfig.use_trained_model_weights = False

        print(f"backup config file to {self.conf.output_dir}")
        shutil.copy(self.config_file, os.path.join(self.conf.output_dir, os.path.split(self.config_file)[1]))



        data_set = dsload.DataSet(self.conf.DatasetConfig)
        # image_dir = image_source_dir, data_entry = data_entry_file,
        # train_ratio = train_patient_ratio,
        # dev_ratio = dev_patient_ratio,
        # output_dir = output_dir, img_dim = 256, class_names = class_names,
        # random_state = split_dataset_random_state, class_mode = class_mode,
        # use_class_balancing = use_class_balancing,
        # positive_weights_multiply = positive_weights_multiply,
        # force_resplit = force_resplit

        print("** create image generators **")
        train_generator = data_set.train_generator(verbosity=self.verbosity)
        dev_generator = data_set.dev_generator(verbosity=self.verbosity)

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
        self.auroc = MultipleClassAUROC(generator=self.dev_generator, steps=self.validation_steps,
                                        class_names=self.class_names,
                                        class_mode=self.class_mode, weights_path=output_weights_path,
                                        stats=self.training_stats)

    def train(self):
        try:
            self.prepare_datasets()
            self.prepare_model()

            callbacks = [
                self.checkpoint,
                TensorBoard(log_dir=os.path.join(self.output_dir, "logs"), batch_size=self.conf.batch_size),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience_reduce_lr, verbose=1),
                self.auroc,
                SaveBaseModel(filepath=self.base_model_weights_file, save_weights_only=False)
            ]
            self.fitter_kwargs["callbacks"] = callbacks

            print("** training start **")
            print(f"** training with: {epochs} epochs @ {train_steps} steps/epoch **")
            self.history = self.model_train.fit_generator(**self.fitter_kwargs)
            self.dump_history()
        finally:
            os.remove(self.running_flag_file)
