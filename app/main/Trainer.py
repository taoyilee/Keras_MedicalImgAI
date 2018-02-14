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
from app.models.model_factory import get_model
from app.utilities.Config import Config


class Trainer:
    DSConfig = None
    IMConfig = None
    MDConfig = None

    # Runtime stuffs
    history = None
    auroc = None
    model = None
    model_train = None
    checkpoint = None
    output_weights_path = None
    train_generator = None
    dev_generator = None
    training_stats = []
    conf = None

    def __init__(self, config_file):
        if not os.path.isfile(config_file):
            raise FileExistsError(f"Configuration file {config_file} not found")

        cp = ConfigParser()
        cp.read(config_file)
        self.config_file = config_file
        self.conf = Config(cp=cp)
        if self.conf.gpu == 1:
            print(f"** Use single gpu only")
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        self.fitter_kwargs = {"verbose": self.conf.progress_verbosity, "max_queue_size": 16, "workers": 16,
                              "epochs": self.conf.epochs, "use_multiprocessing": False}
        self.parse_config()
        self.running_flag_file = os.path.join(self.conf.output_dir, ".training.lock")
        os.makedirs(self.conf.output_dir, exist_ok=True)  # check output_dir, create it if not exists
        self.check_training_lock()

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
                self.training_stats = json.load(open(self.conf.train_stats_file))
                self.conf.initial_learning_rate = self.training_stats["lr"]
                print(f"** learning rate is set to previous final {self.conf.initial_learning_rate} **")
            else:
                print("** trained model weights not found, starting over **")
                self.MDConfig.use_trained_model_weights = False

        print(f"backup config file to {self.conf.output_dir}")
        shutil.copy(self.config_file, os.path.join(self.conf.output_dir, os.path.split(self.config_file)[1]))

        data_set = dsload.DataSet(self.conf.DatasetConfig)

        print("** create image generators **")
        self.train_generator = data_set.train_generator(verbosity=self.conf.verbosity)
        self.dev_generator = data_set.dev_generator(verbosity=self.conf.verbosity)

        if self.conf.train_steps == "auto":
            self.conf.train_steps = self.train_generator.__len__()
        print(f"** train_steps: {self.conf.train_steps} **")

        if self.conf.validation_steps == "auto":
            self.conf.validation_steps = self.dev_generator.__len__()
        print(f"** validation_steps: {self.conf.validation_steps} **")

        # compute class weights
        print("** compute class weights from training data **")
        self.fitter_kwargs["class_weight"] = data_set.class_weights()

        self.fitter_kwargs["generator"] = self.train_generator
        self.fitter_kwargs["steps_per_epoch"] = self.conf.train_steps
        self.fitter_kwargs["validation_steps"] = self.conf.validation_steps
        self.fitter_kwargs["validation_data"] = self.dev_generator

    def prepare_model(self):
        print("** load model **")
        if self.MDConfig.base_model_weights_file is not None:
            print(f"** loading base model weight from {self.MDConfig.base_model_weights_file} **")
        else:
            print(f"** retrain without external base model weight **")

        if self.MDConfig.use_trained_model_weights:
            if self.MDConfig.use_best_weights:
                model_weights_file = os.path.join(self.conf.output_dir, f"best_{self.MDConfig.output_weights_name}")
                print(f"** loading best model weight from {model_weights_file} **")
            else:
                model_weights_file = os.path.join(self.conf.output_dir, self.MDConfig.output_weights_name)
                print(f"** loading final model weight from {model_weights_file} **")
        else:
            model_weights_file = None

        self.model = get_model(self.DSConfig.class_names, self.MDConfig.base_model_weights_file, model_weights_file,
                               image_dimension=self.IMConfig.img_dim, color_mode=self.IMConfig.color_mode,
                               class_mode=self.DSConfig.class_mode,
                               use_base_model_weights=self.MDConfig.use_base_model_weights)
        if self.MDConfig.show_model_summary:
            print(self.model.summary())

        self.output_weights_path = os.path.join(self.conf.output_dir, self.MDConfig.output_weights_name)
        print(f"** set output weights path to: {self.output_weights_path} **")
        self.check_gpu_availability()

        print("** compile model with class weights **")
        optimizer = Adam(lr=self.conf.initial_learning_rate)
        self.model_train.compile(optimizer=optimizer, loss="binary_crossentropy")
        self.auroc = MultipleClassAUROC(generator=self.dev_generator, steps=self.conf.validation_steps,
                                        class_names=self.DSConfig.class_names,
                                        class_mode=self.DSConfig.class_mode, weights_path=self.output_weights_path,
                                        stats=self.training_stats)

    def train(self):
        try:
            self.prepare_datasets()
            self.prepare_model()

            callbacks = [
                self.checkpoint,
                TensorBoard(log_dir=os.path.join(self.conf.output_dir, "logs"), batch_size=self.conf.batch_size),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.conf.patience_reduce_lr, verbose=1),
                self.auroc,
                SaveBaseModel(filepath=self.MDConfig.base_model_weights_file, save_weights_only=False)
            ]
            self.fitter_kwargs["callbacks"] = callbacks

            print("** training start with parameters: **")
            for k, v in self.fitter_kwargs.items():
                print(f"\t{k}: {v}")
            self.history = self.model_train.fit_generator(**self.fitter_kwargs)
            self.dump_history()
        finally:
            os.remove(self.running_flag_file)
