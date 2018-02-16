import json
import os
import pickle
import shutil
from configparser import ConfigParser

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

from app.callback import MultipleClassAUROC, MultiGPUModelCheckpoint, SaveBaseModel
from app.datasets import dataset_loader as dsload
from app.models.model_factory import get_model
from app.utilities.Config import Config


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
    training_stats = {"run": 0}
    conf = None

    def __init__(self, config_file):
        if not os.path.isfile(config_file):
            raise FileExistsError(f"Configuration file {config_file} not found")

        cp = ConfigParser()
        cp.read(config_file)
        self.config_file = config_file
        self.conf = Config(cp=cp)
        if self.conf.gpu != 0:
            print(f"** Use assigned numbers of gpu ({self.conf.gpu}) only")
            CUDA_VISIBLE_DEVICES = ",".join([str(i) for i in range(self.conf.gpu)])
        else:
            print(f"** Use all gpus = ({len(get_available_gpus())})")
            CUDA_VISIBLE_DEVICES = ",".join([str(i) for i in range(len(get_available_gpus()))])
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{CUDA_VISIBLE_DEVICES}"

        self.fitter_kwargs = {"verbose": int(self.conf.progress_train_verbosity), "max_queue_size": 32, "workers": 32,
                              "epochs": self.conf.epochs, "use_multiprocessing": True}
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
        with open(os.path.join(self.conf.output_dir, "history.pkl"), "wb") as f:
            pickle.dump({"history": self.history.history, "auroc": self.auroc.aurocs, }, f)
        with open(self.conf.train_stats_file, 'w') as f:
            json.dump(self.training_stats, f)
        print("** done! **")

    def check_gpu_availability(self):
        self.model_train = self.model
        self.checkpoint = ModelCheckpoint(self.output_weights_path)
        print("** check multiple gpu availability **")
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            self.model_train = multi_gpu_model(self.model, gpus)
            self.model_train.base_model = self.model.base_model
            # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
            self.checkpoint = MultiGPUModelCheckpoint(
                filepath=self.output_weights_path,
                base_model=self.model,
            )

    def prepare_datasets(self):
        if self.MDConfig.is_resume_mode and os.path.isfile(self.conf.train_stats_file):
            print("** attempting to use trained model weights **")
            self.training_stats = json.load(open(self.conf.train_stats_file))
            self.conf.initial_learning_rate = self.training_stats["lr"]
            self.training_stats["run"] += 1
            print("** Run #{} - learning rate is set to previous final".format(self.training_stats["run"]), end="")
            print(f" {self.conf.initial_learning_rate} **")
        else:
            print("** Run #{self.run} - trained model weights not found, starting over **")

        print(f"backup config file to {self.conf.output_dir}")
        shutil.copy(self.config_file, os.path.join(self.conf.output_dir, os.path.split(self.config_file)[1]))

        data_set = dsload.DataSet(self.conf.DatasetConfig)

        print("** create image generators **")
        self.train_generator = data_set.train_generator(verbosity=self.conf.verbosity)
        self.dev_generator = data_set.dev_generator(verbosity=self.conf.verbosity)

        if self.conf.train_steps != "auto":
            print(f"** overriding train_steps: {self.conf.train_steps} **")
            self.fitter_kwargs["steps_per_epoch"] = self.conf.train_steps

        if self.conf.validation_steps != "auto":
            print(f"** overriding validation_steps: {self.conf.validation_steps} **")
            self.fitter_kwargs["validation_steps"] = self.conf.validation_steps

        print("** compute class weights from training data **")
        self.fitter_kwargs["class_weight"] = data_set.class_weights()

        self.fitter_kwargs["generator"] = self.train_generator
        self.fitter_kwargs["validation_data"] = self.dev_generator

    def prepare_model(self):
        print("** load model **")
        if self.MDConfig.base_model_weights_file is not None:
            print(f"** loading base model weight from {self.MDConfig.base_model_weights_file} **")
        else:
            print(f"** Retrain with {self.MDConfig.base_model_weights_file} **")

        self.model = get_model(self.DSConfig.class_names, self.MDConfig.base_model_weights_file,
                               self.MDConfig.trained_model_weights,
                               image_dimension=self.IMConfig.img_dim, color_mode=self.IMConfig.color_mode,
                               class_mode=self.DSConfig.class_mode)
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
            trained_base_weight = os.path.join(self.conf.output_dir, "trained_base_model_weight.h5")

            self.fitter_kwargs["callbacks"] = []
            self.fitter_kwargs["callbacks"].append(self.checkpoint)
            self.fitter_kwargs["callbacks"].append(TensorBoard(
                log_dir=os.path.join(self.conf.output_dir, "logs", "run{}".format(self.training_stats["run"])),
                batch_size=self.conf.batch_size, histogram_freq=0, write_graph=False,
                write_grads=False, write_images=False, embeddings_freq=0))
            self.fitter_kwargs["callbacks"].append(ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                                     patience=self.conf.patience_reduce_lr, verbose=1))
            self.fitter_kwargs["callbacks"].append(self.auroc)
            self.fitter_kwargs["callbacks"].append(SaveBaseModel(filepath=trained_base_weight, save_weights_only=False))

            print("** training start with parameters: **")
            for k, v in self.fitter_kwargs.items():
                print(f"\t{k}: {v}")
            self.history = self.model_train.fit_generator(**self.fitter_kwargs)
            self.dump_history()

        finally:
            os.remove(self.running_flag_file)
