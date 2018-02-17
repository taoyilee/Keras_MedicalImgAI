import json
import os
import shutil
import warnings

import keras.backend as kb
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

from app.datasets.dataset_utility import DataSequence
from app.utilities import Config


def load_generator_data(generator, steps, class_num, cam=False):
    """
    Return some data collected from a generator, use this to ensure all images
    are processed by exactly the same steps in the customized ImageDataGenerator.

    """
    batches_x = []
    batches_x_orig = []
    batches_y_classes = []
    for i in range(class_num):
        batches_y_classes.append([])
    for i in range(steps):
        if cam:
            batch_x, batch_y, batch_x_orig = generator.__getitem__(i)
            batches_x_orig.append(batch_x_orig)
        else:
            batch_x, batch_y = generator.__getitem__(i)
        batches_x.append(batch_x)
        for c, batch_y_class in enumerate(batch_y):
            batches_y_classes[c].append(batch_y_class)
    if cam:
        return np.concatenate(batches_x, axis=0), [np.concatenate(c, axis=0) for c in
                                                   batches_y_classes], np.concatenate(batches_x_orig, axis=0)
    else:
        return np.concatenate(batches_x, axis=0), [np.concatenate(c, axis=0) for c in batches_y_classes]


class ClearGeneratorCache(Callback):
    def __init__(self, train_generator: DataSequence, dev_generator: DataSequence):
        super(Callback, self).__init__()
        self.train_generator = train_generator
        self.dev_generator = dev_generator

    def on_train_begin(self, logs=None):
        print(f"** ClearGeneratorCache callback is ready")

    def on_epoch_begin(self, epoch, logs=None):
        self.train_generator.clear()
        self.dev_generator.clear()


class SaveBaseModel(Callback):
    def __init__(self, filepath, save_weights_only=False):
        super(Callback, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only

    def on_train_begin(self, logs={}):
        print(f"** SaveBaseModel callback is ready")

    def on_epoch_end(self, epoch, logs={}):
        if self.save_weights_only:
            self.model.base_model.save_weights(self.filepath, overwrite=True)
        else:
            self.model.base_model.save(self.filepath, overwrite=True)

    def on_train_end(self, logs={}):
        if self.save_weights_only:
            self.model.base_model.save_weights(self.filepath, overwrite=True)
        else:
            self.model.base_model.save(self.filepath, overwrite=True)


class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """

    def __init__(self, generator: DataSequence, steps, class_names, weights_path, config: Config,
                 class_mode="multiclass"):
        super(Callback, self).__init__()
        self.generator = generator
        self.steps = steps
        self.class_names = class_names
        self.class_mode = class_mode
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"best_{os.path.split(weights_path)[1]}",
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.stats_output_path = config.train_stats_file
        self.stats = json.load(open(self.stats_output_path))

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_train_begin(self, logs={}):
        print(f"** MultipleClassAUROC callback is ready")

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.

        """
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print(f"current learning rate: {self.stats['lr']}")

        """
        y_hat shape: (#samples, len(class_names))
        y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
        """

        test_generator = self.generator
        step_test = test_generator.__len__()
        y_hat = np.array(
            self.model.predict_generator(generator=test_generator, steps=step_test, max_queue_size=10, workers=1,
                                         use_multiprocessing=False, verbose=1)).squeeze()
        y = np.array(test_generator.targets()).squeeze()

        if self.class_mode == "multibinary":
            y_hat = y_hat.swapaxes(0, 1)
        print(f"*** epoch#{epoch + 1} dev auroc ***")
        print(f"y = {np.shape(y)}")
        print(f"y_hat = {np.shape(y_hat)}")

        current_auroc = roc_auc_score(y, y_hat, average=None)
        if len(self.class_names) != len(current_auroc):
            raise Exception(f"Wrong shape in either y or y_hat {len(self.class_names)} != {len(current_auroc)}")
        for i, v in enumerate(self.class_names):
            print(f" {i+1}. {v} AUC = {np.around(current_auroc[i], 3)}")

        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print(f"Mean AUC: {np.around(mean_auroc, 3)}\n")
        if mean_auroc > self.stats["best_mean_auroc"]:
            print(f"update best auroc from {self.stats['best_mean_auroc']} to {mean_auroc}")

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print(f"update log file: {self.best_auroc_log_path}")
            with open(self.best_auroc_log_path, "a") as f:
                f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc}, lr: {self.stats['lr']}\n")

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print(f"update model file: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_auroc"] = mean_auroc
        return


class MultiGPUModelCheckpoint(Callback):
    """
    Checkpointing callback for multi_gpu_model
    copy from https://github.com/keras-team/keras/issues/8463
    """

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(Callback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)
