import argparse

import keras.backend as K
import numpy as np

from app.main.Trainer import Trainer


def main(config_file):
    tr = Trainer(config_file)
    tr.prepare_datasets()
    tr.prepare_loss_function()
    targets = np.array(tr.train_generator.targets())
    targets = targets[0:3]
    # pred = np.random.rand(14)
    pred = 0.3 * np.ones((3, 14), dtype=K.floatx())
    loss = tr.loss_function(targets, pred)
    print(K.eval(loss))


if __name__ == "__main__":
    '''
    Entry Point
    '''

    # use argparse to accept command line variables (config.ini)
    parser = argparse.ArgumentParser(description='Train a specfic dataset')
    parser.add_argument('config', metavar='config', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    main(config_file=args.config)
