import argparse

from app.main.Trainer import Trainer


def main(config_file):
    tr = Trainer(config_file)
    tr.prepare_datasets()


if __name__ == "__main__":
    '''
    Entry Point
    '''

    # use argparse to accept command line variables (config.ini)
    parser = argparse.ArgumentParser(description='Train a specfic dataset')
    parser.add_argument('config', metavar='config', type=str, help='an integer for the accumulator')
    args = parser.parse_args()

    main(config_file=args.config)
