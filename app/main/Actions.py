import os
from configparser import ConfigParser

import GPUtil

from app.utilities.Config import Config


class Actions:
    conf: Config = None
    DSConfig = None
    IMConfig = None
    MDConfig = None

    def __init__(self, config_file: str):
        if not os.path.isfile(config_file):
            raise FileExistsError(f"Configuration file {config_file} not found")
        cp = ConfigParser()
        cp.read(config_file)
        self.config_file = config_file
        self.conf = Config(cp=cp)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress debug message and warnings

        self.parse_config()
        self.check_gpu()

    def check_gpu(self):
        if self.conf.gpu != 0:
            print(f"** Use assigned numbers of gpu ({self.conf.gpu}) only")
            CUDA_VISIBLE_DEVICES = ",".join([str(i) for i in range(self.conf.gpu)])
        else:
            try:
                gpus = len(GPUtil.getGPUs())
            except ValueError:
                gpus = 1
            print(f"** Use all gpus = ({gpus})")
            CUDA_VISIBLE_DEVICES = ",".join([str(i) for i in range(gpus)])
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{CUDA_VISIBLE_DEVICES}"

    def parse_config(self):
        self.DSConfig = self.conf.DatasetConfig
        self.IMConfig = self.conf.ImageConfig
        self.MDConfig = self.conf.ModelConfig
