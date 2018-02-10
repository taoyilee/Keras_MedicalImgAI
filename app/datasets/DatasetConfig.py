from app.imagetoolbox.ImageConfig import ImageConfig
from app.utilities.util_config import cond_assign


class DatasetConfig:
    image_dir = ""
    data_entry = ""
    class_names = ""
    output_dir = ""
    random_state = 0
    train_ratio = 70
    dev_ratio = 10
    batch_size = 32
    img_dim = 256
    scale = 1. / 255
    class_mode = "multiclass"
    use_class_balancing = True
    positive_weights_multiply = 1
    force_resplit = False

    def __init__(self, cp):
        """

        :param cp: Config Parser Section
        :type cp: configparser.ConfigParser
        """
        self.cp = cp
        self.parse_config()

    @property
    def ImageConfig(self):
        return ImageConfig(self.cp)

    def parse_config(self):
        # required configs
        self.image_dir = self.cp["IMAGE"].get("image_dir")
        self.data_entry = self.cp["DATASET"].get("data_entry")
        self.class_names = self.cp["DATASET"].get("class_names")
        self.output_dir = self.cp["DATASET"].get("output_dir")

        # optional configs
        positive_weights_multiply = self.cp["DATASET"].getfloat("positive_weights_multiply")
        force_resplit = self.cp["DATASET"].getboolean("force_resplit")
        use_class_balancing = self.cp["DATASET"].getboolean("use_class_balancing")
        train_ratio = self.cp["DATASET"].getfloat("train_ratio")
        dev_ratio = self.cp["DATASET"].getfloat("dev_ratio")
        batch_size = self.cp["DATASET"].getint("batch_size")
        class_mode = self.cp["DATASET"].get("class_mode")
        random_state = self.cp["DATASET"].getint("random_state")

        self.random_state = cond_assign(random_state, self.random_state)
        self.train_ratio = cond_assign(train_ratio, self.train_ratio)
        self.dev_ratio = cond_assign(dev_ratio, self.dev_ratio)
        self.batch_size = cond_assign(batch_size, self.batch_size)
        self.class_mode = cond_assign(class_mode, self.class_mode)
        self.positive_weights_multiply = cond_assign(positive_weights_multiply, self.positive_weights_multiply)
        self.use_class_balancing = cond_assign(use_class_balancing, self.use_class_balancing)
        self.force_resplit = cond_assign(force_resplit, self.force_resplit)
