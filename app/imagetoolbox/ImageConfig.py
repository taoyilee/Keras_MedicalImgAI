from app.utilities.util_config import cond_assign


class ImageConfig:
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
    color_mode = "grayscale"

    def __init__(self, cp):
        """

        :param cp: Config Parser Section
        :type cp: configparser.ConfigParser
        """
        self.cp = cp
        self.parse_config()

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def parse_config(self):
        # required configs
        self.image_dir = self.cp["IMAGE"].get("image_dir")
        self.data_entry = self.cp["DATASET"].get("data_entry")

        # optional configs
        color_mode = self.cp["IMAGE"].get("color_mode")
        img_dim = self.cp["IMAGE"].getint("img_dim")
        scale = self.cp["IMAGE"].getfloat("scale")

        self.img_dim = cond_assign(img_dim, self.img_dim)
        self.scale = cond_assign(scale, self.scale)
        self.color_mode = cond_assign(color_mode, self.color_mode)
