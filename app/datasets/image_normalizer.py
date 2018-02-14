from app.datasets.NormalizeConfig import NormalizeConfig


class ImageNormalizer:
    def __init__(self, nrm_config):
        """

        :param nrm_config:
        :type nrm_config: NormalizeConfig
        """
        self.nrm_config = nrm_config

    def normalize(self, image):
        if self.nrm_config.normalize_by_mean_var:
            image_n = image - self.nrm_config.norm_mean
            image_n = image_n / self.nrm_config.norm_stdev

        return image_n
