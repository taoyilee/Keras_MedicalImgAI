from app.datasets.NormalizeConfig import NormalizeConfig


class ImageNormalizer:
    def __init__(self, nrm_config):
        """

        :param nrm_config:
        :type nrm_config: NormalizeConfig
        """
        self.nrm_config = nrm_config

    def normalize(self, image):
        return (image - self.nrm_config.norm_mean) / self.nrm_config.norm_stdev
