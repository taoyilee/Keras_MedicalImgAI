def assignIfNotNull(config, default): return config if config is not None else default


def returnPropertIfNotNull(config):
    if config is None or config == "":
        raise ValueError()
    else:
        return config
