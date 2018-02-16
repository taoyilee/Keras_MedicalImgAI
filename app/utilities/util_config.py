def assign_fallback(config, default): return config if config is not None else default


def assign_raise(config):
    if config is None or config == "":
        raise ValueError()
    else:
        return config
