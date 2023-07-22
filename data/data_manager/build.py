from .market1501 import Market1501

__datasets_factory = {
    'market': Market1501,
}


def datasets_names():
    return __datasets_factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __datasets_factory.keys():
        raise KeyError("Unknown datasets: {}, expected to be one of {}".format(name, __datasets_factory.keys()))
    return __datasets_factory[name](*args, **kwargs)


if __name__ == '__main__':
    pass

