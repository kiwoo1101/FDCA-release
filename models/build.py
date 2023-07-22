from .baseline import baseline
from .baseline_ca import baseline_ca


__model_factory = {
    'AGW': baseline,
    'BoT': baseline,
    'RGA': baseline,
    'CA': baseline_ca,
}


def model_names():
    return list(__model_factory.keys())


def init_model(name, **kwargs):
    if name not in __model_factory.keys():
        raise KeyError("Invalid model name, got '{}', but expected to be one of {}".
                       format(name, __model_factory.keys()))
    print("=>Initializing model: {}".format(name))
    model = __model_factory[name](**kwargs)
    print("{} model size: {:.5f}M".format(name, sum(p.numel() for p in model.parameters()) / 1000000.0))
    return model
