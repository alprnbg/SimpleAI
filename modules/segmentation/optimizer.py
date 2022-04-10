import torch.optim

def get_optimizer(config, trainable_params):
    name = config["name"].lower()
    if name == "radam":
        return torch.optim.RAdam(trainable_params, config["lr"])
    elif name == "adam":
        return torch.optim.Adam(trainable_params, config["lr"])
    else:
        assert False