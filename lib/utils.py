import os
import shutil
import glob
import random

import torch
import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def suggest_config(config, search_attr, trial):
    for attr, search_type in search_attr:
        assert search_type in ["categorical", "discrete_uniform",
                               "float", "int", "loguniform",
                               "uniform"]
        dict_attr = config
        attr_key = attr.split(",")[-1]
        for key in attr.split(",")[:-1]:
            dict_attr = dict_attr[key]
        assert isinstance(dict_attr, list)
        if search_type == "categorical":
            dict_attr[attr_key] = trial.suggest_categorical(attr, dict_attr[attr_key])
        elif search_type == "discrete_uniform":
            assert len(dict_attr[attr_key]) == 3
            dict_attr[attr_key] = trial.suggest_discrete_uniform(attr, *dict_attr[attr_key])
        elif search_type == "float":
            assert len(dict_attr[attr_key]) == 2
            dict_attr[attr_key] = trial.suggest_float(attr, *dict_attr[attr_key])
        elif search_type == "int":
            assert len(dict_attr[attr_key]) == 2
            dict_attr[attr_key] = trial.suggest_int(attr, *dict_attr[attr_key])
        elif search_type == "loguniform":
            assert len(dict_attr[attr_key]) == 2
            dict_attr[attr_key] = trial.suggest_loguniform(attr, *dict_attr[attr_key])
        elif search_type == "uniform":
            assert len(dict_attr[attr_key]) == 2
            dict_attr[attr_key] = trial.suggest_uniform(attr, *dict_attr[attr_key])
    return config


def validate_config(config):
    if config["hyper_search"] is not None:
        assert config["training"]["early_stop_patience"] is None
        assert config["training"]["overfit_mode"] == False
        assert config["training"]["debug"] == False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seet(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def record_code_files(log_folder):
    os.makedirs(os.path.join(log_folder, "code"))
    py_files = glob.glob("**/*.py", recursive=True)
    for f in py_files:
        if f.split("/")[0] == "experiments":
            continue
        folder_path = os.path.join(log_folder,"code", "/".join(f.split("/")[:-1]))
        os.makedirs(folder_path ,exist_ok=True)
        shutil.copy2(f, folder_path)


def get_log_folder(log_name):
    if "datetime" in log_name:
        from datetime import datetime
        now = datetime.now()  
        date_time = now.strftime("%m-%d-%Y_%H:%M:%S")
        return os.path.join("experiments", log_name.replace("datetime", date_time))
    else:
        return os.path.join("experiments", log_name)
