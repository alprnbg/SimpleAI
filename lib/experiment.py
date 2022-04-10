import os

import optuna
import torch

from lib.trainer import run_training
from lib.utils import suggest_config, set_seet, record_code_files, get_log_folder


training_config = None
search_attr = None
log_folder = None

def run_experiment(config):
    global training_config, search_attr, log_folder
    
    set_seet(config["seed"])
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    
    training_config = config["training"]
    hyper_config = config["hyper_search"]
    log_folder = get_log_folder(training_config["log_name"])
    os.makedirs(log_folder)
    record_code_files(log_folder)
    print("\nLOG FOLDER:", log_folder)
    
    if hyper_config is not None:
        search_attr = hyper_config["search_attr"]
        study = optuna.create_study(direction=training_config["direction"], 
                                    pruner=hyper_config["pruner"],
                                    study_name=hyper_config["study_name"])
        study.optimize(objective, n_trials=hyper_config["n_trials"], timeout=hyper_config["timeout"])
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        run_training(training_config, log_folder)


def objective(trial):
    global training_config, search_attr, log_folder
    config = suggest_config(training_config, search_attr, trial)
    return run_training(config, log_folder, trial)
