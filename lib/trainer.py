import os
import sys
import copy

import torch
import optuna

from lib.utils import AverageMeter, count_params
from modules.segmentation import *


class Trainer:
    def __init__(self, config, log_folder):
        self.max_epoch = config["max_epoch"]
        self.debug_mode = config["debug_mode"]
        self.log_step = config["log_step"] # todo tensorboard
        self.log_folder = log_folder
        self.train_loader, self.val_loader = get_dataloader(config["data"], device=config["device"],
                                                            overfit_mode=config["overfit_mode"])
        self.model = get_model(config["model"]).to(config["device"])
        self.loss = get_loss(config["loss"])
        self.metrics = get_metrics(config["metrics"])
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_optimizer(config["optimizer"], trainable_params=trainable_params)
        self.scheduler = get_scheduler(config["scheduler"], optimizer=self.optimizer)
        self.data_writer = get_datawriter(config["data_writer"])
        self._init_metrics()

    def _init_metrics(self):
        self.logs = {"train_loss":(AverageMeter(), None), "val_loss":(AverageMeter(), None)}
        for k, metric_func in self.metrics.train_metrics.items():
            self.logs["train_"+k] = (AverageMeter(), metric_func)
        for k, metric_func in self.metrics.val_metrics.items():
            self.logs["val_"+k] = (AverageMeter(), metric_func)
    
    def _reset_metrics(self, mode):
        for k, v in self.logs.items():
            if k.startswith(mode):
                v[0].reset()

    def train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics("train")
        num_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            self._train_iteration(batch)
            self._print_log(epoch, int(100*i/len(self.train_loader)), "train", i+1==num_batches)
        if self.scheduler is not None:
            self.scheduler.step()
        self.val_epoch(epoch)
        return copy.deepcopy(self.logs)

    def val_epoch(self, epoch):
        self.model.eval()
        self._reset_metrics("val")
        num_batches = len(self.val_loader)
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                output = self.model(batch["input"])
                loss, batch_size = self.loss(output, batch["target"])
                self.logs["val_loss"][0].update(loss.item(), batch_size)
                for k,v in self.logs.items():
                    if k.startswith("val") and k!="val_loss":
                        v[0].update(v[1](output, batch["target"]), batch_size)
                self._print_log(epoch, int(100*i/len(self.val_loader)), "val", i+1==num_batches)
    
    def _train_iteration(self, batch):
        self.optimizer.zero_grad()
        output = self.model(batch["input"])
        loss, batch_size = self.loss(output, batch["target"])
        loss.backward()
        self.optimizer.step()
        self.logs["train_loss"][0].update(loss.item(), batch_size)
        for k,v in self.logs.items():
            if k.startswith("train") and k!="train_loss":
                v[0].update(v[1](output, batch["target"]), batch_size)
    
    def save_checkpoint(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.log_folder, f"{name}.pth"))
    
    def debug_epoch(self):
        os.makedirs(os.path.join(self.log_folder, "debug_data"))
        for batch_idx, batch in enumerate(self.train_loader):
            self.data_writer.write_input_target(batch, batch_idx, os.path.join(self.log_folder, 
                                                                               "debug_data"))

    def _print_log(self, epoch, percent, mode, new_line=False):
        print_log = []
        for k,v in self.logs.items():
            if k.startswith(mode):
                print_log.append(k)
                print_log.append(v[0].val)
                print_log.append(v[0].avg)
        print(("Epoch {}/{}|{}%" + "  ||  {}: {:.3f}|Avg:{:.3f}"*int(len(print_log)/3)).\
            format(epoch, self.max_epoch, percent, *print_log), end='\r')
        if new_line:
            print()


def run_training(config, log_folder, trial=None):
    trainer = Trainer(config, log_folder)
    patience = 0
    best_val = None
    print(f"# of parameters in the model: {count_params(trainer.model)}\n")
    if config["debug_mode"]:
        print("Debugging inputs and targets...")
        trainer.debug_epoch()
        sys.exit(0)
    for e in range(1, config["max_epoch"]+1):
        print("----- Epoch {} -----".format(e))
        epoch_logs = trainer.train_epoch(e)
        trainer.save_checkpoint("last")
        current_val = epoch_logs[config["goal_metric"]][0].avg
        improvement = True if (best_val is None) \
                or (config["direction"] == "minimize" and current_val<best_val) \
                or (config["direction"] == "maximize" and current_val>best_val) \
                else False
        if improvement:
            best_val = current_val
            patience = 0
            trainer.save_checkpoint("best")
            print(f"Improvement at epoch {e} with best value of {config['goal_metric']}={best_val}")
            print(f"Model saved to {os.path.join(trainer.log_folder)}")
        else:
            patience += 1
        # early stop
        if config["early_stop_patience"] is not None and patience >= config["early_stop_patience"]:
            print(f"Early Stop at epoch {e} with best value of {config['goal_metric']}={best_val}")
            break
        # optune prunning
        if trial is not None:        
            trial.report(best_val, e-1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    torch.cuda.empty_cache()
    return best_val
