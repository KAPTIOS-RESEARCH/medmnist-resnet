import torch
import os
import logging
import wandb


class EarlyStopping:

    def __init__(self, patience=5, delta=0, verbose=True, enable_wandb=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.stop = False
        self.enable_wandb = enable_wandb

    def __call__(self, model: torch.nn.Module, val_loss: float, log_dir: str, epoch: int):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            model_object = {
                'model_state_dict': model.state_dict(),
                'min_loss': self.best_loss,
                'last_epoch': epoch
            }
            torch.save(model_object, os.path.join(log_dir, 'best_model.pth'))
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
