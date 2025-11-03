import wandb
import logging
import time
import torch
import os
from torch import nn
from torch.optim import Adam
from src.utils.config import instanciate_module
from src.optimisation.early_stopping import EarlyStopping
from tqdm import tqdm
from src.utils.summary import get_acronym


class BaseTrainer(object):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.early_stop = EarlyStopping(
            patience=parameters['early_stopping_patience'], enable_wandb=parameters['track']) if parameters['early_stopping_patience'] else None

        if parameters['optimizer'] is not None:
             self.optimizer = instanciate_module(
                parameters['optimizer']['module_name'],
                parameters['optimizer']['class_name'],
                {**parameters['optimizer']['parameters'],
                "params": self.model.parameters(),
            })
        else:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=parameters['lr'],
                weight_decay=parameters['weight_decay']
            )
        
        if parameters['lr_scheduler'] is not None:
             self.lr_scheduler = instanciate_module(
                parameters['lr_scheduler']['module_name'],
                parameters['lr_scheduler']['class_name'],
                {**parameters['lr_scheduler']['parameters'], 'optimizer': self.optimizer})
        else:
            self.lr_scheduler = None
    
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                            parameters['loss']['class_name'],
                                            parameters['loss']['parameters'])

    def _run_epoch(self, loader, phase="train", epoch=None, num_epochs=None, evaluator=None):
        is_train = phase == "train"
        self.model.train(mode=is_train)
        total_loss = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        desc = (
            f"Epoch [{epoch + 1}/{num_epochs}] - {phase.capitalize()} - LR {current_lr:.6f}"
            if epoch is not None else f"Running {phase} phase"
        )
        metric_sums = {k: 0.0 for k in evaluator.metrics}
        with tqdm(loader, desc=desc, ncols=150) as pbar:
            for i, (data, targets) in enumerate(loader, start=1):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                batch_metrics = evaluator.compute_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    metric_sums[k] += val

                total_loss += loss.item()
                avg_loss = total_loss / i
                avg_metrics = {k: metric_sums[k] / i for k in metric_sums}

                postfix = {'Loss': f'{avg_loss:.6f}'}
                for k, avg_val in avg_metrics.items():
                    postfix[get_acronym(k)] = f"{avg_val:.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)
        avg_loss = total_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in metric_sums.items()}
        return avg_loss, avg_metrics
                
    def train(self, train_loader, epoch=None, num_epochs=None, evaluator=None):
        return self._run_epoch(train_loader, phase="train", epoch=epoch, num_epochs=num_epochs, evaluator=evaluator)

    def validation(self, val_loader, epoch=None, num_epochs=None, evaluator=None):
        return self._run_epoch(val_loader, phase="validation", epoch=epoch, num_epochs=num_epochs, evaluator=evaluator)

    def test(self, test_loader, epoch=None, num_epochs=None, evaluator=None):
        return self._run_epoch(test_loader, phase="test", epoch=epoch, num_epochs=num_epochs, evaluator=evaluator)

    def fit(self, train_dl, val_dl, test_dl, log_dir: str, evaluator):
        start_time = time.time()
        num_epochs = self.parameters['num_epochs']
        best_loss = None
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train(train_dl, epoch, num_epochs, evaluator)
            val_loss, val_metrics = self.validation(val_dl, epoch, num_epochs, evaluator)
            if self.parameters['track']:
                log_data = {
                    f"Train/{self.parameters['loss']['class_name']}": train_loss,
                    f"Validation/{self.parameters['loss']['class_name']}": val_loss,
                    "_step_": epoch
                }
                if train_metrics:
                    for metric_name, value in train_metrics.items():
                        log_data[f"Train/{metric_name}"] = value
                if val_metrics:
                    for metric_name, value in val_metrics.items():
                        log_data[f"Validation/{metric_name}"] = value

                wandb.log(log_data)

            if self.lr_scheduler:
                self.lr_scheduler.step(train_loss)

            if self.early_stop:
                self.early_stop(self.model, val_loss, log_dir, epoch)
                if self.early_stop.stop:
                    logging.info(
                        f"Val loss did not improve for {self.early_stop.patience} epochs.")
                    logging.info(
                        'Training stopped by early stopping mechanism.')
                    break
            else:
                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    model_object = {
                        'model_state_dict': self.model.state_dict(),
                        'min_loss': best_loss,
                        'last_epoch': epoch
                    }
                    torch.save(model_object, os.path.join(
                        log_dir, 'best_model.pth'))

        _, test_metrics = self.test(test_dl, None, None, evaluator)
        if self.parameters['track']:
            if test_metrics:
                for metric_name, value in test_metrics.items():
                    log_data[f"Test/{metric_name}"] = value    
        end_time = time.time()
        logging.info(
            f"Training completed in {end_time - start_time:.2f} seconds.")