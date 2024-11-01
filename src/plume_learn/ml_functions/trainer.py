import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class ModelTrainer:
    def __init__(self, model, loss_calculator, optimizer, device, scheduler=None):
        self.model = model
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.history = defaultdict(list)

    def train_epochs(self, train_dl, valid_dl_list, valid_name_list, epochs, start=0, 
                     valid_every_epochs=1, model_dir=None, save_per_epoch=1):
        
        if isinstance(valid_every_epochs, int) and valid_dl_list:
            valid_every_epochs = [valid_every_epochs] * len(valid_dl_list)
        elif isinstance(valid_every_epochs, list) and len(valid_every_epochs) != len(valid_dl_list):
            raise ValueError("The length of valid_every_epochs should match the number of cross-validation datasets.")
        
        if model_dir and not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        for epoch_idx in range(start, epochs + start):
            print(f"Epoch: {epoch_idx + 1}/{epochs + start}")
            
            train_losses = self.run_epoch(train_dl, is_training=True)
            self.print_losses("Training", train_losses)
            
            metadata = {'epoch': epoch_idx, **{f'train_{k}': v for k, v in train_losses.items()}}

            if valid_dl_list:
                for i, (cv_dl, cv_name) in enumerate(zip(valid_dl_list, valid_name_list)):
                    if (epoch_idx + 1) % valid_every_epochs[i] == 0:
                        valid_losses = self.run_epoch(cv_dl, is_training=False)
                        self.print_losses(cv_name, valid_losses)
                        metadata.update({f'{cv_name}_{k}': v for k, v in valid_losses.items()})

            if model_dir is not None and (epoch_idx + 1) % save_per_epoch == 0:
                self.save_model(os.path.join(model_dir, f'epoch-{epoch_idx + 1}.pt'))

            for key, value in metadata.items():
                if isinstance(value, torch.Tensor):
                    self.history[key].append(value.item())
                else:
                    self.history[key].append(value)
                
        return dict(self.history)

    def run_epoch(self, dataloader, is_training):
        epoch_losses = defaultdict(float)
        self.model.train(is_training)

        with torch.set_grad_enabled(is_training):
            for batch in tqdm(dataloader):
                batch_losses = self.run_batch(batch, is_training)
                for k, v in batch_losses.items():
                    epoch_losses[k] += v

        return {k: v / len(dataloader.dataset) for k, v in epoch_losses.items()}

    def run_batch(self, batch, is_training):
        # print(batch[0].dtype, batch[1].dtype)
        inputs, labels = [t.to(self.device).float() for t in batch]
        # print(inputs.dtype, labels.dtype)
        self.model = self.model.to(self.device)

        if is_training:
            self.optimizer.zero_grad()

        outputs = self.model(inputs)
        losses = self.loss_calculator(outputs, inputs, labels)

        if is_training:
            losses['Total_loss'].backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        return losses

    def print_losses(self, phase, losses):
        print(f"{phase}:")
        for k, v in losses.items():
            if v < 1e-3:
                print(f"  {k}: {v:.4e}")
            else:
                print(f"  {k}: {v:.4f}")

    def prepare_model_for_saving(self, model):
        # Check if model is using DataParallel
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        # Check if model is using DistributedDataParallel
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        
        # Move model to CPU if it's on GPU
        if next(model.parameters()).is_cuda:
            model = model.cpu()
        
        return model

    def save_model(self, path):
        model = self.prepare_model_for_saving(self.model)
        torch.save(model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def get_history(self):
        return dict(self.history)
    
    
    def plot_training_metrics(self, epochs, metrics_dict):
        
        fig, axes = plt.subplots(len(metrics_dict.keys()), 1, figsize=(6, 3*len(metrics_dict.keys())))
        for i, (metric_name, metric_values) in enumerate(metrics_dict.items()):
            axes[i].plot(epochs, metric_values, 'k-')
            axes[i].set_title(metric_name)
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel(metric_name)
        
        plt.tight_layout()
        plt.show()
        