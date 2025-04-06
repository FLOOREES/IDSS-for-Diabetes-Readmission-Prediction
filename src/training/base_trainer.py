# src/training/base_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import time
import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """ Abstract Base Class for PyTorch Trainers. """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_name: str,
        optimizer_params: Dict[str, Any],
        scheduler_name: Optional[str],
        scheduler_params: Optional[Dict[str, Any]],
        criterion: nn.Module, # Loss function
        epochs: int,
        device: torch.device,
        checkpoint_dir: str,
        model_name: str, # e.g., 'autoencoder' or 'predictor'
        early_stopping_patience: Optional[int] = None,
        gradient_clip_value: Optional[float] = None,
        logger: logging.Logger = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self._setup_optimizer(optimizer_name, optimizer_params)
        self.scheduler = self._setup_scheduler(scheduler_name, scheduler_params)
        self.criterion = criterion # Specific loss calculation handled in subclasses
        self.epochs = epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_value = gradient_clip_value
        self.logger = logger or logging.getLogger(__name__)

        self.history: Dict[str, list] = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _setup_optimizer(self, name: str, params: Dict[str, Any]) -> optim.Optimizer:
        """Initializes the optimizer."""
        pass # Implementation needed (e.g., Adam, RMSprop)

    def _setup_scheduler(self, name: Optional[str], params: Optional[Dict[str, Any]]) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Initializes the learning rate scheduler."""
        pass # Implementation needed (e.g., ReduceLROnPlateau, StepLR)

    @abstractmethod
    def _calculate_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Calculates the loss for a batch, handling masking."""
        pass

    def _train_epoch(self) -> float:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        pass # Implementation needed (iterate loader, forward, loss, backward, step)

    def _val_epoch(self) -> float:
        """Runs one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        pass # Implementation needed (iterate loader, forward, loss)

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """Saves model checkpoint."""
        pass # Implementation needed

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.model_name} for {self.epochs} epochs on {self.device}.")
        start_time = time.time()
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            epoch_duration = time.time() - epoch_start_time

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Duration: {epoch_duration:.2f}s")

            # LR Scheduling
            if self.scheduler:
                 # Special handling for ReduceLROnPlateau
                 if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                     self.scheduler.step(val_loss)
                 else:
                     self.scheduler.step() # For schedulers like StepLR

            # Checkpoint Saving & Early Stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, is_best=True)
                self.logger.info(f"Validation loss improved to {val_loss:.4f}. Saved best model.")
            else:
                self.epochs_no_improve += 1
                self.logger.info(f"Validation loss did not improve for {self.epochs_no_improve} epoch(s).")
                self._save_checkpoint(epoch, is_best=False) # Save latest checkpoint

            if self.early_stopping_patience and self.epochs_no_improve >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        total_duration = time.time() - start_time
        self.logger.info(f"Training finished in {total_duration:.2f}s. Best Validation Loss: {self.best_val_loss:.4f}")
        # Load best model weights at the end
        self._load_best_checkpoint()


    def _load_best_checkpoint(self):
        """Loads the best model checkpoint found during training."""
        pass # Implementation needed