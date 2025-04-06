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
from tqdm import tqdm # Add tqdm for progress bars

# Assuming utils.helpers exists for save/load_artifact
from utils.helpers import save_artifact, load_artifact

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
        self.model_name = model_name # Used for checkpoint filenames
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_value = gradient_clip_value
        self.logger = logger or logging.getLogger(__name__)

        self.history: Dict[str, list] = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
        self.latest_model_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_latest.pth")


    def _setup_optimizer(self, name: str, params: Dict[str, Any]) -> optim.Optimizer:
        """Initializes the optimizer."""
        name_lower = name.lower()
        if name_lower == 'adam':
            return optim.Adam(self.model.parameters(), **params)
        elif name_lower == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), **params)
        elif name_lower == 'adamw':
             return optim.AdamW(self.model.parameters(), **params)
        # Add other optimizers as needed
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def _setup_scheduler(self, name: Optional[str], params: Optional[Dict[str, Any]]) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Initializes the learning rate scheduler."""
        if name is None:
            return None
        name_lower = name.lower()
        if name_lower == 'reducelronplateau':
            if params is None: params = {} # Default params if none provided
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)
        elif name_lower == 'steplr':
            if params is None: raise ValueError("StepLR requires parameters (step_size).")
            return optim.lr_scheduler.StepLR(self.optimizer, **params)
        # Add other schedulers as needed
        else:
            self.logger.warning(f"Unsupported scheduler: {name}. No scheduler will be used.")
            return None

    @abstractmethod
    def _calculate_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Calculates the loss for a batch, handling masking."""
        pass

    def _train_epoch(self) -> float:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            # Move all tensor values in batch dict to device
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'features'}
             # Special handling for features if it's a list of dicts (collate fn structure)
            if 'features' in batch: batch_device['features'] = batch['features'] # Keep original structure
            # Or adapt based on your actual collate_fn output structure

            self.optimizer.zero_grad()
            outputs = self.model(batch_device) # Pass the whole batch dict
            loss = self._calculate_loss(outputs, batch_device)

            if torch.isnan(loss):
                self.logger.warning("NaN loss detected during training. Skipping batch.")
                continue # Skip backprop for this batch

            loss.backward()

            if self.gradient_clip_value:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

            self.optimizer.step()

            # Use 'lengths' sum for accurate sample count if available and loss is averaged per sequence
            # Or use mask sum if loss is averaged per timestep
            batch_size = batch['mask'].shape[0] # Get batch size reliably
            total_loss += loss.item() * batch_size # Accumulate total loss for epoch average
            total_samples += batch_size
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")


        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss

    def _val_epoch(self) -> float:
        """Runs one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'features'}
                if 'features' in batch: batch_device['features'] = batch['features']

                outputs = self.model(batch_device)
                loss = self._calculate_loss(outputs, batch_device)

                if torch.isnan(loss):
                    self.logger.warning("NaN loss detected during validation.")
                    # Potentially skip or assign high loss? For average, skipping might be okay.
                    continue

                batch_size = batch['mask'].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")


        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """Saves model checkpoint."""
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save the latest checkpoint
        save_artifact(state, self.latest_model_path)
        self.logger.debug(f"Latest checkpoint saved to {self.latest_model_path}")

        # Save the best checkpoint separately
        if is_best:
            save_artifact(state, self.best_model_path)
            self.logger.debug(f"Best checkpoint saved to {self.best_model_path}")

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

            lr = self.optimizer.param_groups[0]['lr'] # Get current learning rate
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.1e} | Duration: {epoch_duration:.2f}s")

            # Checkpoint Saving
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

            # LR Scheduling
            if self.scheduler:
                 if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                     self.scheduler.step(val_loss) # Step on validation loss
                 # Add other scheduler step logic if needed (e.g., self.scheduler.step() for StepLR)

            # Early Stopping Check
            if self.early_stopping_patience and self.epochs_no_improve >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after epoch {epoch + 1}.")
                break

        total_duration = time.time() - start_time
        self.logger.info(f"Training finished in {total_duration:.2f}s. Best Validation Loss: {self.best_val_loss:.4f}")
        # Load best model weights at the end
        self._load_best_checkpoint()

    def _load_best_checkpoint(self):
        """Loads the best model checkpoint found during training."""
        if os.path.exists(self.best_model_path):
            try:
                self.logger.info(f"Loading best model checkpoint from {self.best_model_path}")
                checkpoint = load_artifact(self.best_model_path, device=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # Optionally load optimizer/scheduler state if needed for resuming
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # if self.scheduler and 'scheduler_state_dict' in checkpoint:
                #    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Best model weights loaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to load best checkpoint: {e}")
        else:
            self.logger.warning("Best checkpoint file not found. Model retains weights from last epoch.")