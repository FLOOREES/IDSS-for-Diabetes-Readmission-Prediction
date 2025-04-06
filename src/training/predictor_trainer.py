# src/training/predictor_trainer.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base_trainer import BaseTrainer
from modeling.predictor import PredictorModel # Adjust import

class PredictorTrainer(BaseTrainer):
    """ Trainer specifically for the Sequence Prediction model. """
    def __init__(self, model: PredictorModel, criterion_name: str = 'nll', **kwargs): # Pass args to BaseTrainer
        # Setup criterion based on name (e.g., NLL for LogSoftmax, BCE for Sigmoid)
        if criterion_name.lower() == 'nll':
            criterion = nn.NLLLoss(reduction='none') # Use with LogSoftmax output
        elif criterion_name.lower() == 'bce':
             criterion = nn.BCEWithLogitsLoss(reduction='none') # Use with raw logits output
        else:
             raise ValueError(f"Unsupported criterion: {criterion_name}")

        super().__init__(model=model, criterion=criterion, model_name="predictor", **kwargs)
        self.criterion_name = criterion_name

    def _calculate_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """ Calculates masked prediction loss (e.g., Cross Entropy). """
        mask = batch['mask'].to(self.device) # (batch, seq_len)
        targets = batch['targets'].to(self.device) # (batch, seq_len)

        # Outputs are likely logits (batch, seq_len, num_classes) or log_probs
        # Targets are labels (batch, seq_len)

        # Reshape for standard loss functions if needed:
        # E.g., for NLLLoss: (N, C), Target: (N) where N = batch * seq_len
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes) # Shape: (batch*seq_len, num_classes)
        targets_flat = targets.view(-1) # Shape: (batch*seq_len)
        mask_flat = mask.view(-1) # Shape: (batch*seq_len)

        # Calculate element-wise loss
        unmasked_loss_flat = self.criterion(outputs_flat, targets_flat) # Shape: (batch*seq_len)

        # Apply mask
        masked_loss_flat = unmasked_loss_flat * mask_flat.float()

        # Average over non-masked elements
        num_valid_elements = mask_flat.sum()
        if num_valid_elements > 0:
            loss = masked_loss_flat.sum() / num_valid_elements
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss

    def configure_optimizers(self, finetune_encoder: bool, encoder_lr_factor: float = 0.1):
        """ Allows setting different LRs for encoder vs head during fine-tuning. """
        pass # Implementation needed if different LRs are desired

    # Override train/val epochs if optimizer configuration needs handling