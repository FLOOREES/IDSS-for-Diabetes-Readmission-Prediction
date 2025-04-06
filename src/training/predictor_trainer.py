# src/training/predictor_trainer.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
# Adjust import based on actual structure - assuming modeling is sibling to training
from modeling.predictor import PredictorModel
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class PredictorTrainer(BaseTrainer):
    """ Trainer specifically for the Sequence Prediction model. """
    def __init__(self, model: PredictorModel, criterion_name: str = 'nll', **kwargs):
        if criterion_name.lower() == 'nll':
            criterion = nn.NLLLoss(reduction='none')
        elif criterion_name.lower() == 'bce':
             # Expects raw logits from model, target should be float
             criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
             raise ValueError(f"Unsupported criterion: {criterion_name}")

        super().__init__(model=model, criterion=criterion, model_name="predictor", **kwargs)
        self.criterion_name = criterion_name

    def _calculate_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """ Calculates masked prediction loss (e.g., BCEWithLogitsLoss). """
        mask = batch['mask'].to(self.device) # (batch, seq_len), bool
        targets = batch['targets'].to(self.device) # (batch, seq_len), long

        # Outputs are logits (batch, seq_len, num_classes) -> num_classes is 1 for BCE
        batch_size, seq_len, num_classes = outputs.shape

        # Flatten outputs and targets
        outputs_flat = outputs.view(-1, num_classes) # Shape: (batch*seq_len, 1)
        targets_flat = targets.view(-1) # Shape: (batch*seq_len)
        mask_flat = mask.view(-1) # Shape: (batch*seq_len)

        # --- Fix the shape and type mismatch for BCEWithLogitsLoss ---
        if self.criterion_name.lower() == 'bce':
            # Target needs to be float and have the same shape as output
            targets_flat = targets_flat.float().unsqueeze(1) # Shape: (batch*seq_len, 1)
            # Verify shape match
            if outputs_flat.shape != targets_flat.shape:
                 logger.error(f"Shape mismatch AFTER fix! Output: {outputs_flat.shape}, Target: {targets_flat.shape}")
                 # Raise error or handle differently
                 raise ValueError("Target and output shapes still mismatch after fix.")
        # --- End Fix ---
        # For NLLLoss, targets should remain long and outputs should be log_probs (usually involves LogSoftmax)
        # Ensure model output and criterion match (e.g., don't use NLLLoss with raw logits)

        # Calculate element-wise loss
        unmasked_loss_flat = self.criterion(outputs_flat, targets_flat) # Shape should now be (batch*seq_len, 1) for BCE

        # If loss output has extra dimension (like from BCE), squeeze it before masking
        if unmasked_loss_flat.ndim > mask_flat.ndim:
             unmasked_loss_flat = unmasked_loss_flat.squeeze(-1) # Shape: (batch*seq_len)

        # Apply mask
        masked_loss_flat = unmasked_loss_flat * mask_flat.float() # Use float mask

        # Average over non-masked elements
        num_valid_elements = mask_flat.sum()
        if num_valid_elements > 0:
            loss = masked_loss_flat.sum() / num_valid_elements
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Ensure requires_grad if needed downstream

        return loss

    def configure_optimizers(self, finetune_encoder: bool, encoder_lr_factor: float = 0.1):
        """ Allows setting different LRs for encoder vs head during fine-tuning. """
        if not finetune_encoder:
             self.logger.info("Optimizer configured for predictor head only.")
             self.optimizer = self._setup_optimizer(self.optimizer_name, {'params': self.model.head.parameters(), **self.optimizer_params})
        else:
            self.logger.info(f"Optimizer configured for fine-tuning encoder (LR factor: {encoder_lr_factor}) and predictor head.")
            # Example: Group parameters
            self.optimizer = self._setup_optimizer(
                self.optimizer_name,
                [
                    {'params': self.model.encoder.parameters(), 'lr': self.optimizer_params.get('lr', 1e-3) * encoder_lr_factor},
                    {'params': self.model.head.parameters()} # Use default LR from optimizer_params for head
                ]
                # Need to adjust how _setup_optimizer handles list of param groups vs single dict
                # Or modify self.optimizer_params before calling _setup_optimizer
            )
            self.logger.warning("Parameter group optimizer setup needs careful implementation in _setup_optimizer.")
            # Fallback to single optimizer if group setup is complex
            # self.optimizer = self._setup_optimizer(self.optimizer_name, {'params': self.model.parameters(), **self.optimizer_params})


    # Override train/val epochs if optimizer configuration needs handling
    # Base class train/val epoch methods should work fine otherwise.