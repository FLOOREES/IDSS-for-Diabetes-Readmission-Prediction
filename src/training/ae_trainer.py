# src/training/ae_trainer.py
import torch
import torch.nn as nn
from typing import Dict, Any
from .base_trainer import BaseTrainer
from src.modeling.autoencoder import Seq2SeqAE # Adjust import based on actual structure

class AETrainer(BaseTrainer):
    """ Trainer specifically for the Sequence Autoencoder. """
    def __init__(self, model: Seq2SeqAE, **kwargs): # Pass args to BaseTrainer
        # Use MSE loss for reconstruction by default
        criterion = nn.MSELoss(reduction='none') # Calculate per element for masking
        super().__init__(model=model, criterion=criterion, model_name="autoencoder", **kwargs)

    def _calculate_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """ Calculates masked MSE loss for autoencoder reconstruction. """
        # Target is the original input feature vector (needs careful construction)
        # Assuming 'num_ohe' and concatenated embeddings form the input reconstructed
        mask = batch['mask'].to(self.device)
        embeddings = self.model.encoder.embedding_manager(batch) # Get embeddings again
        num_ohe = batch['num_ohe'].to(self.device)
        targets = torch.cat((num_ohe, embeddings), dim=-1) # Reconstruct this combined vector

        # Calculate element-wise loss
        unmasked_loss = self.criterion(outputs, targets) # Shape: (batch, seq_len, feature_dim)

        # Apply mask (expand mask to feature dim)
        mask_expanded = mask.unsqueeze(-1).float() # (batch, seq_len, 1)
        masked_loss = unmasked_loss * mask_expanded

        # Average over non-masked elements
        # Be careful about division by zero if mask sum is zero (empty batch/sequence?)
        num_valid_elements = mask_expanded.sum()
        if num_valid_elements > 0:
            loss = masked_loss.sum() / num_valid_elements
        else:
            loss = torch.tensor(0.0, device=self.device) # Or handle as error

        return loss

    # _train_epoch and _val_epoch in BaseTrainer can likely be used directly
    # if they call self._calculate_loss appropriately within the loop.
    # Override if specific AE logic is needed (e.g., logging reconstruction samples).