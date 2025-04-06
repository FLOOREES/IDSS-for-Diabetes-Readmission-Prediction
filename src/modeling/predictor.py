# src/modeling/predictor.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .encoder import EncoderRNN
from .prediction_head import PredictionHead

class PredictorModel(nn.Module):
    """ Combines Encoder and Prediction Head for sequence prediction. """
    def __init__(self, encoder: EncoderRNN, head: PredictionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Prediction forward pass.

        Args:
            batch: Dictionary from collate function containing features.

        Returns:
            logits: Output from the prediction head for each timestep.
        """
        encoder_outputs, _ = self.encoder(batch) # Don't need final hidden state here
        logits = self.head(encoder_outputs)
        return logits

    def get_encoder(self) -> EncoderRNN:
        return self.encoder