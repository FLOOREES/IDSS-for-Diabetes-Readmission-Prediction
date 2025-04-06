# src/modeling/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AdditiveAttention(nn.Module):
    """ Additive (Bahdanau-style) Attention layer. """
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int):
        super().__init__()
        # Define layers (W_enc, W_dec, V)
        pass

    def forward(self, decoder_hidden_state: torch.Tensor, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden_state: Shape (batch, dec_hidden_dim) - typically last layer state.
            encoder_outputs: Shape (batch, seq_len, enc_hidden_dim).
            mask: Shape (batch, seq_len) - True for valid tokens, False for padding.

        Returns:
            context_vector: Shape (batch, enc_hidden_dim).
            attention_weights: Shape (batch, seq_len).
        """
        pass