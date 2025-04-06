# src/modeling/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AdditiveAttention(nn.Module):
    """ Additive (Bahdanau-style) Attention layer. """
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_hidden_dim, attention_dim, bias=False) # W_enc * h_enc
        self.decoder_proj = nn.Linear(decoder_hidden_dim, attention_dim, bias=False) # W_dec * h_dec
        self.v = nn.Linear(attention_dim, 1, bias=False) # V * tanh(...)

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
        batch_size, seq_len, _ = encoder_outputs.shape

        # Project encoder outputs: (batch, seq_len, enc_hidden_dim) -> (batch, seq_len, attention_dim)
        proj_enc = self.encoder_proj(encoder_outputs)

        # Project decoder state: (batch, dec_hidden_dim) -> (batch, 1, attention_dim)
        proj_dec = self.decoder_proj(decoder_hidden_state).unsqueeze(1)

        # Calculate energy score = V * tanh(W_enc*h_enc + W_dec*h_dec)
        # Use broadcasting for addition: (batch, seq_len, attn) + (batch, 1, attn) -> (batch, seq_len, attn)
        energy = torch.tanh(proj_enc + proj_dec) # (batch, seq_len, attention_dim)

        # Get attention scores: (batch, seq_len, attention_dim) -> (batch, seq_len, 1) -> (batch, seq_len)
        attention_scores = self.v(energy).squeeze(-1) # (batch, seq_len)

        # Apply mask: set scores for padded steps to negative infinity
        if mask is not None:
            # Ensure mask is boolean or float, shape (batch, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10) # Use large negative number

        # Calculate attention weights (probabilities)
        attention_weights = F.softmax(attention_scores, dim=1) # (batch, seq_len)

        # Calculate context vector (weighted sum of encoder outputs)
        # (batch, 1, seq_len) bmm (batch, seq_len, enc_hidden_dim) -> (batch, 1, enc_hidden_dim)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # -> (batch, enc_hidden_dim)
        context_vector = context_vector.squeeze(1)

        return context_vector, attention_weights