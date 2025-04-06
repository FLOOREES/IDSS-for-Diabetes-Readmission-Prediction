# src/modeling/decoder.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any
import logging

# Assuming attention module is defined
from .attention import AdditiveAttention

logger = logging.getLogger(__name__)

class DecoderRNN(nn.Module):
    """ LSTM/GRU Decoder, potentially using Attention, for sequence reconstruction. """
    def __init__(
        self,
        reconstruction_dim: int, # Should match the encoder's RNN input dim
        encoder_hidden_dim: int,
        decoder_hidden_dim: int, # Can be same as encoder or different
        n_layers: int = 1,
        dropout: float = 0.1,
        use_gru: bool = False,
        use_attention: bool = True,
        attention_dim: Optional[int] = None # Changed from 'attention' instance
    ):
        super().__init__()
        self.reconstruction_dim = reconstruction_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_attention = use_attention

        RNN = nn.GRU if use_gru else nn.LSTM

        # Define RNN input dimension based on whether attention is used AFTER RNN
        # If we pass encoder_outputs directly to RNN first
        rnn_input_dim = encoder_hidden_dim # Decoder RNN processes encoder's output sequence
        self.rnn = RNN(rnn_input_dim, decoder_hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        if use_attention:
            if not attention_dim: raise ValueError("attention_dim must be provided if use_attention=True")
            # Attention compares decoder hidden state with encoder outputs
            self.attention = AdditiveAttention(encoder_hidden_dim, decoder_hidden_dim, attention_dim)
            # Output layer input combines decoder hidden state and attention context
            fc_input_dim = decoder_hidden_dim + encoder_hidden_dim
        else:
            self.attention = None
            # Output layer just takes decoder hidden state
            fc_input_dim = decoder_hidden_dim

        self.fc_out = nn.Linear(fc_input_dim, reconstruction_dim)


    def forward(
        self,
        encoder_outputs: torch.Tensor, # (batch, seq_len, enc_hidden_dim)
        encoder_final_hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], # (n_layers, batch, enc_hidden_dim)
        mask: Optional[torch.Tensor] = None # (batch, seq_len)
    ) -> torch.Tensor:
        """
        Forward pass for AE decoder. Applies RNN then optionally Attention + Projection.
        """
        batch_size, seq_len, _ = encoder_outputs.shape

        # 1. Pass encoder outputs through decoder RNN, initialized by final encoder state
        # The initial hidden state for the decoder RNN is the final hidden state of the encoder.
        decoder_outputs, _ = self.rnn(encoder_outputs, encoder_final_hidden)
        # decoder_outputs shape: (batch, seq_len, decoder_hidden_dim)

        if self.use_attention and self.attention is not None:
            # 2. Apply Attention
            # We need a query for each timestep. Use the decoder_outputs as queries.
            # This requires iterating or implementing differently.
            # Simplification: Use the *final* encoder hidden state as a single query
            # to get one context vector, and combine that with *all* decoder outputs.

            # Extract final layer's hidden state from encoder to use as query
            if isinstance(encoder_final_hidden, tuple): # LSTM
                # Use hidden state h, not cell state c
                query = encoder_final_hidden[0][-1] # Shape: (batch, encoder_hidden_dim)
            else: # GRU
                query = encoder_final_hidden[-1] # Shape: (batch, encoder_hidden_dim)
            # Note: Query dim must match decoder_hidden_dim expected by AdditiveAttention
            # If encoder_hidden_dim != decoder_hidden_dim, need projection or adjust attention.
            # Assuming here encoder_hidden_dim == decoder_hidden_dim for simplicity.
            if query.shape[-1] != self.decoder_hidden_dim:
                 logger.warning(f"Query dim ({query.shape[-1]}) doesn't match decoder hidden dim ({self.decoder_hidden_dim}) for attention. Check dims.")
                 # Fallback or add projection if needed. For now, proceed assuming they match.


            # Calculate context vector using the single query over all encoder outputs
            context, attn_weights = self.attention(query, encoder_outputs, mask)
            # context shape: (batch, encoder_hidden_dim)

            # 3. Combine and Project
            # Repeat context vector for each timestep
            context_repeated = context.unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq_len, encoder_hidden_dim)

            # Concatenate decoder outputs with the repeated context
            combined = torch.cat((decoder_outputs, context_repeated), dim=-1)
            # combined shape: (batch, seq_len, decoder_hidden_dim + encoder_hidden_dim)

            reconstructions = self.fc_out(self.dropout(combined))
            # reconstructions shape: (batch, seq_len, reconstruction_dim)

        else:
            # No attention, just project decoder RNN outputs
            reconstructions = self.fc_out(self.dropout(decoder_outputs))
            # reconstructions shape: (batch, seq_len, reconstruction_dim)

        return reconstructions