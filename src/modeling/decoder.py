# src/modeling/decoder.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any

# Assuming attention module is defined
from .attention import AdditiveAttention

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
        attention_dim: Optional[int] = None
    ):
        super().__init__()
        self.reconstruction_dim = reconstruction_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_attention = use_attention

        RNN = nn.GRU if use_gru else nn.LSTM
        # Input to decoder RNN might depend on attention strategy
        # If using attention context vector: encoder_hidden_dim + some_input
        # If passing encoder outputs through decoder: decoder_hidden_dim
        # Let's align with a common pattern: use attention context + previous hidden state (simplified here)
        rnn_input_dim = encoder_hidden_dim # Input is primarily attention context in simpler AEs

        self.rnn = RNN(rnn_input_dim, decoder_hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if use_attention:
            if not attention_dim: raise ValueError("attention_dim must be provided if use_attention=True")
            self.attention = AdditiveAttention(encoder_hidden_dim, decoder_hidden_dim, attention_dim)
            # Output layer input combines decoder hidden state and attention context
            self.fc_out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim, reconstruction_dim)
        else:
            self.attention = None
            # Output layer just takes decoder hidden state
            self.fc_out = nn.Linear(decoder_hidden_dim, reconstruction_dim)


    def forward(
        self,
        encoder_outputs: torch.Tensor, # (batch, seq_len, enc_hidden_dim)
        encoder_final_hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], # (n_layers, batch, enc_hidden_dim)
        mask: Optional[torch.Tensor] = None # (batch, seq_len)
        # teacher_forcing_ratio: float = 0.0 # Not typically used in basic AE reconstruction
    ) -> torch.Tensor:
        """
        Forward pass for AE decoder. Simplified: processes entire sequence context at once.

        Args:
            encoder_outputs: Outputs from all encoder timesteps.
            encoder_final_hidden: Final hidden state tuple (h, c) or tensor h from encoder.
            mask: Padding mask.

        Returns:
            reconstructions: Tensor of reconstructed features (batch, seq_len, reconstruction_dim).
        """
        # Simplified AE decoder: Often uses encoder_outputs directly or applies attention globally.
        # A more complex seq2seq would iterate timestep by timestep.

        # Strategy 1: Pass encoder outputs through decoder RNN
        # Initialize decoder RNN with final encoder state
        decoder_outputs, _ = self.rnn(encoder_outputs, encoder_final_hidden) # Assumes rnn input dim matches encoder_outputs dim

        if self.use_attention:
            # Apply attention at each step (conceptually)
            # This requires adapting the attention mechanism or using a simpler global context
            batch_size, seq_len, _ = encoder_outputs.shape
            reconstructions = []

            # Get initial decoder hidden state (last layer of encoder's final state)
            if isinstance(encoder_final_hidden, tuple): # LSTM
                dec_hidden = encoder_final_hidden[0][-1] # (batch, dec_hidden_dim)
            else: # GRU
                dec_hidden = encoder_final_hidden[-1] # (batch, dec_hidden_dim)

            for t in range(seq_len):
                # Use previous decoder hidden state to query attention
                context, attn_weights = self.attention(dec_hidden, encoder_outputs, mask) # (batch, enc_hidden_dim), (batch, seq_len)

                # Combine context with decoder output for this step
                # Here, using decoder_outputs directly calculated earlier is simpler for AE
                step_output = decoder_outputs[:, t, :] # (batch, dec_hidden_dim)
                combined_output = torch.cat((step_output, context), dim=-1) # (batch, dec_hidden + enc_hidden)
                step_reconstruction = self.fc_out(self.dropout(combined_output)) # (batch, recon_dim)
                reconstructions.append(step_reconstruction)

                # Update decoder hidden state (simplified - not running RNN step-by-step here)
                # In a real step-by-step decoder, you'd update dec_hidden = rnn_output_hidden

            output_tensor = torch.stack(reconstructions, dim=1) # (batch, seq_len, recon_dim)

        else:
            # No attention, just project decoder outputs
            output_tensor = self.fc_out(self.dropout(decoder_outputs)) # (batch, seq_len, recon_dim)

        return output_tensor