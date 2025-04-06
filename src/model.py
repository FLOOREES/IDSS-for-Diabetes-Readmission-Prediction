# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union

class EncoderRNN(nn.Module):
    """LSTM or GRU Encoder"""
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 1, dropout: float = 0.1, use_gru: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        RNN = nn.GRU if use_gru else nn.LSTM
        self.rnn = RNN(input_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            x_lengths: Original lengths for packing (optional, requires sorting)

        Returns:
            outputs: Hidden states for all timesteps (batch_size, seq_len, hidden_dim)
            hidden: Final hidden state (and cell state for LSTM)
                    (n_layers, batch_size, hidden_dim)
        """
        x = self.dropout(x)

        # Packing improves efficiency if sequences have varied lengths significantly
        # Note: Requires sequences to be sorted by length in descending order within the batch
        # if x_lengths is not None:
        #     x = nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu(), batch_first=True, enforce_sorted=False) # Set enforce_sorted=True if pre-sorted

        outputs, hidden = self.rnn(x) # hidden is (h_n, c_n) for LSTM, h_n for GRU

        # If packed:
        # if x_lengths is not None:
        #     outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden

class AdditiveAttention(nn.Module):
    """ Additive (Bahdanau-style) Attention """
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_hidden_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False) # Attention scoring vector

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            decoder_hidden: Previous decoder hidden state (batch_size, decoder_hidden_dim)
                            -> Needs to be unsqueezed if it's just the last layer's state
            encoder_outputs: Encoder hidden states (batch_size, seq_len, encoder_hidden_dim)
            mask: Attention mask (batch_size, seq_len) - True for valid steps, False for padding

        Returns:
            context_vector: Weighted sum of encoder outputs (batch_size, encoder_hidden_dim)
            attention_weights: Weights applied (batch_size, seq_len)
        """
        # decoder_hidden shape: (batch_size, dec_hid_dim) -> (batch_size, 1, attn_dim)
        # encoder_outputs shape: (batch_size, seq_len, enc_hid_dim) -> (batch_size, seq_len, attn_dim)
        seq_len = encoder_outputs.shape[1]
        decoder_hidden_unsqueezed = decoder_hidden.unsqueeze(1) # (batch_size, 1, dec_hid_dim) - Assuming last layer state

        # Project encoder and decoder states to attention dimension
        proj_enc = self.encoder_proj(encoder_outputs) # (batch_size, seq_len, attn_dim)
        proj_dec = self.decoder_proj(decoder_hidden_unsqueezed) # (batch_size, 1, attn_dim)

        # Calculate energy (scores)
        # Use broadcasting for addition: (batch, seq_len, attn) + (batch, 1, attn) -> (batch, seq_len, attn)
        energy = torch.tanh(proj_enc + proj_dec) # (batch_size, seq_len, attn_dim)

        # Calculate attention scores
        attention_scores = self.v(energy).squeeze(2) # (batch_size, seq_len)

        # Apply mask (set scores for padded steps to -inf before softmax)
        if mask is not None:
            # Ensure mask has same shape or is broadcastable
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10) # Mask == 0 means padding

        # Calculate attention weights (probabilities)
        attention_weights = F.softmax(attention_scores, dim=1) # (batch_size, seq_len)

        # Calculate context vector (weighted sum)
        # Unsqueeze weights for batch matrix multiplication: (batch, 1, seq_len) bmm (batch, seq_len, enc_hid) -> (batch, 1, enc_hid)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # (batch_size, 1, encoder_hidden_dim)
        context_vector = context_vector.squeeze(1) # (batch_size, encoder_hidden_dim)

        return context_vector, attention_weights


class DecoderRNN(nn.Module):
    """LSTM or GRU Decoder with Attention"""
    def __init__(self, output_dim: int, hidden_dim: int, encoder_hidden_dim: int, attention: nn.Module, n_layers: int = 1, dropout: float = 0.1, use_gru: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.n_layers = n_layers
        self.attention = attention
        RNN = nn.GRU if use_gru else nn.LSTM

        # Decoder RNN input: Concatenation of previous output feature vector (or embedding) and context vector
        # In AE, we can simplify: Feed context + simplified input (like encoder hidden state) or just context
        # Let's try context + encoder state representation
        self.rnn = RNN(encoder_hidden_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim) # To reconstruct original features
        self.dropout = nn.Dropout(dropout)

    def forward(self, # Simplified AE version, not step-by-step generation
                encoder_final_hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                encoder_outputs: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Simplified forward pass for AE reconstruction using attention on all encoder outputs.
        Args:
            encoder_final_hidden: Final hidden state from encoder (layers, batch, hid_dim)
                                 (or tuple (h,c) for LSTM)
            encoder_outputs: All hidden states from encoder (batch, seq_len, enc_hid_dim)
            attention_mask: Mask for attention (batch, seq_len)

        Returns:
            reconstructions: Reconstructed sequences (batch, seq_len, output_dim)
        """
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        # Use the final layer's hidden state from the encoder to query attention
        # If LSTM, use hidden state h, not cell state c
        if isinstance(encoder_final_hidden, tuple): # LSTM
            decoder_hidden_query = encoder_final_hidden[0][-1] # (batch, hid_dim) - Last layer
        else: # GRU
            decoder_hidden_query = encoder_final_hidden[-1] # (batch, hid_dim) - Last layer


        # --- Attention across all encoder outputs ---
        # Calculate context vector for each decoder step (here simplified: use one context derived from final state)
        # For a true step-by-step decoder, this would be inside a loop.
        # Let's adapt the Keras idea: Pass encoder_outputs through decoder RNN and THEN apply attention

        # 1. Pass encoder outputs through decoder RNN (initialized with encoder state)
        decoder_outputs, _ = self.rnn(encoder_outputs, encoder_final_hidden) # (batch, seq_len, dec_hid_dim)

        # 2. Apply Attention (using decoder outputs as query, encoder outputs as key/value)
        # This requires iterating or adapting attention layer
        # Let's try Keras' simpler approach: Attention uses encoder outputs and final encoder state,
        # then combine context with decoder outputs (which used encoder outputs as input).

        # Alternative more aligned with Keras example structure:
        # Assume decoder_hidden_query is representative for the whole sequence context in AE
        context_vector, _ = self.attention(decoder_hidden_query, encoder_outputs, attention_mask)
        # context_vector shape: (batch_size, encoder_hidden_dim)

        # Repeat context vector for each timestep to combine with decoder outputs
        context_vector_repeated = context_vector.unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq_len, enc_hid_dim)

        # Combine decoder outputs (from RNN) and the attention context
        # Example: Concatenate or Add
        # combined = torch.cat((decoder_outputs, context_vector_repeated), dim=2)
        # combined_dim = self.hidden_dim + self.encoder_hidden_dim
        # Using decoder_outputs directly might be simpler first
        combined = self.dropout(decoder_outputs) # (batch, seq_len, dec_hid_dim)
        combined_dim = self.hidden_dim

        # Final layer to reconstruct features
        # Apply Linear layer to each timestep
        reconstructions = self.fc_out(combined) # (batch, seq_len, output_dim)

        return reconstructions


class Seq2SeqAE(nn.Module):
    """Sequence-to-Sequence Autoencoder Model"""
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the autoencoder.
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            attention_mask: Mask for attention (batch, seq_len)

        Returns:
            reconstructions: Reconstructed sequences (batch, seq_len, output_dim)
        """
        encoder_outputs, encoder_final_hidden = self.encoder(x)
        reconstructions = self.decoder(encoder_final_hidden, encoder_outputs, attention_mask)
        return reconstructions

    def get_encoder(self):
        """Returns the encoder module."""
        return self.encoder

    def get_decoder(self):
         """Returns the decoder module."""
         return self.decoder