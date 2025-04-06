# src/modeling/encoder.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any

# Assuming attention and embedding modules are defined elsewhere
from .attention import AdditiveAttention
from .embeddings import EmbeddingManager

class EncoderRNN(nn.Module):
    """ LSTM/GRU Encoder with optional Attention and Embedding management. """
    def __init__(
        self,
        num_ohe_features: int, # Dimension of numerical + OHE features
        embedding_manager: EmbeddingManager,
        hidden_dim: int,
        n_layers: int = 1,
        dropout: float = 0.1,
        use_gru: bool = False,
        use_attention: bool = True, # Note: Attention is typically used in Decoder or post-Encoder
        attention_dim: Optional[int] = None # Needed if using attention here (less common)
    ):
        super().__init__()
        self.embedding_manager = embedding_manager
        self.total_embedding_dim = embedding_manager.get_total_embedding_dim()
        self.input_dim = num_ohe_features + self.total_embedding_dim # Final input dim to RNN
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_attention = use_attention # Store for reference, though maybe not used directly here

        RNN = nn.GRU if use_gru else nn.LSTM
        self.rnn = RNN(self.input_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Attention is usually applied *by* the decoder *to* the encoder outputs,
        # or in self-attention scenarios (Transformers). Not typically part of standard RNN encoder.
        # If needed here for some reason, initialize it:
        # if use_attention and attention_dim:
        #     self.attention = AdditiveAttention(hidden_dim, hidden_dim, attention_dim) # Example config
        # else:
        #     self.attention = None

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the encoder.

        Args:
            batch: Dictionary from the collate function containing feature tensors/labels.

        Returns:
            encoder_outputs: Hidden states for all timesteps (batch, seq_len, hidden_dim).
            final_hidden_state: Final hidden state (and cell state for LSTM)
                                (n_layers, batch, hidden_dim).
        """
        # 1. Get embeddings
        embeddings = self.embedding_manager(batch) # (batch, seq_len, total_emb_dim)

        # 2. Get numerical/OHE features
        num_ohe = batch['num_ohe'].to(embeddings.device) # (batch, seq_len, num_ohe_dim)

        # 3. Concatenate features
        rnn_input = torch.cat((num_ohe, embeddings), dim=-1) # (batch, seq_len, input_dim)
        rnn_input = self.dropout(rnn_input)

        # 4. Pass through RNN
        # Handle potential packing if lengths are provided and needed
        # lengths = batch.get('lengths')
        # if lengths is not None: packed_input = nn.utils.rnn.pack_padded_sequence(...) etc.
        encoder_outputs, final_hidden_state = self.rnn(rnn_input)
        # if lengths is not None: encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(...)

        # 5. Optional: Apply attention (less common for standard encoder)
        # if self.attention: ...

        return encoder_outputs, final_hidden_state