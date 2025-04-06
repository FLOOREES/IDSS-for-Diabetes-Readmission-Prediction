# src/modeling/prediction_head.py
import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    """ Simple linear head for sequence prediction. """
    def __init__(self, input_dim: int, output_dim: int = 1): # Default to 1 for binary sigmoid
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # Activation (like Sigmoid or LogSoftmax) is often applied *after* the head in the loss function

    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_outputs: Shape (batch, seq_len, input_dim == encoder_hidden_dim).

        Returns:
            logits: Shape (batch, seq_len, output_dim).
        """
        # Apply linear layer to each timestep's hidden state
        logits = self.fc(encoder_outputs)
        return logits