# src/modeling/prediction_head.py
import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    """ Simple linear head for sequence prediction. """
    def __init__(self, input_dim: int, output_dim: int = 3): # Default changed for clarity, but value comes from call
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # Activation like Softmax is typically applied *after* the head
        # or incorporated into the loss function (like nn.CrossEntropyLoss)

    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_outputs: Shape (batch, seq_len, input_dim).
        Returns:
            logits: Shape (batch, seq_len, output_dim). Raw scores for each class.
        """
        logits = self.fc(encoder_outputs)
        return logits