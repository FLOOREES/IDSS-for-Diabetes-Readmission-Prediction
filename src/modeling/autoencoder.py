# src/modeling/autoencoder.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .encoder import EncoderRNN
from .decoder import DecoderRNN

class Seq2SeqAE(nn.Module):
    """ Combines Encoder and Decoder for Autoencoding. """
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Full autoencoder forward pass.

        Args:
            batch: Dictionary from collate function containing features and mask.

        Returns:
            reconstructions: Output from the decoder.
        """
        mask = batch.get('mask')
        encoder_outputs, encoder_final_hidden = self.encoder(batch)
        reconstructions = self.decoder(encoder_outputs, encoder_final_hidden, mask)
        return reconstructions

    def get_encoder(self) -> EncoderRNN:
        return self.encoder