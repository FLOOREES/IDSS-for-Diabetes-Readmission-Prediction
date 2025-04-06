# src/modeling/embeddings.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingManager(nn.Module):
    """
    Manages learned and pre-computed embeddings for sequence models.
    """
    def __init__(
        self,
        learned_emb_config: Dict[str, Tuple[int, int]], # col_name -> (vocab_size, emb_dim)
        precomputed_emb_config: Dict[str, Tuple[str, bool]], # col_name -> (npy_path, finetune_flag)
        device: torch.device
    ):
        super().__init__()
        self.learned_emb_layers = nn.ModuleDict()
        self.precomputed_emb_layers = nn.ModuleDict()
        self.total_learned_emb_dim = 0
        self.total_precomputed_emb_dim = 0
        self.device = device

        # Initialize learned embeddings
        for col, (vocab_size, emb_dim) in learned_emb_config.items():
            self.learned_emb_layers[col] = nn.Embedding(vocab_size, emb_dim, padding_idx=0) # Assume 0 is padding index
            self.total_learned_emb_dim += emb_dim
            logger.info(f"Initialized learned embedding for '{col}' (Vocab: {vocab_size}, Dim: {emb_dim})")

        # Initialize pre-computed embeddings
        for col, (npy_path, finetune) in precomputed_emb_config.items():
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"Precomputed embedding file not found: {npy_path}")
            weights = torch.from_numpy(np.load(npy_path)).float()
            num_embeddings, emb_dim = weights.shape
            embedding_layer = nn.Embedding(num_embeddings, emb_dim, padding_idx=0) # Assume 0 is padding
            embedding_layer.weight.data.copy_(weights)
            embedding_layer.weight.requires_grad = finetune
            self.precomputed_emb_layers[col] = embedding_layer
            self.total_precomputed_emb_dim += emb_dim
            logger.info(f"Initialized precomputed embedding for '{col}' (Shape: {weights.shape}, Finetune: {finetune})")

        logger.info(f"Total Learned Embedding Dim: {self.total_learned_emb_dim}")
        logger.info(f"Total Precomputed Embedding Dim: {self.total_precomputed_emb_dim}")

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Looks up embeddings based on labels in the batch and concatenates them.

        Args:
            batch: A dictionary containing label tensors, e.g.,
                   batch['learned_labels'][col_name] and batch['precomputed_labels'][col_name].

        Returns:
            Concatenated embeddings for each timestep (batch, seq_len, total_emb_dim).
        """
        all_embeddings = []

        # Process learned embeddings
        if 'learned_labels' in batch:
            for col_name, layer in self.learned_emb_layers.items():
                if col_name in batch['learned_labels']:
                    labels = batch['learned_labels'][col_name].to(self.device)
                    all_embeddings.append(layer(labels))
                else:
                    logger.warning(f"Labels for learned embedding '{col_name}' not found in batch.")

        # Process pre-computed embeddings
        if 'precomputed_labels' in batch:
            for col_name, layer in self.precomputed_emb_layers.items():
                if col_name in batch['precomputed_labels']:
                    labels = batch['precomputed_labels'][col_name].to(self.device)
                     # Clamp labels to be within valid range [0, num_embeddings-1]
                    num_embeddings = layer.num_embeddings
                    labels_clamped = torch.clamp(labels, 0, num_embeddings - 1)
                    if (labels != labels_clamped).any():
                        logger.warning(f"Labels for precomputed embedding '{col_name}' were out of bounds and clamped.")
                    all_embeddings.append(layer(labels_clamped))
                else:
                    logger.warning(f"Labels for precomputed embedding '{col_name}' not found in batch.")

        if not all_embeddings:
            # Handle case with no embedding features
             batch_size = batch.get("mask", torch.empty(0)).shape[0] # Get batch size from mask or elsewhere
             seq_len = batch.get("mask", torch.empty(0,0)).shape[1]
             return torch.empty(batch_size, seq_len, 0, device=self.device) # Return shape (batch, seq, 0)


        # Concatenate along the feature dimension
        concatenated_embeddings = torch.cat(all_embeddings, dim=-1)
        return concatenated_embeddings

    def get_total_embedding_dim(self) -> int:
        """Returns the total dimension of all concatenated embeddings."""
        return self.total_learned_emb_dim + self.total_precomputed_emb_dim