# src/data_preparation/collators.py
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any

def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to pad sequences within a batch and restructure features.

    Args:
        batch: A list of dictionaries, where each dictionary is the output
               of PatientSequenceDataset.__getitem__.

    Returns:
        A dictionary containing padded tensors for each feature type,
        padded targets, the mask, and original lengths.
    """
    # Determine max sequence length in this batch
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    max_len = torch.max(lengths).item()

    # --- Initialize lists to hold padded data for each feature type ---
    # Numerical + OHE
    num_ohe_sequences = []
    # Learned Embeddings (one list per configured column)
    learned_label_sequences = {}
    if batch and batch[0]['features']:
        for col_name in batch[0]['features'][0]['learned_labels'].keys():
             learned_label_sequences[col_name] = []
    # Precomputed Embeddings (one list per configured column)
    precomputed_label_sequences = {}
    if batch and batch[0]['features']:
        for col_name in batch[0]['features'][0]['precomputed_labels'].keys():
             precomputed_label_sequences[col_name] = []
    # Targets
    target_sequences = []

    # --- Iterate through the batch and collect data ---
    for item in batch:
        feature_seq = item['features'] # List of visit dicts
        target_seq = item['targets']   # Tensor
        seq_len = item['length']
        padding_len = max_len - seq_len

        # Pad numerical/OHE features
        num_ohe_data = [visit['num_ohe'] for visit in feature_seq]
        if num_ohe_data:
            # Stack and pad
            padded_num_ohe = torch.cat(
                [torch.stack(num_ohe_data, dim=0)] + \
                [torch.zeros(padding_len, num_ohe_data[0].shape[0])] # Padding tensor
            , dim=0)
        else: # Handle empty sequence case
            padded_num_ohe = torch.zeros(max_len, 0) # Assuming 0 features if empty
        num_ohe_sequences.append(padded_num_ohe)


        # Pad learned embedding labels
        for col_name in learned_label_sequences.keys():
            labels = [visit['learned_labels'][col_name] for visit in feature_seq]
            padded_labels = torch.cat(
                [torch.tensor(labels, dtype=torch.long)] + \
                [torch.zeros(padding_len, dtype=torch.long)] # Use 0 for padding labels
            , dim=0)
            learned_label_sequences[col_name].append(padded_labels)

        # Pad precomputed embedding labels
        for col_name in precomputed_label_sequences.keys():
            labels = [visit['precomputed_labels'][col_name] for visit in feature_seq]
            padded_labels = torch.cat(
                [torch.tensor(labels, dtype=torch.long)] + \
                [torch.zeros(padding_len, dtype=torch.long)]
            , dim=0)
            precomputed_label_sequences[col_name].append(padded_labels)

        # Pad targets
        padded_targets = torch.cat([target_seq] + [torch.zeros(padding_len, dtype=torch.long)], dim=0) # Use 0 for padding targets
        target_sequences.append(padded_targets)

    # --- Stack sequences into batch tensors ---
    collated_batch = {}
    if num_ohe_sequences:
        collated_batch['num_ohe'] = torch.stack(num_ohe_sequences, dim=0) # (batch, seq_len, num_ohe_dim)

    collated_batch['learned_labels'] = {}
    for col_name, sequences in learned_label_sequences.items():
        collated_batch['learned_labels'][col_name] = torch.stack(sequences, dim=0) # (batch, seq_len)

    collated_batch['precomputed_labels'] = {}
    for col_name, sequences in precomputed_label_sequences.items():
        collated_batch['precomputed_labels'][col_name] = torch.stack(sequences, dim=0) # (batch, seq_len)

    collated_batch['targets'] = torch.stack(target_sequences, dim=0) # (batch, seq_len)
    collated_batch['lengths'] = lengths

    # --- Create mask ---
    # True for non-padded values
    mask = torch.arange(max_len)[None, :] < lengths[:, None] # (batch, seq_len)
    collated_batch['mask'] = mask

    return collated_batch