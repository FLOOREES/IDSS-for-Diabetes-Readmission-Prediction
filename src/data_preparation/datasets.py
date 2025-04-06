# src/data_preparation/datasets.py
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class PatientSequenceDataset(Dataset):
    """
    PyTorch Dataset for patient sequences with structured features and targets.
    """
    def __init__(
        self,
        feature_sequences: List[List[Dict[str, Any]]],
        target_sequences: List[List[int]],
        patient_ids: List[Any]
    ):
        """
        Args:
            feature_sequences: List where each item is a patient's sequence of visits.
                               Each visit is a dictionary containing feature tensors/labels.
            target_sequences: List where each item is a patient's sequence of target labels.
            patient_ids: List of patient identifiers corresponding to the sequences.
        """
        if len(feature_sequences) != len(target_sequences) or len(feature_sequences) != len(patient_ids):
            raise ValueError("Mismatch in lengths of sequences, targets, and patient IDs.")

        self.feature_sequences = feature_sequences
        self.target_sequences = target_sequences
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        """Returns the number of patients (sequences)."""
        return len(self.feature_sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing the feature sequence, target sequence,
        patient ID, and original length for a single patient.
        """
        features = self.feature_sequences[idx]
        targets = torch.tensor(self.target_sequences[idx], dtype=torch.long) # Use long for CrossEntropy
        pat_id = self.patient_ids[idx]
        length = len(features) # Original length before padding

        # The collate function will handle structuring features and padding
        return {
            "features": features, # List of dicts
            "targets": targets,   # Tensor
            "length": length,     # Int
            "patient_id": pat_id  # Original ID
        }