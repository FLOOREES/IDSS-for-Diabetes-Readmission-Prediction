# data_preparation.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class SequenceDataPreparer:
    """
    Prepares sequential data from a flat DataFrame for PyTorch RNN models.

    Handles grouping, sorting, scaling, padding (batch_first=True),
    and creates attention masks.
    """
    def __init__(
        self,
        patient_id_col: str = 'patient_nbr',
        sort_col: str = 'encounter_id',
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        max_seq_length: Optional[int] = None, # If None, pads to max length in batch
        logger: logging.Logger = None
    ):
        self.patient_id_col = patient_id_col
        self.sort_col = sort_col
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.feature_cols = self.numerical_features + self.categorical_features
        self.max_seq_length = max_seq_length # Can be None for DataLoader padding
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = None

        if not self.feature_cols:
            raise ValueError("At least one numerical or categorical feature must be specified.")
        self.logger.info(f"SequenceDataPreparer initialized. Features: {self.feature_cols}, Max Length: {self.max_seq_length or 'Batch Max'}")


    def fit_scaler(self, df_train: pd.DataFrame):
        """Fits StandardScaler on numerical features of training data."""
        if not self.numerical_features:
            self.logger.info("No numerical features for scaling.")
            return

        self.logger.info(f"Fitting StandardScaler on {len(self.numerical_features)} numerical features.")
        all_train_numerical_data = df_train[self.numerical_features]
        if all_train_numerical_data.isnull().any().any():
             self.logger.warning("NaNs found before scaling. Using mean imputation for fitting.")
             means = all_train_numerical_data.mean()
             all_train_numerical_data = all_train_numerical_data.fillna(means)

        self.scaler = StandardScaler()
        self.scaler.fit(all_train_numerical_data)
        self.logger.info("StandardScaler fitted.")


    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
         """Applies scaling to the DataFrame columns."""
         df_processed = df.copy()
         if self.scaler and self.numerical_features:
            self.logger.info("Applying fitted StandardScaler.")
            numerical_data = df_processed[self.numerical_features]
            if numerical_data.isnull().any().any():
                self.logger.warning("NaNs found during transform. Filling with fitted mean.")
                means = pd.Series(self.scaler.mean_, index=self.numerical_features)
                numerical_data = numerical_data.fillna(means)
            df_processed[self.numerical_features] = self.scaler.transform(numerical_data)
         elif not self.scaler and self.numerical_features:
             self.logger.warning("Scaler not fitted. Skipping scaling.")

         missing_cols = [col for col in self.feature_cols if col not in df_processed.columns]
         if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

         return df_processed


    def create_sequences_and_ids(self, df: pd.DataFrame) -> Tuple[List[torch.Tensor], List]:
        """Groups data, scales, and creates list of sequences (tensors) and patient IDs."""
        df_processed = self.transform_dataframe(df) # Apply scaling first

        sequences = []
        patient_ids = []
        grouped = df_processed.sort_values(by=[self.patient_id_col, self.sort_col]).groupby(self.patient_id_col)

        for patient_id, group in grouped:
            sequence_np = group[self.feature_cols].values
            # Truncate if max_seq_length is set
            if self.max_seq_length and len(sequence_np) > self.max_seq_length:
                 sequence_np = sequence_np[-self.max_seq_length:] # Keep most recent visits (post-truncation)

            sequences.append(torch.tensor(sequence_np, dtype=torch.float32))
            patient_ids.append(patient_id)

        self.logger.info(f"Created {len(sequences)} sequences for {len(patient_ids)} patients.")
        return sequences, patient_ids


class PatientSequenceDataset(Dataset):
    """PyTorch Dataset for patient sequences."""
    def __init__(self, sequences: List[torch.Tensor], patient_ids: List):
        self.sequences = sequences
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return sequence and its original length (useful for packing) or just sequence
        seq = self.sequences[idx]
        # patient_id = self.patient_ids[idx] # Can return if needed later
        return seq, seq.shape[0] # Return sequence and original length


def collate_fn_pad(batch):
    """
    Collate function to pad sequences within a batch.
    batch: List of tuples (sequence, length) from Dataset.__getitem__
    """
    sequences = [item[0] for item in batch]
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)

    # Pad sequences to the max length in this batch
    # batch_first=True makes output (batch, seq_len, features)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Create attention mask (True for non-padded values)
    # Mask shape should be (batch, seq_len)
    max_len = padded_sequences.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None] # Broadcasting comparison

    return padded_sequences, lengths, mask