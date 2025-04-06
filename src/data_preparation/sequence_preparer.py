# src/data_preparation/sequence_preparer.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib # Or use pickle
from typing import List, Dict, Tuple, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)

class SequenceDataPreparer:
    """
    Prepares flat encounter data into sequences suitable for PyTorch RNNs.
    Handles scaling, grouping, sorting, and structuring features for embedding layers.
    """
    def __init__(
        self,
        patient_id_col: str,
        timestamp_col: str,
        target_col: str,
        numerical_features: List[str],
        ohe_feature_prefixes: List[str],
        learned_emb_cols: Dict[str, int], # Col name -> vocab size
        precomputed_emb_cols: List[str],
        max_seq_length: Optional[int] = None,
        scaler_path: Optional[str] = None, # Path to save/load scaler
        logger: logging.Logger = None
    ):
        """
        Initializes the preparer with feature definitions and configuration.
        """
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.numerical_features = numerical_features
        self.ohe_feature_prefixes = ohe_feature_prefixes
        self.learned_emb_cols = learned_emb_cols
        self.precomputed_emb_cols = precomputed_emb_cols
        self.max_seq_length = max_seq_length
        self.scaler_path = scaler_path
        self.logger = logger or logging.getLogger(__name__)
        self.scaler: Optional[StandardScaler] = None
        self.fitted = False
        self._ohe_features_actual: List[str] = [] # Determined after seeing data

        self.logger.info(f"SequenceDataPreparer initialized. Max length: {self.max_seq_length or 'Dynamic'}")
        if self.scaler_path and os.path.exists(self.scaler_path):
            self._load_scaler()

    def _identify_ohe_features(self, df: pd.DataFrame):
        """Dynamically identifies OHE columns based on prefixes."""
        self._ohe_features_actual = []
        for prefix in self.ohe_feature_prefixes:
            cols = [col for col in df.columns if col.startswith(prefix + '_')]
            if not cols:
                self.logger.warning(f"No columns found with prefix '{prefix}_'.")
            self._ohe_features_actual.extend(cols)
        self.logger.info(f"Identified {len(self._ohe_features_actual)} OHE columns.")

    def _validate_columns(self, df: pd.DataFrame):
        """Checks if all required feature columns exist in the DataFrame."""
        required_cols = (
            [self.patient_id_col, self.timestamp_col, self.target_col] +
            self.numerical_features +
            list(self.learned_emb_cols.keys()) +
            self.precomputed_emb_cols +
            self._ohe_features_actual # Use actual OHE cols identified
        )
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in DataFrame: {missing}")
        # Check learned embedding columns match expected vocab size (optional, harder to check here)
        self.logger.debug("All required columns validated.")


    def fit_scaler(self, df_train: pd.DataFrame):
        """Fits the StandardScaler on numerical features of the training data."""
        if not self.numerical_features:
            self.logger.info("No numerical features specified for scaling.")
            return
        if self.fitted:
            self.logger.warning("Scaler already fitted or loaded. Skipping fit.")
            return

        self.logger.info(f"Fitting StandardScaler on {len(self.numerical_features)} numerical features.")
        numerical_data = df_train[self.numerical_features].astype(float) # Ensure float
        if numerical_data.isnull().any().any():
             self.logger.warning("NaNs found in numerical features before scaling. Using mean imputation for fitting.")
             means = numerical_data.mean()
             numerical_data = numerical_data.fillna(means)

        self.scaler = StandardScaler()
        self.scaler.fit(numerical_data)
        self.fitted = True
        self.logger.info("StandardScaler fitted.")
        if self.scaler_path:
            self._save_scaler()

    def _save_scaler(self):
        """Saves the fitted scaler."""
        if self.scaler and self.scaler_path:
            try:
                joblib.dump(self.scaler, self.scaler_path)
                self.logger.info(f"Scaler saved to {self.scaler_path}")
            except Exception as e:
                self.logger.error(f"Failed to save scaler to {self.scaler_path}: {e}")

    def _load_scaler(self):
        """Loads the scaler from file."""
        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                self.fitted = True
                self.logger.info(f"Scaler loaded from {self.scaler_path}")
            except Exception as e:
                self.logger.error(f"Failed to load scaler from {self.scaler_path}: {e}. Scaler not loaded.")
                self.scaler = None
                self.fitted = False
        else:
             self.logger.warning(f"Scaler path specified ({self.scaler_path}) but file not found. Scaler not loaded.")


    def transform(self, df: pd.DataFrame) -> Tuple[List[List[Dict[str, Any]]], List[List[int]], List[Any]]:
        """
        Transforms the DataFrame into sequences of feature dictionaries and targets.

        Returns:
            Tuple: (feature_sequences, target_sequences, patient_ids)
                   where feature_sequences is a list of patient sequences,
                   and each patient sequence is a list of visit dictionaries.
                   Each visit dictionary contains keys like 'num_ohe', 'learned_cat_labels', 'precomputed_cat_labels'.
                   target_sequences is a list of lists of target labels.
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit_scaler() on training data first.")
        if not self._ohe_features_actual:
            self._identify_ohe_features(df) # Identify OHE columns if not done yet

        self.logger.info(f"Transforming DataFrame ({len(df)} rows) into sequences.")
        self._validate_columns(df) # Validate after identifying OHE cols

        df_processed = df.copy()

        # Apply scaling
        if self.scaler and self.numerical_features:
            self.logger.debug("Applying fitted StandardScaler.")
            numerical_data = df_processed[self.numerical_features].astype(float)
            if numerical_data.isnull().any().any():
                self.logger.warning("NaNs found during transform numerical data. Filling with fitted mean.")
                means = pd.Series(self.scaler.mean_, index=self.numerical_features)
                numerical_data = numerical_data.fillna(means)
            df_processed[self.numerical_features] = self.scaler.transform(numerical_data)

        # --- Prepare for Grouping ---
        # Combine numerical and OHE features for easier access per row
        num_ohe_cols = self.numerical_features + self._ohe_features_actual
        learned_cols = list(self.learned_emb_cols.keys())
        precomputed_cols = self.precomputed_emb_cols

        all_feature_sequences = []
        all_target_sequences = []
        all_patient_ids = []

        # Group by patient and sort encounters
        grouped = df_processed.sort_values(by=[self.patient_id_col, self.timestamp_col]).groupby(self.patient_id_col)

        for patient_id, group in grouped:
            patient_feature_sequence = []
            patient_target_sequence = []

            # Extract features for each visit
            for _, row in group.iterrows():
                visit_features = {}
                # 1. Numerical + OHE features
                visit_features['num_ohe'] = torch.tensor(row[num_ohe_cols].values.astype(np.float32))

                # 2. Labels for learned embeddings
                learned_labels = {col: int(row[col]) for col in learned_cols}
                visit_features['learned_labels'] = learned_labels

                # 3. Labels for pre-computed embeddings
                precomputed_labels = {col: int(row[col]) for col in precomputed_cols}
                visit_features['precomputed_labels'] = precomputed_labels

                patient_feature_sequence.append(visit_features)
                patient_target_sequence.append(int(row[self.target_col]))

            # Truncate sequences if max_seq_length is set (keep recent visits)
            if self.max_seq_length and len(patient_feature_sequence) > self.max_seq_length:
                 patient_feature_sequence = patient_feature_sequence[-self.max_seq_length:]
                 patient_target_sequence = patient_target_sequence[-self.max_seq_length:]

            all_feature_sequences.append(patient_feature_sequence)
            all_target_sequences.append(patient_target_sequence)
            all_patient_ids.append(patient_id)

        self.logger.info(f"Created {len(all_feature_sequences)} sequences for {len(all_patient_ids)} patients.")
        return all_feature_sequences, all_target_sequences, all_patient_ids