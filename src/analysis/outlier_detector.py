# src/analysis/outlier_detector.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib # Or pickle
from sklearn.ensemble import IsolationForest
from typing import Optional, Union, Tuple, Dict, Any, List
import logging
import os
from tqdm import tqdm
from modeling.model_builder import build_autoencoder_from_config

# Import necessary modules (adjust paths as needed)
from modeling.autoencoder import Seq2SeqAE
from modeling.encoder import EncoderRNN
from data_preparation import SequenceDataPreparer, PatientSequenceDataset, pad_collate_fn
from utils.helpers import load_artifact, save_artifact

logger = logging.getLogger(__name__)

class OutlierDetector:
    """
    Performs outlier detection using a trained Autoencoder or Encoder.
    Supports visit-level (reconstruction error) and patient-level (embedding) modes.
    """
    def __init__(
        self,
        data_preparer: SequenceDataPreparer,
        ae_model_load_path: Optional[str] = None,
        isolation_forest_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        sample_batch_for_build: Optional[Dict[str, Any]] = None,
        logger: logging.Logger = None
    ):
        self.data_preparer = data_preparer
        self.ae_model: Optional[Seq2SeqAE] = None
        self.encoder: Optional[EncoderRNN] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.visit_error_threshold: Optional[float] = None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        self.sample_batch_for_build = sample_batch_for_build # Store for potential use

        if ae_model_load_path:
            if self.sample_batch_for_build is None:
                 raise ValueError("sample_batch_for_build is required to load AE model from path.")
            self._load_ae_model(ae_model_load_path, self.sample_batch_for_build)
            if self.ae_model:
                 self.encoder = self.ae_model.get_encoder()
        else:
             self.logger.warning("No AE model path provided. Using passed model object if available.")

        if isolation_forest_path and os.path.exists(isolation_forest_path):
             self._load_isolation_forest(isolation_forest_path)
        elif isolation_forest_path:
             self.logger.warning(f"Isolation Forest path specified ({isolation_forest_path}) but file not found.")


    def _load_ae_model(self, path: str, sample_batch: Dict[str, Any]):
        """ Loads the full Seq2SeqAE model checkpoint. """
        # --- Keep existing implementation ---
        self.logger.info(f"Loading AE model from {path}")
        try:
            self.ae_model = build_autoencoder_from_config(sample_batch, self.logger, self.device)
            checkpoint = load_artifact(path, device=self.device)
            if 'model_state_dict' not in checkpoint: raise KeyError("Missing 'model_state_dict'.")
            self.ae_model.load_state_dict(checkpoint['model_state_dict'])
            self.ae_model.to(self.device)
            self.ae_model.eval()
            self.visit_error_threshold = checkpoint.get('visit_error_threshold', None)
            if self.visit_error_threshold: self.logger.info(f"Loaded visit error threshold: {self.visit_error_threshold:.4f}")
            self.logger.info("AE model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load AE model from {path}: {e}", exc_info=True)
            self.ae_model = None; raise

    def _load_isolation_forest(self, path: str):
        """ Loads a trained Isolation Forest model. """
        # --- Keep existing implementation ---
        self.logger.info(f"Loading Isolation Forest model from {path}")
        try:
            self.isolation_forest = load_artifact(path)
            self.logger.info("Isolation Forest loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Isolation Forest from {path}: {e}")
            self.isolation_forest = None

    def _save_isolation_forest(self, path: str):
         """ Saves a trained Isolation Forest model. """
         # --- Keep existing implementation ---
         if self.isolation_forest:
             self.logger.info(f"Saving Isolation Forest model to {path}")
             save_artifact(self.isolation_forest, path)
         else:
             self.logger.warning("No Isolation Forest model to save.")


    # **** MODIFIED _run_inference ****
    def _run_inference(self, df: pd.DataFrame, mode: str) -> Dict[str, Any]:
        """
        Helper function to run AE or Encoder inference and collect results.
        For AE mode, collects errors per visit *mapped back to original index*.
        For Encoder mode, collects embeddings per patient.
        """
        if (mode == 'ae' and not self.ae_model) or (mode == 'encoder' and not self.encoder):
            raise RuntimeError(f"{mode.upper()} model not loaded or available.")

        self.logger.info(f"Running {mode} inference on {len(df)} rows...")
        model = self.ae_model if mode == 'ae' else self.encoder
        model.eval()

        # Prepare data
        # We need the original index to map results back correctly
        df_sorted = df.sort_values([self.data_preparer.patient_id_col, self.data_preparer.timestamp_col])
        original_indices = df_sorted.index.tolist() # Store original index

        feature_seqs, target_seqs, patient_ids_seq_order = self.data_preparer.transform(df_sorted)
        if not feature_seqs: return {'visit_errors': [], 'embeddings': [], 'patient_ids': []} # Return empty dict

        dataset = PatientSequenceDataset(feature_seqs, target_seqs, patient_ids_seq_order)
        # Use larger batch size for inference
        infer_batch_size = getattr(self.data_preparer, 'max_seq_length', 64) * 2 # Example adaptive batch size
        data_loader = DataLoader(dataset, batch_size=infer_batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=0)

        # --- Store results differently ---
        all_visit_errors: List[Dict[str, Any]] = [] # For AE: [{'original_index': idx, 'error': val}]
        all_patient_embeddings: List[np.ndarray] = [] # For Encoder
        processed_patient_ids: List[Any] = [] # For Encoder

        recon_criterion = nn.L1Loss(reduction='none') # MAE for error calculation

        with torch.no_grad():
            batch_start_original_index = 0 # Pointer into the original_indices list
            for batch in tqdm(data_loader, desc=f"{mode.upper()} Inference", leave=False):
                batch_pids = batch['patient_id']
                mask = batch['mask'] # CPU tensor (batch, seq_len)

                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k not in ['features', 'patient_id']}
                if 'features' in batch: batch_device['features'] = batch['features']

                if mode == 'ae':
                    reconstructions = model(batch_device)
                    embeddings = model.encoder.embedding_manager(batch)
                    num_ohe = batch['num_ohe'].to(self.device)
                    targets = torch.cat((num_ohe, embeddings), dim=-1)
                    mae_per_element = recon_criterion(reconstructions, targets)
                    mae_per_step = torch.sum(mae_per_element, dim=-1).cpu().numpy() # (batch, seq_len)

                    # --- Map visit errors back to original index within the loop ---
                    mask_cpu = mask.cpu().numpy()
                    for i in range(len(batch_pids)): # Iterate through sequences in batch
                        seq_mask = mask_cpu[i]
                        valid_len = int(seq_mask.sum())
                        actual_errors = mae_per_step[i][seq_mask] # Valid errors for this sequence

                        # Get the corresponding original indices for this sequence's valid steps
                        indices_for_seq = original_indices[batch_start_original_index : batch_start_original_index + valid_len]

                        if len(indices_for_seq) != len(actual_errors):
                            self.logger.warning(f"Length mismatch mapping errors for patient {batch_pids[i]}. Skipping.")
                        else:
                            for original_idx, error_val in zip(indices_for_seq, actual_errors):
                                all_visit_errors.append({'original_index': original_idx, 'error': error_val})

                        batch_start_original_index += valid_len # Move pointer

                elif mode == 'encoder':
                    _, final_hidden = model(batch_device)
                    if isinstance(final_hidden, tuple): embedding = final_hidden[0][-1].cpu().numpy()
                    else: embedding = final_hidden[-1].cpu().numpy()
                    all_patient_embeddings.append(embedding)
                    processed_patient_ids.extend(batch_pids)

        # --- Consolidate results ---
        final_results: Dict[str, Any] = {}
        if mode == 'ae':
            final_results['visit_errors'] = all_visit_errors # List of dicts
        elif mode == 'encoder':
            # Concatenate embeddings and ensure unique patient IDs
            if all_patient_embeddings:
                 final_results['embeddings'] = np.concatenate(all_patient_embeddings, axis=0)
                 # Create pandas series to easily drop duplicates based on index (patient id)
                 temp_series = pd.Series(list(final_results['embeddings']), index=processed_patient_ids)
                 temp_series = temp_series[~temp_series.index.duplicated(keep='first')]
                 final_results['patient_ids'] = temp_series.index.tolist()
                 final_results['embeddings'] = np.stack(temp_series.values) # Stack back into array
            else:
                 final_results['embeddings'] = np.empty((0,0))
                 final_results['patient_ids'] = []

        self.logger.info("Inference complete.")
        return final_results
    # **** END MODIFIED _run_inference ****


    def calculate_and_set_visit_threshold(self, df_train: pd.DataFrame, percentile: float):
        """ Calculates reconstruction error threshold on training data. """
        if not self.ae_model: raise RuntimeError("AE model needed to calculate threshold.")
        self.logger.info(f"Calculating visit error threshold ({percentile}th percentile) on training data...")

        results = self._run_inference(df_train, mode='ae')
        visit_errors_list = results.get('visit_errors', [])

        if not visit_errors_list:
             self.logger.error("No visit errors generated from training data inference. Cannot set threshold.")
             self.visit_error_threshold = float('inf')
             return

        # Extract just the error values
        all_valid_errors = [item['error'] for item in visit_errors_list if not np.isnan(item['error'])]

        if len(all_valid_errors) == 0:
            self.logger.warning("No valid (non-NaN) errors found in training data.")
            self.visit_error_threshold = float('inf')
        else:
            self.visit_error_threshold = np.percentile(all_valid_errors, percentile)
            self.logger.info(f"Visit-level MAE threshold set: {self.visit_error_threshold:.4f}")


    def detect_visit_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects visit-level outliers based on reconstruction error.
        Returns the input DataFrame with 'reconstruction_error' and 'is_outlier_visit' columns.
        """
        if self.ae_model is None: raise RuntimeError("AE model not loaded.")
        if self.visit_error_threshold is None: raise RuntimeError("Visit error threshold not calculated/set.")

        self.logger.info("Detecting visit-level outliers...")
        results = self._run_inference(df, mode='ae')
        visit_errors_list = results.get('visit_errors', [])

        if not visit_errors_list:
            self.logger.warning("No errors from inference, returning original DataFrame with NaNs.")
            df_out = df.copy()
            df_out['reconstruction_error'] = np.nan
            df_out['is_outlier_visit'] = False
            return df_out

        # Create DataFrame from mapping and merge back
        error_df = pd.DataFrame(visit_errors_list).set_index('original_index')
        error_df.rename(columns={'error': 'reconstruction_error'}, inplace=True)
        df_out = df.join(error_df, how='left') # Use left join to keep all original rows

        # Apply threshold
        # Handle potential NaNs in reconstruction_error if mapping failed for some rows
        df_out['is_outlier_visit'] = (df_out['reconstruction_error'] > self.visit_error_threshold) & (~df_out['reconstruction_error'].isnull())
        num_outliers = df_out['is_outlier_visit'].sum()
        self.logger.info(f"Visit outlier detection complete. Found {num_outliers} potential outliers.")

        return df_out


    def train_isolation_forest(self, df_train: pd.DataFrame, save_path: Optional[str] = None, **if_params):
        """ Extracts embeddings from training data and trains Isolation Forest. """
        if self.encoder is None: raise RuntimeError("Encoder model not loaded.")

        self.logger.info("Extracting patient embeddings from training data...")
        results = self._run_inference(df_train, mode='encoder')
        embeddings = results.get('embeddings')
        patient_ids = results.get('patient_ids') # Already unique

        if embeddings is None or len(embeddings) == 0:
            self.logger.error("Failed to extract embeddings from training data.")
            return

        embedding_df_train = pd.DataFrame(embeddings, index=patient_ids)

        self.logger.info(f"Training Isolation Forest with params: {if_params}")
        self.isolation_forest = IsolationForest(**if_params)
        self.isolation_forest.fit(embedding_df_train)
        self.logger.info("Isolation Forest training complete.")

        if save_path:
            self._save_isolation_forest(save_path)


    def detect_patient_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects patient-level outliers using embeddings and Isolation Forest.
        Returns a DataFrame indexed by patient_id with 'if_score' and 'is_outlier_patient' columns.
        """
        if self.encoder is None: raise RuntimeError("Encoder model not loaded.")
        if self.isolation_forest is None: raise RuntimeError("Isolation Forest model not loaded or trained.")

        self.logger.info("Extracting patient embeddings for outlier detection...")
        results = self._run_inference(df, mode='encoder')
        embeddings = results.get('embeddings')
        patient_ids = results.get('patient_ids') # Already unique

        if embeddings is None or len(embeddings) == 0:
            self.logger.error("Failed to extract embeddings.")
            return pd.DataFrame(columns=['if_score', 'is_outlier_patient'])

        embedding_df = pd.DataFrame(embeddings, index=patient_ids)
        # No need to drop duplicates here if _run_inference handles it

        self.logger.info(f"Scoring {len(embedding_df)} unique patients with Isolation Forest...")
        scores = self.isolation_forest.decision_function(embedding_df)
        predictions = self.isolation_forest.predict(embedding_df) # -1 for outliers, 1 for inliers

        results_df = pd.DataFrame({
            'if_score': scores,
            'is_outlier_patient': predictions == -1 # True if outlier
        }, index=embedding_df.index)
        results_df.index.name = self.data_preparer.patient_id_col

        num_outliers = results_df['is_outlier_patient'].sum()
        self.logger.info(f"Patient outlier detection complete. Found {num_outliers} potential outliers.")

        return results_df