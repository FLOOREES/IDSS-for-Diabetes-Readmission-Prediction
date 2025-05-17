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
import json # For saving summary dicts

# Assume these are importable or passed in
from src.modeling.autoencoder import Seq2SeqAE
from src.modeling.encoder import EncoderRNN
from src.modeling.model_builder import build_autoencoder_from_config
from src.data_preparation import SequenceDataPreparer, PatientSequenceDataset, pad_collate_fn
from src.utils.helpers import load_artifact, save_artifact

logger = logging.getLogger(__name__)

class OutlierDetector:
    # --- __init__, _load_*, _save_isolation_forest, _run_inference remain the same ---
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
        self.sample_batch_for_build = sample_batch_for_build

        if ae_model_load_path:
            if self.sample_batch_for_build is None:
                 raise ValueError("sample_batch_for_build required to load AE model from path.")
            self._load_ae_model(ae_model_load_path, self.sample_batch_for_build)
            if self.ae_model: self.encoder = self.ae_model.get_encoder()
        else: self.logger.warning("No AE model path provided.")

        if isolation_forest_path and os.path.exists(isolation_forest_path):
             self._load_isolation_forest(isolation_forest_path)
        elif isolation_forest_path: self.logger.warning(f"IF path ({isolation_forest_path}) not found.")

    def _load_ae_model(self, path: str, sample_batch: Dict[str, Any]):
        self.logger.info(f"Loading AE model from {path}")
        try:
            self.ae_model = build_autoencoder_from_config(sample_batch, self.logger, self.device)
            checkpoint = load_artifact(path, device=self.device)
            if 'model_state_dict' not in checkpoint: raise KeyError("Missing 'model_state_dict'.")
            self.ae_model.load_state_dict(checkpoint['model_state_dict'])
            self.ae_model.to(self.device); self.ae_model.eval()
            self.visit_error_threshold = checkpoint.get('visit_error_threshold', None)
            if self.visit_error_threshold: self.logger.info(f"Loaded visit error threshold: {self.visit_error_threshold:.4f}")
            self.logger.info("AE model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load AE model from {path}: {e}", exc_info=True)
            self.ae_model = None; raise

    def _load_isolation_forest(self, path: str):
        self.logger.info(f"Loading IF model from {path}")
        try: self.isolation_forest = load_artifact(path); self.logger.info("IF loaded.")
        except Exception as e: self.logger.error(f"Failed to load IF: {e}"); self.isolation_forest = None

    def _save_isolation_forest(self, path: str):
         if self.isolation_forest: self.logger.info(f"Saving IF model to {path}"); save_artifact(self.isolation_forest, path)
         else: self.logger.warning("No IF model to save.")

    def _run_inference(self, df: pd.DataFrame, mode: str) -> Dict[str, Any]:
         # --- Keep implementation from previous step ---
        if (mode == 'ae' and not self.ae_model) or (mode == 'encoder' and not self.encoder):
            raise RuntimeError(f"{mode.upper()} model not loaded.")
        self.logger.info(f"Running {mode} inference on {len(df)} rows...")
        model = self.ae_model if mode == 'ae' else self.encoder
        model.eval()
        df_sorted = df.sort_values([self.data_preparer.patient_id_col, self.data_preparer.timestamp_col])
        original_indices = df_sorted.index.tolist()
        feature_seqs, target_seqs, patient_ids_seq_order = self.data_preparer.transform(df_sorted)
        if not feature_seqs: return {'visit_errors': [], 'embeddings': [], 'patient_ids': []}
        dataset = PatientSequenceDataset(feature_seqs, target_seqs, patient_ids_seq_order)
        infer_batch_size = getattr(self.data_preparer, 'max_seq_length', 64) * 2
        data_loader = DataLoader(dataset, batch_size=infer_batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=0)
        all_visit_errors: List[Dict[str, Any]] = []
        all_patient_embeddings: List[np.ndarray] = []
        processed_patient_ids: List[Any] = []
        recon_criterion = nn.L1Loss(reduction='none')
        with torch.no_grad():
            batch_start_original_index = 0
            for batch in tqdm(data_loader, desc=f"{mode.upper()} Inference", leave=False):
                batch_pids = batch['patient_id']; mask = batch['mask']
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k not in ['features', 'patient_id']}
                if 'features' in batch: batch_device['features'] = batch['features']
                if mode == 'ae':
                    reconstructions = model(batch_device)
                    embeddings = model.encoder.embedding_manager(batch)
                    num_ohe = batch['num_ohe'].to(self.device)
                    targets = torch.cat((num_ohe, embeddings), dim=-1)
                    mae_per_element = recon_criterion(reconstructions, targets)
                    mae_per_step = torch.sum(mae_per_element, dim=-1).cpu().numpy()
                    mask_cpu = mask.cpu().numpy()
                    for i in range(len(batch_pids)):
                        seq_mask = mask_cpu[i]; valid_len = int(seq_mask.sum())
                        actual_errors = mae_per_step[i][seq_mask]
                        indices_for_seq = original_indices[batch_start_original_index : batch_start_original_index + valid_len]
                        if len(indices_for_seq) == len(actual_errors):
                            for original_idx, error_val in zip(indices_for_seq, actual_errors):
                                all_visit_errors.append({'original_index': original_idx, 'error': error_val})
                        else: self.logger.warning(f"Length mismatch mapping errors for patient {batch_pids[i]}. Skipping.")
                        batch_start_original_index += valid_len
                elif mode == 'encoder':
                    _, final_hidden = model(batch_device)
                    if isinstance(final_hidden, tuple): embedding = final_hidden[0][-1].cpu().numpy()
                    else: embedding = final_hidden[-1].cpu().numpy()
                    all_patient_embeddings.append(embedding); processed_patient_ids.extend(batch_pids)
        final_results: Dict[str, Any] = {}
        if mode == 'ae': final_results['visit_errors'] = all_visit_errors
        elif mode == 'encoder':
            if all_patient_embeddings:
                 embeddings_concat = np.concatenate(all_patient_embeddings, axis=0)
                 temp_series = pd.Series(list(embeddings_concat), index=processed_patient_ids)
                 temp_series = temp_series[~temp_series.index.duplicated(keep='first')]
                 final_results['patient_ids'] = temp_series.index.tolist()
                 final_results['embeddings'] = np.stack(temp_series.values)
            else: final_results['embeddings'] = np.empty((0,0)); final_results['patient_ids'] = []
        self.logger.info("Inference complete.")
        return final_results

    def calculate_and_set_visit_threshold(self, df_train: pd.DataFrame, percentile: float):
        """ Calculates reconstruction error threshold on training data. """
        # --- Keep existing implementation ---
        if not self.ae_model: raise RuntimeError("AE model needed.")
        self.logger.info(f"Calculating visit error threshold ({percentile}th percentile)...")
        results = self._run_inference(df_train, mode='ae')
        visit_errors_list = results.get('visit_errors', [])
        if not visit_errors_list: self.logger.error("No errors from train data."); self.visit_error_threshold = float('inf'); return
        all_valid_errors = [item['error'] for item in visit_errors_list if not np.isnan(item['error'])]
        if not all_valid_errors: self.logger.warning("No valid errors found."); self.visit_error_threshold = float('inf')
        else: self.visit_error_threshold = np.percentile(all_valid_errors, percentile); self.logger.info(f"Visit MAE threshold set: {self.visit_error_threshold:.4f}")


    # **** MODIFIED detect_visit_outliers ****
    def detect_visit_outliers(self, df: pd.DataFrame, results_save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Detects visit-level outliers based on reconstruction error. Logs summary
        and optionally saves detailed outlier info.

        Args:
            df: Input DataFrame to analyze.
            results_save_path: Optional path to save outlier details (JSON).

        Returns:
            DataFrame with 'reconstruction_error' and 'is_outlier_visit' columns.
        """
        if self.ae_model is None: raise RuntimeError("AE model not loaded.")
        if self.visit_error_threshold is None: raise RuntimeError("Visit error threshold not set.")

        self.logger.info("Detecting visit-level outliers...")
        results = self._run_inference(df, mode='ae')
        visit_errors_list = results.get('visit_errors', [])

        if not visit_errors_list:
            # ... (keep existing handling for no results) ...
            self.logger.warning("No results from inference, returning original DataFrame with NaNs.")
            df_out = df.copy(); df_out['reconstruction_error'] = np.nan; df_out['is_outlier_visit'] = False
            return df_out

        error_df = pd.DataFrame(visit_errors_list).set_index('original_index')
        error_df.rename(columns={'error': 'reconstruction_error'}, inplace=True)
        df_out = df.join(error_df, how='left')

        df_out['is_outlier_visit'] = (df_out['reconstruction_error'] > self.visit_error_threshold) & (~df_out['reconstruction_error'].isnull())
        num_outliers = int(df_out['is_outlier_visit'].sum()) # Cast to int for JSON
        total_visits = len(df_out)
        outlier_percentage = (num_outliers / total_visits * 100) if total_visits > 0 else 0

        # --- Logging and Saving Enhanced Results ---
        summary = {
            "mode": "visit",
            "total_visits_analyzed": total_visits,
            "threshold_percentile": self.data_preparer.visit_error_threshold_percentile if hasattr(self.data_preparer, 'visit_error_threshold_percentile') else 'N/A', # Get percentile if stored
            "reconstruction_error_threshold": self.visit_error_threshold,
            "num_outliers_detected": num_outliers,
            "percentage_outliers": f"{outlier_percentage:.2f}%",
            "reconstruction_error_mean": df_out['reconstruction_error'].mean(),
            "reconstruction_error_median": df_out['reconstruction_error'].median(),
            "reconstruction_error_std": df_out['reconstruction_error'].std(),
        }
        self.logger.info("--- Visit Outlier Detection Summary ---")
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")

        # Extract details of outliers
        outlier_details_df = df_out[df_out['is_outlier_visit']][[
            self.data_preparer.patient_id_col, # Add patient ID
            self.data_preparer.timestamp_col, # Add timestamp/encounter ID
            'reconstruction_error'
        ]].reset_index() # Keep original index if needed, or reset
        outlier_details_df.rename(columns={'index': 'original_df_index'}, inplace=True) # Clarify index name
        outlier_details_list = outlier_details_df.to_dict(orient='records')

        self.logger.info(f"Top 5 Visit Outliers (by error):\n{outlier_details_df.nlargest(5, 'reconstruction_error')}")

        if results_save_path:
            results_to_save = {
                "summary": summary,
                "outlier_details": outlier_details_list
            }
            try:
                with open(results_save_path, 'w') as f:
                    json.dump(results_to_save, f, indent=4, default=str) # Use default=str for numpy types
                self.logger.info(f"Outlier detection results saved to: {results_save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save outlier results: {e}")
        # --- End Logging/Saving ---

        return df_out

    # **** MODIFIED detect_patient_outliers ****
    def detect_patient_outliers(self, df: pd.DataFrame, results_save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Detects patient-level outliers using embeddings and Isolation Forest. Logs summary
        and optionally saves detailed outlier info.

        Args:
            df: Input DataFrame to analyze.
            results_save_path: Optional path to save outlier details (JSON).

        Returns:
            DataFrame indexed by patient_id with 'if_score' and 'is_outlier_patient' columns.
        """
        if self.encoder is None: raise RuntimeError("Encoder model not loaded.")
        if self.isolation_forest is None: raise RuntimeError("Isolation Forest model not loaded or trained.")

        self.logger.info("Extracting patient embeddings for outlier detection...")
        results = self._run_inference(df, mode='encoder')
        embeddings = results.get('embeddings')
        patient_ids = results.get('patient_ids') # Unique patient IDs

        if embeddings is None or len(embeddings) == 0:
             # ... (keep existing handling) ...
            self.logger.error("Failed to extract embeddings.")
            return pd.DataFrame(columns=['if_score', 'is_outlier_patient'])

        embedding_df = pd.DataFrame(embeddings, index=patient_ids)
        embedding_df.index.name = self.data_preparer.patient_id_col

        self.logger.info(f"Scoring {len(embedding_df)} unique patients with Isolation Forest...")
        scores = self.isolation_forest.decision_function(embedding_df)
        predictions = self.isolation_forest.predict(embedding_df) # -1 for outliers, 1 for inliers

        results_df = pd.DataFrame({
            'if_score': scores, # Lower score -> more anomalous
            'is_outlier_patient': predictions == -1 # True if outlier
        }, index=embedding_df.index)

        num_outliers = int(results_df['is_outlier_patient'].sum())
        total_patients = len(results_df)
        outlier_percentage = (num_outliers / total_patients * 100) if total_patients > 0 else 0

        # --- Logging and Saving Enhanced Results ---
        summary = {
            "mode": "patient",
            "total_patients_analyzed": total_patients,
            "if_contamination": getattr(self.isolation_forest, 'contamination', 'N/A'),
            "num_outliers_detected": num_outliers,
            "percentage_outliers": f"{outlier_percentage:.2f}%",
            "if_score_mean": results_df['if_score'].mean(),
            "if_score_median": results_df['if_score'].median(),
            "if_score_std": results_df['if_score'].std(),
        }
        self.logger.info("--- Patient Outlier Detection Summary ---")
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")

        outlier_details_df = results_df[results_df['is_outlier_patient']].sort_values('if_score').reset_index()
        outlier_details_list = outlier_details_df.to_dict(orient='records')

        self.logger.info(f"Top 5 Patient Outliers (by score):\n{outlier_details_df.head(5)}")

        if results_save_path:
            results_to_save = {
                "summary": summary,
                "outlier_details": outlier_details_list
            }
            try:
                with open(results_save_path, 'w') as f:
                    json.dump(results_to_save, f, indent=4, default=str)
                self.logger.info(f"Outlier detection results saved to: {results_save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save outlier results: {e}")
        # --- End Logging/Saving ---

        return results_df # Return the DataFrame as before

    # --- train_isolation_forest remains the same ---
    def train_isolation_forest(self, df_train: pd.DataFrame, save_path: Optional[str] = None, **if_params):
        """ Extracts embeddings from training data and trains Isolation Forest. """
        if self.encoder is None: raise RuntimeError("Encoder model not loaded.")
        self.logger.info("Extracting patient embeddings from training data...")
        results = self._run_inference(df_train, mode='encoder')
        embeddings = results.get('embeddings'); patient_ids = results.get('patient_ids')
        if embeddings is None or len(embeddings) == 0: self.logger.error("Failed to extract embeddings."); return
        embedding_df_train = pd.DataFrame(embeddings, index=patient_ids)
        self.logger.info(f"Training Isolation Forest with params: {if_params}")
        self.isolation_forest = IsolationForest(**if_params)
        self.isolation_forest.fit(embedding_df_train)
        self.logger.info("Isolation Forest training complete.")
        if save_path: self._save_isolation_forest(save_path)