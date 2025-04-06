# src/analysis/predictor_inference.py
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Union, Tuple, Dict, Any, List
import logging
import os
from tqdm import tqdm

# Import necessary modules (adjust paths as needed)
from modeling.predictor import PredictorModel
# Need model building logic if loading state_dict
from modeling import PredictionHead
from data_preparation import SequenceDataPreparer, PatientSequenceDataset, pad_collate_fn
from utils.helpers import load_artifact
from modeling.model_builder import build_autoencoder_from_config # Assuming this is a helper function to reconstruct the model from config


logger = logging.getLogger(__name__)

class Predictor:
    """
    Handles inference using the trained sequence prediction model.
    """
    def __init__(
        self,
        model_path: Optional[str], # Path to trained PredictorModel (.pth) state_dict
        model_config: Optional[Dict[str, Any]] = None, # Config needed if rebuilding arch
        data_preparer: Optional[SequenceDataPreparer] = None,
        # Allow passing trained model directly
        trained_model: Optional[PredictorModel] = None,
        device: Optional[torch.device] = None,
        logger: logging.Logger = None
    ):
        self.model: Optional[PredictorModel] = trained_model
        self.data_preparer = data_preparer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        self.model_config = model_config # Store config if needed for rebuild

        if self.model is None:
            if model_path and model_config and data_preparer:
                 # Need sample batch info from data_preparer (or config) to build AE first
                 self.logger.warning("Loading predictor from path requires sample batch simulation or stored config.")
                 # Placeholder: Need a way to get dims to build AE/Encoder/Head
                 # self._load_model(model_path, model_config, sample_batch_placeholder)
                 raise NotImplementedError("Loading predictor from path needs sample_batch logic.")
            elif model_path:
                 raise ValueError("model_config and data_preparer needed to load predictor from path.")
            else:
                 self.logger.warning("No trained model or path provided during initialization.")
        elif trained_model:
            self.model = trained_model.to(self.device)
            self.model.eval() # Ensure eval mode
            self.logger.info("Using provided pre-trained predictor model.")


    def _load_model(self, path: str, config: Dict[str, Any], sample_batch: Dict):
         """ Loads the trained PredictorModel checkpoint. """
         self.logger.info(f"Loading Predictor model from {path}")
         try:
            # 1. Rebuild AE architecture to get Encoder
            # This dependency on AE building is a bit awkward, consider saving/loading PredictorModel directly
            temp_ae = build_autoencoder_from_config(sample_batch, self.logger, self.device) # Config for AE needed
            encoder = temp_ae.get_encoder()

            # 2. Rebuild Prediction Head
            num_classes = config.get('num_classes', 3) # Get num_classes from config
            hidden_dim = config.get('hidden_dim', 128) # Get hidden_dim from config
            head = PredictionHead(input_dim=hidden_dim, output_dim=num_classes)

            # 3. Combine into PredictorModel
            self.model = PredictorModel(encoder, head)

            # 4. Load state dict
            checkpoint = load_artifact(path, device=self.device)
            if 'model_state_dict' not in checkpoint: raise KeyError("Missing 'model_state_dict'.")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Predictor model loaded successfully.")

         except Exception as e:
             self.logger.error(f"Failed to load Predictor model from {path}: {e}", exc_info=True)
             self.model = None
             raise


    def predict_sequence(self, df_patient: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts readmission probabilities/classes for each visit in a single patient's sequence.
        """
        if self.model is None: raise RuntimeError("Predictor model not loaded.")
        if self.data_preparer is None: raise RuntimeError("DataPreparer not provided.")

        self.logger.debug(f"Predicting sequence for patient ID: {df_patient[self.data_preparer.patient_id_col].iloc[0]}")
        self.model.eval()

        # Prepare single sequence
        feature_seqs, target_seqs, pids = self.data_preparer.transform(df_patient)
        if not feature_seqs: return df_patient # Return original if no sequence generated

        dataset = PatientSequenceDataset(feature_seqs, target_seqs, pids)
        # Create a batch of size 1 using the collate function
        batch = pad_collate_fn([dataset[0]]) # Get first (only) item and collate

        # Move to device
        batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'features'}
        if 'features' in batch: batch_device['features'] = batch['features']

        with torch.no_grad():
            logits = self.model(batch_device) # (1, seq_len, num_classes)

        # Apply activation (Softmax for multi-class)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy() # (seq_len, num_classes)
        predicted_classes = np.argmax(probs, axis=-1) # (seq_len,)

        # Get valid length using mask
        mask = batch['mask'].squeeze(0).cpu().numpy() # (seq_len,)
        valid_len = int(mask.sum())

        # Extract valid predictions
        valid_probs = probs[mask]
        valid_preds = predicted_classes[mask]

        # Create output DataFrame
        df_out = df_patient.copy()
        if len(df_out) == valid_len:
             # Add probability columns
             for i in range(probs.shape[-1]): # probs shape (valid_len, num_classes)
                  df_out[f'pred_class_{i}_prob'] = valid_probs[:, i]
             df_out['predicted_class'] = valid_preds
        else:
             self.logger.warning(f"Length mismatch in predict_sequence ({len(df_out)} vs {valid_len}). Adding NaNs.")
             for i in range(probs.shape[-1]): df_out[f'pred_class_{i}_prob'] = np.nan
             df_out['predicted_class'] = np.nan


        return df_out


    def predict_bulk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts readmission probabilities/classes for all visits in a larger DataFrame.
        """
        if self.model is None: raise RuntimeError("Predictor model not loaded.")
        if self.data_preparer is None: raise RuntimeError("DataPreparer not provided.")

        self.logger.info(f"Running bulk prediction on {len(df)} rows...")
        self.model.eval()

        # Prepare data
        feature_seqs, target_seqs, patient_ids = self.data_preparer.transform(df)
        if not feature_seqs:
            self.logger.warning("No sequences generated from input DataFrame.")
            return df # Or add empty prediction columns

        dataset = PatientSequenceDataset(feature_seqs, target_seqs, patient_ids)
        # Use a reasonable batch size for inference
        batch_size = self.data_preparer.max_seq_length * 2 if self.data_preparer.max_seq_length else 128
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=0)

        all_predictions = [] # Store results as dicts before creating DataFrame

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting Batches", leave=False):
                batch_pids = batch['patient_id'] # List of patient IDs in this batch
                mask = batch['mask'] # CPU tensor (batch, seq_len)

                # Move necessary tensors to device
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k not in ['features', 'patient_id']}
                if 'features' in batch: batch_device['features'] = batch['features']

                logits = self.model(batch_device) # (batch, seq_len, num_classes)
                probs = F.softmax(logits, dim=-1) # (batch, seq_len, num_classes)
                preds = torch.argmax(probs, dim=-1) # (batch, seq_len)

                # Move results to CPU for processing
                probs_cpu = probs.cpu().numpy()
                preds_cpu = preds.cpu().numpy()
                mask_cpu = mask.cpu().numpy()

                # Map results back to individual visits
                # Iterate through each sequence in the batch
                for i in range(len(batch_pids)):
                    pid = batch_pids[i]
                    seq_mask = mask_cpu[i]
                    valid_len = int(seq_mask.sum())

                    # Extract valid predictions for this sequence
                    seq_probs = probs_cpu[i][seq_mask] # (valid_len, num_classes)
                    seq_preds = preds_cpu[i][seq_mask] # (valid_len,)

                    # Find the original DataFrame rows for this patient
                    # Requires the original DataFrame's index and sorting consistency
                    # This assumes df was sorted by patient/timestamp before transform
                    # NOTE: This mapping logic is complex and prone to off-by-one errors
                    # Need a robust way to link sequence step 'j' to original encounter ID or index

                    # Alternative: Store encounter_id alongside features in dataset/collate
                    # Or, more simply, assume the order matches df filtered by patient ID
                    patient_rows = df[df[self.data_preparer.patient_id_col] == pid].sort_values(self.data_preparer.timestamp_col)

                    # Handle potential truncation
                    if self.data_preparer.max_seq_length and len(patient_rows) > self.data_preparer.max_seq_length:
                         target_indices = patient_rows.index[-valid_len:] # Get indices of last visits
                    else:
                         target_indices = patient_rows.index[:valid_len] # Get indices of first visits

                    if len(target_indices) != valid_len:
                         self.logger.warning(f"Mapping length mismatch for patient {pid}: DF slice {len(target_indices)}, valid preds {valid_len}. Skipping patient.")
                         continue

                    # Store results with original index
                    for j, original_idx in enumerate(target_indices):
                         result = {'original_index': original_idx}
                         for k in range(probs.shape[-1]): # Add prob for each class
                              result[f'pred_class_{k}_prob'] = seq_probs[j, k]
                         result['predicted_class'] = seq_preds[j]
                         all_predictions.append(result)

        # Create final DataFrame
        if not all_predictions:
            self.logger.warning("No predictions were generated.")
            df_out = df.copy()
            # Add empty prediction columns
            for k in range(logits.shape[-1]): df_out[f'pred_class_{k}_prob'] = np.nan
            df_out['predicted_class'] = np.nan
            return df_out

        pred_df = pd.DataFrame(all_predictions).set_index('original_index')
        df_out = df.join(pred_df) # Join predictions based on original index

        self.logger.info("Bulk prediction finished.")
        return df_out