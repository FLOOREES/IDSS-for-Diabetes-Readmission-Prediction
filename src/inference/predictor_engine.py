import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F # For softmax
import logging
import os
import json # For loading OHE feature names
from typing import Dict, Any, List, Optional
from functools import partial

# --- Your Project's Modules ---
# We'll need to import your main config. Let's assume it can be imported as AppConfig
# If not, you'd import individual variables.
# Example: from src import config as AppConfig
# For this example, I'll refer to AppConfig.VARIABLE_NAME.
# You'll need to adjust based on how your config is structured and imported.
from src import config as AppConfig # Placeholder, adjust as per your project structure

from src.preprocessing.first_phase import FirstPhasePreprocessor
from src.preprocessing.second_phase import SecondPhasePreprocessor
from src.data_preparation.sequence_preparer import SequenceDataPreparer
# PatientSequenceDataset is not directly instantiated for a single item, 
# but its output structure is replicated for pad_collate_fn
from src.data_preparation.collators import pad_collate_fn 
from src.modeling.predictor import PredictorModel
from src.modeling.model_builder import build_autoencoder_from_config # For encoder structure
from src.modeling.prediction_head import PredictionHead
from src.utils import load_artifact # Assuming load_artifact is in src.utils

logger = logging.getLogger(__name__)

class SinglePatientPredictorEngine:
    def __init__(self, cfg: Any): # cfg should be your main config object/module
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SinglePatientPredictorEngine initialized on device: {self.device}")

        # --- 1. Load All Preprocessing Artifacts & Initialize Preprocessors ---
        logger.info("Loading preprocessing artifacts and initializing preprocessors...")
        
        # FirstPhasePreprocessor
        self.phase1_processor = FirstPhasePreprocessor(
            drop_columns=self.cfg.DROP_COLUMNS,
            one_hot_columns=self.cfg.ONE_HOT_COLUMNS,
            ordinal_mappings=self.cfg.ORDINAL_MAPPINGS,
            treatment_columns=self.cfg.TREATMENT_COLUMNS,
            treatment_mapping=self.cfg.TREATMENT_MAPPING,
            missing_values_encoding=self.cfg.MISSING_VALUES,
            ohe_encoder_path=self.cfg.PHASE1_OHE_ENCODER_PATH,
            ohe_feature_names_path=self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH,
            low_variance_cols_path=self.cfg.PHASE1_LOW_VAR_COLS_PATH,
            logger=logger.getChild("Phase1") # Pass a child logger for better context
        )
        if not self.phase1_processor.load_fitted_state(): # This method loads the artifacts
            raise RuntimeError("Critical: Failed to load fitted state for FirstPhasePreprocessor. Ensure artifacts exist.")
        logger.info("FirstPhasePreprocessor state loaded successfully.")


        # SecondPhasePreprocessor (it loads its artifacts internally based on path existence)
        self.phase2_processor = SecondPhasePreprocessor(
            diag_embeddings_path=self.cfg.DIAG_EMBEDDINGS_PATH,
            diag_label_encoder_path=self.cfg.DIAG_LABEL_ENCODER_PATH,
            label_encoders_path=self.cfg.LABEL_ENCODERS_PATH,
            icd9_hierarchy_path=self.cfg.ICD9_HIERARCHY_PATH,
            icd9_chapters_path=self.cfg.ICD9_CHAPTERS_PATH,
            spacy_model_name=self.cfg.SPACY_MODEL_NAME,
            label_encode_columns=self.cfg.LABEL_ENCODING,
            embedding_dim=getattr(self.cfg, 'DIAGNOSIS_EMBEDDING_DIM', 16),
            tsne_n_components=getattr(self.cfg, 'DIAGNOSIS_TSNE_COMPONENTS', 16), # Ensure this matches embedding_dim if tSNE is the final step
            logger=logger.getChild("Phase2")
        )
        # Add a check here to ensure phase2_processor's artifacts (embeddings, LE maps) are ready if possible,
        # or rely on its internal checks during its first transform if that's how it's designed.
        # For now, we assume it will load them when its transform is called, or they are already in memory if used before.
        logger.info("SecondPhasePreprocessor initialized.")

        # Load OHE feature names list for SequenceDataPreparer
        actual_ohe_columns_list: Optional[List[str]] = None
        if not os.path.exists(self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH):
            raise RuntimeError(f"Critical: OHE feature names file not found at {self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH}")
        try:
            with open(self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH, 'r') as f:
                actual_ohe_columns_list = json.load(f)
            if not isinstance(actual_ohe_columns_list, list):
                raise ValueError("OHE feature names file did not contain a list.")
            logger.info(f"OHE feature names list loaded successfully from {self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH}")
        except Exception as e:
            raise RuntimeError(f"Critical: Failed to load or parse OHE feature names from {self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH}: {e}")
        
        self.actual_ohe_columns_for_shap_naming: List[str] = actual_ohe_columns_list


        self.data_preparer = SequenceDataPreparer(
            patient_id_col=self.cfg.PATIENT_ID_COL,
            timestamp_col=self.cfg.ENCOUNTER_ID_COL,
            target_col=self.cfg.TARGET_COL, # Target col might not be in inference data, handled in _prepare_model_input
            numerical_features=self.cfg.NUMERICAL_FEATURES,
            ohe_columns=actual_ohe_columns_list,
            learned_emb_cols=self.cfg.LEARNED_EMB_COLS,
            precomputed_emb_cols=self.cfg.PRECOMPUTED_EMB_COLS,
            max_seq_length=self.cfg.MAX_SEQ_LENGTH,
            scaler_path=self.cfg.SCALER_PATH, # Scaler is loaded in SequenceDataPreparer's __init__
            logger=logger.getChild("DataPreparer")
        )
        if not self.data_preparer.scaler or not self.data_preparer.fitted:
             raise RuntimeError(f"Critical: Scaler not loaded by SequenceDataPreparer from {self.cfg.SCALER_PATH}. Ensure artifact exists and is valid.")
        logger.info("SequenceDataPreparer state (scaler) loaded successfully.")

        # --- 2. Load the Trained PredictorModel ---
        logger.info(f"Loading PredictorModel from {self.cfg.PREDICTOR_MODEL_LOAD_PATH}...")
        # Uses our refactored build_autoencoder_from_config which gets structural info from AppConfig
        temp_ae = build_autoencoder_from_config(logger=logger.getChild("AEBuilder"), device=self.device)
        encoder = temp_ae.get_encoder()
        
        num_classes = getattr(self.cfg, 'NUM_CLASSES', 3)
        encoder_hidden_dim = getattr(self.cfg, 'HIDDEN_DIM', 128) # This is the encoder's output dim
        head = PredictionHead(input_dim=encoder_hidden_dim, output_dim=num_classes)
        
        self.predictor_model = PredictorModel(encoder, head)
        
        checkpoint = load_artifact(self.cfg.PREDICTOR_MODEL_LOAD_PATH, device=self.device)
        if 'model_state_dict' not in checkpoint: 
            raise KeyError(f"Checkpoint from {self.cfg.PREDICTOR_MODEL_LOAD_PATH} missing 'model_state_dict'.")
        self.predictor_model.load_state_dict(checkpoint['model_state_dict'])
        self.predictor_model.to(self.device)
        self.predictor_model.eval() # Set to evaluation mode
        logger.info("PredictorModel loaded, configured, and set to eval mode.")

    def _preprocess_raw_patient_df(self, raw_patient_df: pd.DataFrame) -> pd.DataFrame:
        """Applies Phase 1 and Phase 2 preprocessing to a raw patient DataFrame."""
        logger.debug(f"Applying Phase 1 preprocessing to patient data (shape: {raw_patient_df.shape}).")
        df_p1 = self.phase1_processor.transform(raw_patient_df.copy())
        logger.debug(f"After Phase 1, shape: {df_p1.shape}.")
        
        logger.debug(f"Applying Phase 2 preprocessing (shape: {df_p1.shape}).")
        # For inference, we don't need to save the intermediate output of phase2_processor
        df_p2 = self.phase2_processor.transform(df_p1.copy(), output_path=None)
        logger.debug(f"After Phase 2, shape: {df_p2.shape}.")
        return df_p2

    def _prepare_model_input_batch(self, processed_patient_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Uses SequenceDataPreparer and pad_collate_fn to prepare a model-ready input batch
        from a single patient's fully preprocessed DataFrame.
        """
        logger.debug(f"Preparing model input from processed_patient_df (shape: {processed_patient_df.shape}).")
        
        # Ensure target column exists for SequenceDataPreparer, even if dummy for inference
        temp_target_col_name = self.data_preparer.target_col
        df_for_sdp = processed_patient_df.copy() # Work on a copy
        if temp_target_col_name not in df_for_sdp.columns:
            logger.info(f"Target column '{temp_target_col_name}' not in inference input. Adding dummy target.")
            df_for_sdp[temp_target_col_name] = 0 # Dummy value, won't be used for prediction loss

        # SequenceDataPreparer.transform returns lists (each with one item for a single patient)
        feature_seqs, target_seqs, pids = self.data_preparer.transform(df_for_sdp)
        
        if not feature_seqs: # Should be a list containing one patient's sequence
            msg = "No feature sequences generated by SequenceDataPreparer for the input patient data."
            logger.error(msg)
            raise ValueError(msg)

        # Construct the item structure that PatientSequenceDataset.__getitem__ produces
        dataset_like_item = {
            "features": feature_seqs[0], # The list of visit dictionaries for this patient
            "targets": torch.tensor(target_seqs[0], dtype=torch.long), # Target tensor for this patient
            "length": len(feature_seqs[0]), # Number of visits for this patient
            "patient_id": pids[0] # Patient ID
        }
        
        # Use pad_collate_fn to create a batch of size 1
        # pad_collate_fn expects a list of such items
        max_len_for_padding = getattr(self.cfg, 'MAX_SEQ_LENGTH', None)
        collate_fn_with_config = partial(pad_collate_fn, enforced_max_len=max_len_for_padding)

        collated_batch = collate_fn_with_config([dataset_like_item]) 
        logger.debug("Patient data collated into a batch of size 1.")
        return collated_batch

    def _move_batch_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to move all tensors in the batch (including nested ones) to device."""
        moved_batch = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict): # For 'learned_labels', 'precomputed_labels'
                moved_batch[key] = {k: v.to(self.device) for k, v in value.items() if isinstance(v, torch.Tensor)}
            else: 
                moved_batch[key] = value 
        return moved_batch

    def predict_for_patient(self, raw_patient_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs end-to-end prediction for a single patient's raw data DataFrame.

        Args:
            raw_patient_df: Pandas DataFrame containing all visits for a single patient.
                            Must be sorted chronologically if order matters beyond timestamp_col.

        Returns:
            A dictionary containing:
                'visit_predictions': List of dicts, one per visit, with probabilities and predicted class.
                'processed_model_input': The collated batch fed to the model (useful for SHAP).
        """
        if raw_patient_df.empty or self.cfg.PATIENT_ID_COL not in raw_patient_df or \
           raw_patient_df[self.cfg.PATIENT_ID_COL].nunique() != 1:
            raise ValueError("Input must be a DataFrame for a single patient, containing patient ID.")

        patient_id_val = raw_patient_df[self.cfg.PATIENT_ID_COL].iloc[0]
        logger.info(f"Starting single-case prediction for patient ID: {patient_id_val} with {len(raw_patient_df)} visits.")

        # 1. Full Preprocessing (Phase 1 & Phase 2)
        processed_patient_df = self._preprocess_raw_patient_df(raw_patient_df)

        # 2. Prepare data for model input (SequenceDataPreparer + pad_collate_fn)
        model_input_batch = self._prepare_model_input_batch(processed_patient_df)
        
        # 3. Move batch to device
        batch_on_device = self._move_batch_to_device(model_input_batch)

        # 4. Get model prediction (logits)
        with torch.no_grad():
            logits = self.predictor_model(batch_on_device) # Shape: (1, seq_len, num_classes)

        # 5. Process logits to probabilities and classes
        probs_all_visits = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy() # (seq_len, num_classes)
        preds_all_visits = np.argmax(probs_all_visits, axis=-1) # (seq_len,)
        
        mask = model_input_batch['mask'].squeeze(0).cpu().numpy().astype(bool) # (seq_len,) boolean
        num_actual_visits = int(mask.sum())

        # Filter to actual visits using the mask
        probs_actual_visits = probs_all_visits[mask]
        preds_actual_visits = preds_all_visits[mask]

        visit_predictions_output = []
        for i in range(num_actual_visits):
            prediction_details = {f'prob_class_{j}': float(probs_actual_visits[i, j]) for j in range(probs_actual_visits.shape[1])}
            prediction_details['predicted_class'] = int(preds_actual_visits[i])
            # You could add original encounter_id here if you carry it through processed_patient_df
            # For now, it's just a list of predictions for the valid timesteps.
            visit_predictions_output.append(prediction_details)

        logger.info(f"Prediction complete. Output generated for {num_actual_visits} actual visits.")
        
        return {
            "patient_id": patient_id_val,
            "num_visits_processed": num_actual_visits,
            "visit_predictions": visit_predictions_output, # List of dicts for each valid visit
            "processed_model_input": model_input_batch # Return this for SHAP (it's on CPU from pad_collate_fn)
        }