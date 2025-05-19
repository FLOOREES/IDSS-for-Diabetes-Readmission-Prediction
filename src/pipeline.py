# src/pipeline.py
"""
pipeline.py
===========
Class-based workflow for the diabetes readmission project.
Orchestrates preprocessing, model training, and analysis.
"""

import logging
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import os
import json # For saving metrics
from pathlib import Path # Using Path for consistency

from typing import Tuple, Dict, Any, Optional, List
from torch.utils.data import DataLoader
from functools import partial

# Project imports (ensure these paths are correct relative to your project structure)
from .preprocessing.first_phase import FirstPhasePreprocessor
from .preprocessing.second_phase import SecondPhasePreprocessor
from .data_preparation import SequenceDataPreparer, PatientSequenceDataset, pad_collate_fn
from .modeling.model_builder import build_autoencoder_from_config
from .modeling import PredictionHead, PredictorModel, Seq2SeqAE 
from .training import AETrainer, PredictorTrainer
from .utils import setup_logging, load_artifact


logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """Helper class to serialize numpy types to JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Pipeline:
    """Orchestrates the end-to-end sequential modeling workflow."""

    def __init__(self, cfg: Any): # cfg is the imported config module
        """
        Initializes the pipeline with configuration.
        Sets up logging, device, and random seeds.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        setup_logging(log_file=self.cfg.LOG_FILE) # From utils
        np.random.seed(self.cfg.RANDOM_SEED)
        torch.manual_seed(self.cfg.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.RANDOM_SEED)
        
        logger.info(f"Pipeline initialized. Using device: {self.device}")

        # Attributes to store intermediate results/objects
        self.data_preparer: Optional[SequenceDataPreparer] = None


    def run(self) -> None:
        """Executes the full workflow."""
        logger.info("========== Workflow Started ==========")

        df_final = self._run_preprocessing_and_load()
        df_train, df_val, df_test = self._split_data(df_final)

        actual_ohe_columns_list: Optional[List[str]] = None
        if os.path.exists(self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH):
            logger.info(f"Loading OHE feature names from {self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH}")
            try:
                with open(self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH, 'r') as f:
                    actual_ohe_columns_list = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load OHE feature names from {self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH}: {e}", exc_info=True)
                actual_ohe_columns_list = None # Fallback or raise error

        self.data_preparer = SequenceDataPreparer(
            patient_id_col=self.cfg.PATIENT_ID_COL, timestamp_col=self.cfg.ENCOUNTER_ID_COL,
            target_col=self.cfg.TARGET_COL, numerical_features=self.cfg.NUMERICAL_FEATURES,
            ohe_columns=actual_ohe_columns_list, learned_emb_cols=self.cfg.LEARNED_EMB_COLS,
            precomputed_emb_cols=self.cfg.PRECOMPUTED_EMB_COLS, max_seq_length=self.cfg.MAX_SEQ_LENGTH,
            scaler_path=self.cfg.SCALER_PATH, logger=logger
        )
        
        train_loader, val_loader = self._prepare_dataloaders(self.data_preparer, df_train, df_val, self.cfg.AE_BATCH_SIZE)
        
        if not train_loader.dataset:
            logger.error("Training dataset is empty. Cannot proceed.")
            raise ValueError("Training dataset is empty, cannot get sample batch or train.")

        trained_ae = self._train_or_load_autoencoder(train_loader, val_loader)
        trained_predictor = self._train_or_load_predictor(trained_ae, train_loader, val_loader)

        self._run_outlier_detection(trained_ae, df_train, df_val, self.data_preparer)
        self._run_prediction(trained_predictor, df_test, self.data_preparer)

        logger.info("========== Workflow Finished ==========")

    def _run_preprocessing_and_load(self) -> pd.DataFrame:
        """
        Handles preprocessing if the final encoded data file does not exist.
        Otherwise, loads the existing file. Also ensures critical ID columns are present.
        """
        logger.info("--- Stage: Preprocessing and Data Loading ---") # Using the module-level logger from pipeline.py
        final_data_path = Path(self.cfg.FINAL_ENCODED_DATA_PATH)
        
        should_run_preprocessing = not final_data_path.exists()

        if should_run_preprocessing:
            logger.info(f"Preprocessed data not found at {final_data_path}. Running preprocessing phases.")
            
            # --- Phase 1 Preprocessing ---
            logger.info("Running Phase 1 Preprocessing...")
            df_raw = FirstPhasePreprocessor.load_data(self.cfg.RAW_DATA_PATH, self.cfg.MISSING_VALUES)
            phase1_processor = FirstPhasePreprocessor(
                drop_columns=self.cfg.DROP_COLUMNS,
                one_hot_columns=self.cfg.ONE_HOT_COLUMNS, # config.ONE_HOT_COLUMNS should be e.g. ['gender', 'admission_type_id']
                                                          # 'race' is handled internally by the preprocessor's OHE logic now after imputation
                ordinal_mappings=self.cfg.ORDINAL_MAPPINGS,
                treatment_columns=self.cfg.TREATMENT_COLUMNS,
                treatment_mapping=self.cfg.TREATMENT_MAPPING,
                missing_values_encoding=self.cfg.MISSING_VALUES, # This is {'?': pd.NA} for df.replace()
                
                # New paths for saved artifacts from config.py
                ohe_encoder_path=self.cfg.PHASE1_OHE_ENCODER_PATH,
                ohe_feature_names_path=self.cfg.PHASE1_OHE_FEATURE_NAMES_PATH,
                low_variance_cols_path=self.cfg.PHASE1_LOW_VAR_COLS_PATH,
                logger=logger # Pass the pipeline's logger (or a child logger)
            )
            
            # The transform method will now internally handle fitting and saving 
            # its components if they haven't been loaded and artifacts don't exist.
            df_phase1 = phase1_processor.transform(df_raw.copy())
            logger.info(f"Phase 1 complete. Shape: {df_phase1.shape}")

            # --- Phase 2 Preprocessing ---
            logger.info("Running Phase 2 Preprocessing...")
            phase2_processor = SecondPhasePreprocessor(
                diag_embeddings_path=self.cfg.DIAG_EMBEDDINGS_PATH,
                diag_label_encoder_path=self.cfg.DIAG_LABEL_ENCODER_PATH,
                label_encoders_path=self.cfg.LABEL_ENCODERS_PATH,
                icd9_hierarchy_path=self.cfg.ICD9_HIERARCHY_PATH,
                icd9_chapters_path=self.cfg.ICD9_CHAPTERS_PATH,
                spacy_model_name=self.cfg.SPACY_MODEL_NAME,
                label_encode_columns=self.cfg.LABEL_ENCODING,
                embedding_dim=getattr(self.cfg, 'DIAGNOSIS_EMBEDDING_DIM', 16), 
            )
            final_data_path.parent.mkdir(parents=True, exist_ok=True)
            df_processed = phase2_processor.transform(df_phase1.copy(), output_path=str(final_data_path))
            logger.info(f"Preprocessing complete. Data saved to: {final_data_path}. Shape: {df_processed.shape}")
        else: # Data exists
            logger.info(f"Using existing preprocessed data from: {final_data_path}")

        # --- Load (or use already loaded) final data ---
        try:
            if 'df_processed' in locals() and should_run_preprocessing:
                df_final = df_processed
            else:
                df_final = pd.read_csv(final_data_path, low_memory=False)
            logger.info(f"Successfully obtained final encoded data. Shape: {df_final.shape}")
        except Exception as e:
            logger.error(f"Failed to load final encoded data from {final_data_path}: {e}", exc_info=True)
            raise

        # --- Critical ID Check & Merge (Safety Net) ---
        # This part remains unchanged
        raw_encounter_col = getattr(self.cfg, 'RAW_ENCOUNTER_ID_COL_IN_RAW_FILE', 'encounter_id')
        raw_patient_col = getattr(self.cfg, 'RAW_PATIENT_ID_COL_IN_RAW_FILE', 'patient_nbr')

        if self.cfg.PATIENT_ID_COL not in df_final.columns or \
           self.cfg.ENCOUNTER_ID_COL not in df_final.columns:
            logger.warning(
                f"Patient ID ('{self.cfg.PATIENT_ID_COL}') or Encounter ID ('{self.cfg.ENCOUNTER_ID_COL}') "
                f"missing in the loaded data {final_data_path}. Attempting to merge from raw data as a fallback."
            )
            try:
                df_raw_ids = pd.read_csv(
                    self.cfg.RAW_DATA_PATH,
                    usecols=[raw_encounter_col, raw_patient_col] 
                )
                df_raw_ids.rename(columns={
                    raw_encounter_col: self.cfg.ENCOUNTER_ID_COL,
                    raw_patient_col: self.cfg.PATIENT_ID_COL
                }, inplace=True)

                df_final = df_final.reset_index(drop=True) 
                df_raw_ids = df_raw_ids.reset_index(drop=True)

                # Ensure the lengths match for simple assignment, or use a merge
                if len(df_final) == len(df_raw_ids):
                    if self.cfg.ENCOUNTER_ID_COL not in df_final.columns:
                        df_final[self.cfg.ENCOUNTER_ID_COL] = df_raw_ids[self.cfg.ENCOUNTER_ID_COL]
                    if self.cfg.PATIENT_ID_COL not in df_final.columns:
                        df_final[self.cfg.PATIENT_ID_COL] = df_raw_ids[self.cfg.PATIENT_ID_COL]
                    logger.info("IDs assigned back into the DataFrame based on index alignment.")
                else:
                    logger.warning(f"Length mismatch between df_final ({len(df_final)}) and df_raw_ids ({len(df_raw_ids)}). Cannot directly assign IDs. Fallback merge might be needed if IDs are still missing.")
                    # As a more robust fallback, you could consider a merge if index alignment isn't guaranteed
                    # For now, we'll rely on the original logic, but this is a potential point of failure if df_final was modified in ways that broke row order.
                    # However, since df_final is either loaded from CSV or the direct output of phase2, row order from df_raw should be preserved.
                    
            except Exception as e:
                logger.error(f"Failed to merge missing IDs from raw data: {e}. This might impact downstream tasks.", exc_info=True)
                raise ValueError("Critical ID columns missing and could not be merged.") from e
        
        df_final = df_final.reset_index(drop=True) 
        logger.info("Final data ready for splitting.")
        return df_final

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs patient-level train/validation/test splits."""
        logger.info("--- Splitting Data (Patient Level) ---")
        if self.cfg.PATIENT_ID_COL not in df.columns:
            raise ValueError(f"Patient ID column '{self.cfg.PATIENT_ID_COL}' not found.")

        patient_groups = df[self.cfg.PATIENT_ID_COL] # Groups for splitting (ensures patient integrity)
        all_indices = np.arange(len(df)) # Indices to be split

        # First, split into (train + validation) and test sets
        gss_test = GroupShuffleSplit(n_splits=1, test_size=self.cfg.TEST_SPLIT_SIZE, random_state=self.cfg.RANDOM_SEED)
        train_val_idx_pos, test_idx_pos = next(gss_test.split(all_indices, groups=patient_groups))
        
        df_test = df.iloc[test_idx_pos].copy()       # Create test DataFrame
        df_train_val = df.iloc[train_val_idx_pos].copy() # Create combined train/validation DataFrame

        # Now, split (train + validation) into train and validation sets
        # Adjust validation proportion because it's now a fraction of df_train_val, not the original df
        val_proportion_adjusted = self.cfg.VALIDATION_SPLIT_SIZE / (1 - self.cfg.TEST_SPLIT_SIZE)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_proportion_adjusted, random_state=self.cfg.RANDOM_SEED)
        
        train_val_patient_groups = df_train_val[self.cfg.PATIENT_ID_COL] # Groups from the df_train_val set
        train_val_indices_pos = np.arange(len(df_train_val)) # Indices for df_train_val

        train_idx_pos, val_idx_pos = next(gss_val.split(train_val_indices_pos, groups=train_val_patient_groups))
        
        df_train = df_train_val.iloc[train_idx_pos].copy() # Create train DataFrame
        df_val = df_train_val.iloc[val_idx_pos].copy()     # Create validation DataFrame

        logger.info(f"Train: {len(df_train)} ({df_train[self.cfg.PATIENT_ID_COL].nunique()} patients), "
                    f"Val: {len(df_val)} ({df_val[self.cfg.PATIENT_ID_COL].nunique()} patients), "
                    f"Test: {len(df_test)} ({df_test[self.cfg.PATIENT_ID_COL].nunique()} patients)")
        
        # Critical check for patient overlap between sets
        train_pats = set(df_train[self.cfg.PATIENT_ID_COL].unique())
        val_pats = set(df_val[self.cfg.PATIENT_ID_COL].unique())
        test_pats = set(df_test[self.cfg.PATIENT_ID_COL].unique())
        
        if not (train_pats.isdisjoint(val_pats) and \
                train_pats.isdisjoint(test_pats) and \
                val_pats.isdisjoint(test_pats)):
            logger.error("Patient overlap detected between splits! This is a critical issue.")
            # Consider raising an exception here if this is a hard stop condition:
            # raise ValueError("Patient overlap detected between splits! Critical data integrity issue.")
        else:
            logger.info("Patient overlap check passed. Splits are clean.")
            
        return df_train, df_val, df_test

    def _prepare_dataloaders(self, data_preparer: SequenceDataPreparer, df_train: pd.DataFrame, df_val: pd.DataFrame, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Prepares sequences and creates DataLoaders."""
        logger.info("--- Preparing Sequences and DataLoaders ---")
        data_preparer.fit_scaler(df_train) 
        train_feat_seqs, train_tgt_seqs, train_pids = data_preparer.transform(df_train)
        val_feat_seqs, val_tgt_seqs, val_pids = data_preparer.transform(df_val)

        train_dataset = PatientSequenceDataset(train_feat_seqs, train_tgt_seqs, train_pids)
        val_dataset = PatientSequenceDataset(val_feat_seqs, val_tgt_seqs, val_pids)

        num_workers = getattr(self.cfg, 'DATALOADER_NUM_WORKERS', 2) 
        pin_memory = getattr(self.cfg, 'DATALOADER_PIN_MEMORY', True)

        max_len_for_padding = getattr(self.cfg, 'MAX_SEQ_LENGTH', None)
        collate_fn_configured = partial(pad_collate_fn, enforced_max_len=max_len_for_padding)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_configured, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_configured, num_workers=num_workers, pin_memory=pin_memory)
        logger.info("Train and Validation DataLoaders created.")
        return train_loader, val_loader

    def _train_or_load_autoencoder(self, train_loader: DataLoader, val_loader: DataLoader) -> Seq2SeqAE:
        """Trains or loads the autoencoder model."""
        if self.cfg.TRAIN_AE:
            logger.info("--- Training Autoencoder ---")
            autoencoder = build_autoencoder_from_config(logger, self.device) 
            ae_trainer = AETrainer(
                model=autoencoder, train_loader=train_loader, val_loader=val_loader,
                optimizer_name=self.cfg.AE_OPTIMIZER, optimizer_params={'lr': self.cfg.AE_LEARNING_RATE, 'weight_decay': self.cfg.AE_WEIGHT_DECAY},
                scheduler_name='ReduceLROnPlateau', scheduler_params={'mode': 'min', 'factor': self.cfg.AE_SCHEDULER_FACTOR, 'patience': self.cfg.AE_SCHEDULER_PATIENCE},
                epochs=self.cfg.AE_EPOCHS, device=self.device, checkpoint_dir=self.cfg.MODELS_DIR,
                early_stopping_patience=self.cfg.AE_EARLY_STOPPING_PATIENCE, 
                gradient_clip_value=getattr(self.cfg, 'GRADIENT_CLIP_VALUE', 1.0),
            )
            ae_trainer.train()
            logger.info("--- Autoencoder Training Complete ---")
            return ae_trainer.model
        else:
            logger.info(f"--- Loading Pre-trained Autoencoder from {self.cfg.AE_MODEL_LOAD_PATH} ---")
            if not os.path.exists(self.cfg.AE_MODEL_LOAD_PATH):
                raise FileNotFoundError(f"AE model checkpoint not found: {self.cfg.AE_MODEL_LOAD_PATH}")
            autoencoder = build_autoencoder_from_config(logger, self.device) 
            checkpoint = load_artifact(self.cfg.AE_MODEL_LOAD_PATH, device=self.device)
            if 'model_state_dict' not in checkpoint: raise KeyError("Checkpoint missing 'model_state_dict'.")
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
            autoencoder.to(self.device).eval()
            logger.info("Pre-trained Autoencoder loaded.")
            return autoencoder

    def _train_or_load_predictor(self, trained_ae: Seq2SeqAE, train_loader: DataLoader, val_loader: DataLoader) -> PredictorModel:
        """Trains or loads the predictor model."""
        if self.cfg.TRAIN_PREDICTOR:
            logger.info("--- Training Predictor ---")
            encoder = trained_ae.get_encoder()
            for param in encoder.parameters(): 
                param.requires_grad = self.cfg.PREDICTOR_FINETUNE_ENCODER
            
            num_classes = getattr(self.cfg, 'NUM_CLASSES', 3) 
            prediction_head = PredictionHead(input_dim=self.cfg.HIDDEN_DIM, output_dim=num_classes)
            predictor_model = PredictorModel(encoder.to(self.device), prediction_head.to(self.device))

            predictor_trainer = PredictorTrainer(
                model=predictor_model, train_loader=train_loader, val_loader=val_loader,
                optimizer_name=self.cfg.PREDICTOR_OPTIMIZER,
                optimizer_params={'lr': self.cfg.PREDICTOR_LEARNING_RATE, 'weight_decay': self.cfg.PREDICTOR_WEIGHT_DECAY},
                scheduler_name='ReduceLROnPlateau',
                scheduler_params={'mode': 'min', 'factor': self.cfg.PREDICTOR_SCHEDULER_FACTOR, 'patience': self.cfg.PREDICTOR_SCHEDULER_PATIENCE},
                criterion_name='crossentropy', epochs=self.cfg.PREDICTOR_EPOCHS, device=self.device,
                checkpoint_dir=self.cfg.MODELS_DIR,
                early_stopping_patience=self.cfg.PREDICTOR_EARLY_STOPPING_PATIENCE,
                gradient_clip_value=getattr(self.cfg, 'GRADIENT_CLIP_VALUE', 1.0),
            )
            predictor_trainer.train()
            logger.info("--- Predictor Training Complete ---")
            return predictor_trainer.model
        else:
            logger.info(f"--- Loading Pre-trained PredictorModel from {self.cfg.PREDICTOR_MODEL_LOAD_PATH} ---")
            if not os.path.exists(self.cfg.PREDICTOR_MODEL_LOAD_PATH):
                raise FileNotFoundError(f"Predictor model checkpoint not found: {self.cfg.PREDICTOR_MODEL_LOAD_PATH}")

            temp_ae = build_autoencoder_from_config(logger, self.device) 
            encoder = temp_ae.get_encoder()
            num_classes = getattr(self.cfg, 'NUM_CLASSES', 3)
            prediction_head = PredictionHead(input_dim=self.cfg.HIDDEN_DIM, output_dim=num_classes)
            predictor_model = PredictorModel(encoder.to(self.device), prediction_head.to(self.device))

            checkpoint = load_artifact(self.cfg.PREDICTOR_MODEL_LOAD_PATH, device=self.device)
            if 'model_state_dict' not in checkpoint: raise KeyError("Checkpoint missing 'model_state_dict'.")
            predictor_model.load_state_dict(checkpoint['model_state_dict'])
            predictor_model.to(self.device).eval()
            logger.info("Pre-trained PredictorModel loaded.")
            return predictor_model

    def _run_outlier_detection(self, trained_ae: Seq2SeqAE, df_train: pd.DataFrame, df_full: pd.DataFrame, data_preparer: SequenceDataPreparer):
        """Performs outlier detection."""
        logger.info("--- Running Outlier Detection ---")
        from .analysis.outlier_detector import OutlierDetector 
        
        results_dir = Path(self.cfg.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        outlier_results_save_path = results_dir / f"outlier_results_{self.cfg.OUTLIER_MODE}.json"

        outlier_detector = OutlierDetector(
            data_preparer=data_preparer,
            isolation_forest_path=self.cfg.ISOLATION_FOREST_PATH, 
            device=self.device,
            sample_batch_for_build=None 
        )
        outlier_detector.ae_model = trained_ae.to(self.device)
        outlier_detector.encoder = trained_ae.get_encoder().to(self.device)

        if self.cfg.OUTLIER_MODE == 'visit':
            logger.info("Detecting visit-level outliers...")
            outlier_detector.calculate_and_set_visit_threshold(df_train, percentile=self.cfg.VISIT_ERROR_PERCENTILE)
            outlier_detector.detect_visit_outliers(df_full, results_save_path=str(outlier_results_save_path))
        elif self.cfg.OUTLIER_MODE == 'patient':
            logger.info("Detecting patient-level outliers...")
            outlier_detector.train_isolation_forest(
                df_train, save_path=self.cfg.ISOLATION_FOREST_PATH,
                n_estimators=self.cfg.IF_N_ESTIMATORS, contamination=self.cfg.IF_CONTAMINATION,
                random_state=self.cfg.RANDOM_SEED
            )
            outlier_detector.detect_patient_outliers(df_full, results_save_path=str(outlier_results_save_path))
        else:
            logger.info(f"Outlier detection mode '{self.cfg.OUTLIER_MODE}' not recognized or disabled. Skipping.")
        logger.info("--- Outlier Detection Complete ---")


    def _run_prediction(self, trained_predictor: PredictorModel, df_test: pd.DataFrame, data_preparer: SequenceDataPreparer):
        """Runs prediction on the test set and evaluates."""
        logger.info("--- Running Prediction & Evaluation (Multi-Class) ---")
        from .analysis.predictor_inference import Predictor as PredictorInference 
        
        results_dir = Path(self.cfg.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        evaluation_save_path = results_dir / "prediction_evaluation_metrics.json"

        predictor_evaluator = PredictorInference(
            model_path=None,
            model_config=None,
            data_preparer=data_preparer, 
            device=self.device,
            trained_model=trained_predictor.to(self.device) 
        )
        df_predictions = predictor_evaluator.predict_bulk(df_test)
        logger.info(f"Prediction complete. Results shape: {df_predictions.shape}")

        logger.info("--- Evaluating Predictions ---")
        evaluation_metrics = predictor_evaluator.evaluate(df_predictions, target_col=self.cfg.TARGET_COL)

        if "error" in evaluation_metrics:
            logger.error(f"Evaluation failed: {evaluation_metrics['error']}")
        else:
            logger.info(f"  Accuracy: {evaluation_metrics.get('accuracy', 'N/A'):.4f}")
            logger.info("  Classification Report:")
            
            if 'classification_report_dict' in evaluation_metrics: 
                logger.info(json.dumps(evaluation_metrics['classification_report_dict'], indent=4))
            elif 'predicted_class' in df_predictions and self.cfg.TARGET_COL in df_predictions:
                from sklearn.metrics import classification_report 
                labels_in_eval = evaluation_metrics.get('labels_in_evaluation', sorted(df_predictions[self.cfg.TARGET_COL].unique()))
                report_str = classification_report(
                    df_predictions[self.cfg.TARGET_COL].astype(int),
                    df_predictions['predicted_class'].astype(int),
                    labels=labels_in_eval,
                    zero_division=0
                )
                for line in report_str.split('\n'): logger.info(f"    {line}")

            logger.info(f"Saving evaluation metrics to: {evaluation_save_path}")
            try:
                with open(evaluation_save_path, 'w') as f:
                    json.dump(evaluation_metrics, f, indent=4, cls=NpEncoder)
                logger.info("Evaluation metrics saved.")
            except Exception as e:
                logger.error(f"Failed to save evaluation metrics: {e}")
        logger.info("--- Prediction & Evaluation Complete ---")