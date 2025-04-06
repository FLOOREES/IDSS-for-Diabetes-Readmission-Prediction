# src/main_workflow.py
import logging
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import os

from typing import Tuple, Dict, Any

# Project imports (assuming src is in PYTHONPATH or using relative imports carefully)
from config import ( # Your config file
    RAW_DATA_PATH, NO_MISSINGS_ENCODED_PATH, DIAG_EMBEDDINGS_PATH, DIAG_LABEL_ENCODER_PATH, LABEL_ENCODERS_PATH,
    ICD9_HIERARCHY_PATH, ICD9_CHAPTERS_PATH, SPACY_MODEL_NAME, MISSING_VALUES,
    DROP_COLUMNS, ONE_HOT_COLUMNS, ORDINAL_MAPPINGS, TREATMENT_COLUMNS,
    TREATMENT_MAPPING, LABEL_ENCODING,

    LOG_FILE, RANDOM_SEED, PATIENT_ID_COL, TEST_SPLIT_SIZE, VALIDATION_SPLIT_SIZE,
    OTHER_EMBEDDING_DIM, HIDDEN_DIM, NUM_RNN_LAYERS, DROPOUT, USE_GRU, USE_ATTENTION,
    ATTENTION_DIM, AE_BATCH_SIZE, AE_EPOCHS, PREDICTOR_EPOCHS,
    LEARNED_EMB_COLS, FINETUNE_DIAG_EMBEDDINGS, PRECOMPUTED_EMB_COLS,AE_OPTIMIZER,
    AE_LEARNING_RATE,AE_WEIGHT_DECAY, AE_SCHEDULER_FACTOR, AE_SCHEDULER_PATIENCE,
    AE_EARLY_STOPPING_PATIENCE, PREDICTOR_OPTIMIZER, PREDICTOR_LEARNING_RATE,
    MODELS_DIR, PREDICTOR_EARLY_STOPPING_PATIENCE, PREDICTOR_SCHEDULER_FACTOR,
    PREDICTOR_SCHEDULER_PATIENCE, PREDICTOR_WEIGHT_DECAY, PREDICTOR_FINETUNE_ENCODER,
    SCALER_PATH, ISOLATION_FOREST_PATH, IF_N_ESTIMATORS, IF_CONTAMINATION,
    OUTLIER_MODE, VISIT_ERROR_PERCENTILE,
    FINAL_ENCODED_DATA_PATH, ENCOUNTER_ID_COL, TARGET_COL, NUMERICAL_FEATURES,
    OHE_FEATURES_PREFIX, ICD9_HIERARCHY_PATH, ICD9_CHAPTERS_PATH,
    MAX_SEQ_LENGTH
)

# User needs to install these or ensure they are in requirements.txt
from preprocessing.first_phase import FirstPhasePreprocessor # Assuming these exist from user
from preprocessing.second_phase import SecondPhasePreprocessor # Assuming these exist from user

from data_preparation import SequenceDataPreparer, PatientSequenceDataset, pad_collate_fn
from modeling import (
    EmbeddingManager, EncoderRNN, AdditiveAttention, DecoderRNN,
    Seq2SeqAE, PredictionHead, PredictorModel
)
from training import AETrainer, PredictorTrainer
from analysis import OutlierDetector, Predictor
from utils import setup_logging, save_artifact, load_artifact
from torch.utils.data import DataLoader

# --- Setup ---
setup_logging(log_file=LOG_FILE)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ==============================================================================
# Workflow Control Flag
# ==============================================================================
TRAIN_AE = False # <<< SET THIS TO False TO LOAD SAVED AE MODEL >>>
# ==============================================================================


# ==============================================================================
# Workflow Functions (Keep previous functions: run_preprocessing, split_data, etc.)
# ==============================================================================
def run_preprocessing():
    """Runs Phase 1 and Phase 2 preprocessing."""
    logger.info("--- Running Preprocessing Phases ---")
    df_raw = FirstPhasePreprocessor.load_data(RAW_DATA_PATH, MISSING_VALUES)
    phase1_processor = FirstPhasePreprocessor(
        drop_columns=DROP_COLUMNS, one_hot_columns=ONE_HOT_COLUMNS,
        ordinal_mappings=ORDINAL_MAPPINGS, treatment_columns=TREATMENT_COLUMNS,
        treatment_mapping=TREATMENT_MAPPING, missing_values_encoding=MISSING_VALUES
    )
    df_phase1 = phase1_processor.transform(df_raw.copy())

    phase2_processor = SecondPhasePreprocessor(
        diag_embeddings_path=DIAG_EMBEDDINGS_PATH,
        diag_label_encoder_path=DIAG_LABEL_ENCODER_PATH,
        label_encoders_path=LABEL_ENCODERS_PATH,
        icd9_hierarchy_path=ICD9_HIERARCHY_PATH,
        icd9_chapters_path=ICD9_CHAPTERS_PATH,
        spacy_model_name=SPACY_MODEL_NAME,
        label_encode_columns=LABEL_ENCODING,
        embedding_dim=8, tsne_n_components=8
    )
    df_phase2 = phase2_processor.transform(df_phase1.copy(), output_path=NO_MISSINGS_ENCODED_PATH)
    logger.info("--- Preprocessing Complete ---")
    logger.info(f"Data shape after Phase 2: {df_phase2.shape}")


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Performs patient-level train/validation/test splits using DataFrame indices (revised)."""
    logger.info("--- Splitting Data (Patient Level - Revised Index Handling) ---")

    if PATIENT_ID_COL not in df.columns:
        raise ValueError(f"Patient ID column '{PATIENT_ID_COL}' not found in DataFrame for splitting.")

    patient_groups = df[PATIENT_ID_COL]
    all_indices = np.arange(len(df)) # Use integer positions directly

    logger.info(f"Total rows before split: {len(df)}")
    logger.info(f"Total unique patients: {patient_groups.nunique()}")

    # Split off test set first
    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED)
    # Pass integer positions as X and patient groups
    train_val_idx_pos, test_idx_pos = next(gss_test.split(all_indices, groups=patient_groups))

    # Create test DataFrame using iloc with integer positions
    df_test = df.iloc[test_idx_pos].copy()
    # Create temporary train+validation DataFrame using iloc
    df_train_val = df.iloc[train_val_idx_pos].copy()
    logger.info(f"Test set created: {len(df_test)} rows, {df_test[PATIENT_ID_COL].nunique()} patients.")

    # Split train/validation from the df_train_val set
    # Calculate the validation proportion relative to the train_val set size
    val_proportion_adjusted = VALIDATION_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_proportion_adjusted, random_state=RANDOM_SEED)
    train_val_patient_groups = df_train_val[PATIENT_ID_COL]
    train_val_indices_pos = np.arange(len(df_train_val)) # Integer positions for train_val

    # Split using integer positions and train_val patient groups
    train_idx_pos, val_idx_pos = next(gss_val.split(train_val_indices_pos, groups=train_val_patient_groups))

    # Create final train and validation DataFrames using iloc with integer positions
    df_train = df_train_val.iloc[train_idx_pos].copy()
    df_val = df_train_val.iloc[val_idx_pos].copy()

    # Log final counts
    logger.info(f"Train set created: {len(df_train)} rows, {df_train[PATIENT_ID_COL].nunique()} patients.")
    logger.info(f"Validation set created: {len(df_val)} rows, {df_val[PATIENT_ID_COL].nunique()} patients.")
    logger.info("--- Data Splitting Complete ---")

    # Sanity check: Ensure no patient overlap between sets
    train_pats = set(df_train[PATIENT_ID_COL].unique())
    val_pats = set(df_val[PATIENT_ID_COL].unique())
    test_pats = set(df_test[PATIENT_ID_COL].unique())
    if train_pats.intersection(val_pats): logger.warning("Overlap found between Train and Validation patients!")
    if train_pats.intersection(test_pats): logger.warning("Overlap found between Train and Test patients!")
    if val_pats.intersection(test_pats): logger.warning("Overlap found between Validation and Test patients!")

    return df_train, df_val, df_test


def prepare_dataloaders(
    data_preparer: SequenceDataPreparer,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Prepares sequences and creates DataLoaders."""
    logger.info("--- Preparing Sequences and DataLoaders ---")
    # Fit scaler on training data
    data_preparer.fit_scaler(df_train)

    # Transform data into sequences
    train_feat_seqs, train_tgt_seqs, train_pids = data_preparer.transform(df_train)
    val_feat_seqs, val_tgt_seqs, val_pids = data_preparer.transform(df_val)

    # Create datasets
    train_dataset = PatientSequenceDataset(train_feat_seqs, train_tgt_seqs, train_pids)
    val_dataset = PatientSequenceDataset(val_feat_seqs, val_tgt_seqs, val_pids)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=2, pin_memory=True)

    logger.info("Train and Validation DataLoaders created.")
    logger.info("--- Sequence Preparation Complete ---")
    return train_loader, val_loader


def build_autoencoder_from_config(sample_batch: Dict) -> Seq2SeqAE:
    """ Helper function to instantiate the AE model based on config. """
    logger.info("Building AE model architecture from config...")
    # Define Embedding Manager Config
    learned_emb_config = {
        col: (vocab_size, OTHER_EMBEDDING_DIM)
        for col, vocab_size in LEARNED_EMB_COLS.items()
    }
    precomputed_emb_config = {
        col: (DIAG_EMBEDDINGS_PATH, FINETUNE_DIAG_EMBEDDINGS)
        for col in PRECOMPUTED_EMB_COLS
    }
    embedding_manager = EmbeddingManager(learned_emb_config, precomputed_emb_config, device)
    total_emb_dim = embedding_manager.get_total_embedding_dim()
    num_ohe_features = sample_batch['num_ohe'].shape[-1]
    logger.info(f"Determined Num OHE Features for build: {num_ohe_features}")
    encoder_input_dim = num_ohe_features + total_emb_dim

    encoder = EncoderRNN(
        num_ohe_features=num_ohe_features, embedding_manager=embedding_manager,
        hidden_dim=HIDDEN_DIM, n_layers=NUM_RNN_LAYERS, dropout=DROPOUT,
        use_gru=USE_GRU, use_attention=USE_ATTENTION
    )
    decoder = DecoderRNN(
        reconstruction_dim=encoder_input_dim, encoder_hidden_dim=HIDDEN_DIM,
        decoder_hidden_dim=HIDDEN_DIM, n_layers=NUM_RNN_LAYERS, dropout=DROPOUT,
        use_gru=USE_GRU, use_attention=USE_ATTENTION, attention_dim=ATTENTION_DIM
    )
    autoencoder = Seq2SeqAE(encoder, decoder)
    logger.info("AE model architecture built.")
    return autoencoder


def train_autoencoder(train_loader: DataLoader, val_loader: DataLoader) -> Seq2SeqAE:
    """Trains the sequence autoencoder."""
    logger.info("--- Training Autoencoder ---")
    # 1. Instantiate Model Components using helper
    sample_batch = next(iter(train_loader))
    autoencoder = build_autoencoder_from_config(sample_batch)

    # 2. Instantiate Trainer
    ae_trainer = AETrainer(
        model=autoencoder, train_loader=train_loader, val_loader=val_loader,
        optimizer_name=AE_OPTIMIZER, optimizer_params={'lr': AE_LEARNING_RATE, 'weight_decay': AE_WEIGHT_DECAY},
        scheduler_name='ReduceLROnPlateau', scheduler_params={'mode': 'min', 'factor': AE_SCHEDULER_FACTOR, 'patience': AE_SCHEDULER_PATIENCE},
        epochs=AE_EPOCHS, device=device, checkpoint_dir=MODELS_DIR,
        early_stopping_patience=AE_EARLY_STOPPING_PATIENCE, gradient_clip_value=1.0
    )
    # 3. Run Training
    ae_trainer.train()
    logger.info("--- Autoencoder Training Complete ---")
    return ae_trainer.model # Return the trained model (best weights loaded)

def load_autoencoder(model_path: str, sample_batch: Dict) -> Seq2SeqAE:
    """Loads a pre-trained autoencoder model."""
    logger.info(f"--- Loading Pre-trained Autoencoder from {model_path} ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AE model checkpoint not found at: {model_path}")

    # 1. Rebuild the model architecture using the *exact same config*
    autoencoder = build_autoencoder_from_config(sample_batch)

    # 2. Load the state dictionary
    try:
        checkpoint = load_artifact(model_path, device=device)
        # Ensure checkpoint contains the state dict
        if 'model_state_dict' not in checkpoint:
             raise KeyError("Checkpoint dictionary does not contain 'model_state_dict'.")

        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.to(device) # Ensure model is on correct device
        autoencoder.eval() # Set to evaluation mode
        logger.info("Pre-trained Autoencoder loaded successfully.")
        return autoencoder
    except Exception as e:
        logger.error(f"Failed to load AE model state_dict from {model_path}: {e}", exc_info=True)
        raise

def train_predictor(
    trained_ae: Seq2SeqAE,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> PredictorModel:
    """Trains the prediction model using the pre-trained encoder."""
    logger.info("--- Training Predictor ---")

    # 1. Get Pre-trained Encoder
    encoder = trained_ae.get_encoder()
    if PREDICTOR_FINETUNE_ENCODER:
        logger.info("Encoder will be fine-tuned.")
        for param in encoder.parameters():
            param.requires_grad = True
    else:
        logger.info("Encoder weights will be frozen.")
        for param in encoder.parameters():
            param.requires_grad = False

    # 2. Instantiate Prediction Head
    # Assuming binary classification for readmitted (output_dim=1 for BCEWithLogitsLoss)
    # Or output_dim=3 if predicting NO/<30/>30 with NLLLoss/CrossEntropy
    num_classes = 1 # For BCEWithLogitsLoss
    prediction_head = PredictionHead(input_dim=HIDDEN_DIM, output_dim=num_classes)

    predictor_model = PredictorModel(encoder, prediction_head)

    # 3. Instantiate Trainer
    predictor_trainer = PredictorTrainer(
        model=predictor_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=PREDICTOR_OPTIMIZER,
        optimizer_params={'lr': PREDICTOR_LEARNING_RATE, 'weight_decay': PREDICTOR_WEIGHT_DECAY},
        scheduler_name='ReduceLROnPlateau',
        scheduler_params={'mode': 'min', 'factor': PREDICTOR_SCHEDULER_FACTOR, 'patience': PREDICTOR_SCHEDULER_PATIENCE},
        criterion_name='bce', # Use 'bce' with BCEWithLogitsLoss
        epochs=PREDICTOR_EPOCHS,
        device=device,
        checkpoint_dir=MODELS_DIR,
        early_stopping_patience=PREDICTOR_EARLY_STOPPING_PATIENCE,
        gradient_clip_value=1.0
    )
     # Optional: Configure different learning rates for encoder/head
    # predictor_trainer.configure_optimizers(finetune_encoder=PREDICTOR_FINETUNE_ENCODER, encoder_lr_factor=0.1)


    # 4. Run Training
    predictor_trainer.train()

    # 5. Save final predictor model
    # save_artifact(predictor_trainer.model.state_dict(), PREDICTOR_MODEL_PATH)
    logger.info("--- Predictor Training Complete ---")
    return predictor_trainer.model


def run_outlier_detection(trained_ae: Seq2SeqAE, df_train: pd.DataFrame, df_full: pd.DataFrame, data_preparer: SequenceDataPreparer):
    """Performs outlier detection using the trained AE/Encoder."""
    logger.info("--- Running Outlier Detection ---")
    outlier_detector = OutlierDetector(
        ae_model_path=None, # Pass model directly if available
        encoder_model_path=None,
        encoder_config=None, # Need to store/pass config if loading encoder state_dict
        data_preparer=data_preparer,
        isolation_forest_path=ISOLATION_FOREST_PATH, # Path for saving/loading
        device=device
    )
    # Pass the trained model objects directly
    outlier_detector.ae_model = trained_ae.to(device)
    outlier_detector.encoder = trained_ae.get_encoder().to(device)


    if OUTLIER_MODE == 'visit':
        logger.info("Detecting visit-level outliers...")
        outlier_detector.calculate_and_set_visit_threshold(df_train, percentile=VISIT_ERROR_PERCENTILE)
        df_outliers = outlier_detector.detect_visit_outliers(df_full)
        logger.info(f"Visit outlier detection complete. Results shape: {df_outliers.shape}")
        print(df_outliers[['reconstruction_error', 'is_outlier_visit']].head())
        print(df_outliers['is_outlier_visit'].value_counts())

    elif OUTLIER_MODE == 'patient':
        logger.info("Detecting patient-level outliers...")
        outlier_detector.train_isolation_forest(
            df_train,
            save_path=ISOLATION_FOREST_PATH,
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=RANDOM_SEED
        )
        df_patient_outliers = outlier_detector.detect_patient_outliers(df_full)
        logger.info(f"Patient outlier detection complete. Results shape: {df_patient_outliers.shape}")
        print(df_patient_outliers.head())
        print(df_patient_outliers['is_outlier_patient'].value_counts())

    else:
        logger.error(f"Invalid OUTLIER_MODE: {OUTLIER_MODE}")

    logger.info("--- Outlier Detection Complete ---")


def run_prediction(trained_predictor: PredictorModel, df_test: pd.DataFrame, data_preparer: SequenceDataPreparer):
    """Runs prediction on the test set."""
    logger.info("--- Running Prediction ---")
    predictor = Predictor(
        model_path=None, # Pass model directly
        model_config=None, # Not needed if passing model
        data_preparer=data_preparer,
        device=device
    )
    predictor.model = trained_predictor.to(device) # Assign loaded/trained model

    # Example: Predict on the entire test set
    df_predictions = predictor.predict_bulk(df_test)
    logger.info(f"Prediction complete. Results shape: {df_predictions.shape}")
    print(df_predictions[[TARGET_COL, 'readmission_prob']].head()) # Adjust column name based on output

    # Add evaluation logic here (e.g., calculate AUC on df_predictions)
    logger.info("--- Prediction Complete ---")


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    logger.info("========== Workflow Started ==========")

    # 1. Run Preprocessing (Optional)
    # run_preprocessing() # Uncomment to force run
    if not os.path.exists(FINAL_ENCODED_DATA_PATH):
        logger.info(f"{FINAL_ENCODED_DATA_PATH} not found. Running preprocessing...")
        run_preprocessing()
    else:
        logger.info(f"Using pre-existing encoded data: {FINAL_ENCODED_DATA_PATH}")

    # 2. Load Data
    try:
        df_final = pd.read_csv(FINAL_ENCODED_DATA_PATH, low_memory=False)
        logger.info(f"Loaded final encoded data. Shape: {df_final.shape}")
        # Merge IDs if necessary (keep your existing logic if needed)
        if PATIENT_ID_COL not in df_final.columns or ENCOUNTER_ID_COL not in df_final.columns:
             logger.warning("Patient/Encounter IDs missing. Merging from raw data.")
             df_raw_ids = pd.read_csv(RAW_DATA_PATH, usecols=['encounter_id', 'patient_nbr'])
             # Ensure indices align before assigning
             df_final = df_final.reset_index(drop=True)
             df_raw_ids = df_raw_ids.reset_index(drop=True)
             df_final[ENCOUNTER_ID_COL] = df_raw_ids['encounter_id']
             df_final[PATIENT_ID_COL] = df_raw_ids['patient_nbr']
             logger.info("IDs merged back.")
        df_final.reset_index(drop=True, inplace=True) # Ensure clean index
        logger.info("DataFrame index reset.")
    except Exception as e:
        logger.error(f"Failed to load final encoded data: {e}", exc_info=True); exit()

    # 3. Split Data
    df_train, df_val, df_test = split_data(df_final)

    # 4. Prepare DataLoaders
    data_preparer = SequenceDataPreparer(
        patient_id_col=PATIENT_ID_COL, timestamp_col=ENCOUNTER_ID_COL, target_col=TARGET_COL,
        numerical_features=NUMERICAL_FEATURES, ohe_feature_prefixes=OHE_FEATURES_PREFIX,
        learned_emb_cols=LEARNED_EMB_COLS, precomputed_emb_cols=PRECOMPUTED_EMB_COLS,
        max_seq_length=MAX_SEQ_LENGTH, scaler_path=SCALER_PATH
    )
    # Need a sample batch to determine dims for loading AE if not training
    # Prepare loaders *before* deciding whether to train or load AE
    train_loader, val_loader = prepare_dataloaders(data_preparer, df_train, df_val, AE_BATCH_SIZE)
    sample_batch_for_build = next(iter(train_loader)) # Get a sample batch

    # 5. Train or Load Autoencoder
    if TRAIN_AE:
        trained_ae = train_autoencoder(train_loader, val_loader)
    else:
        # Ensure the path is defined correctly in config.py
        ae_model_load_path = os.path.join(MODELS_DIR, f"autoencoder_best.pth") # Path to best saved model
        trained_ae = load_autoencoder(ae_model_load_path, sample_batch_for_build)

    # 6. Train Predictor (using the loaded or newly trained AE)
    # Consider using PREDICTOR_BATCH_SIZE if different from AE_BATCH_SIZE
    # train_loader_pred, val_loader_pred = prepare_dataloaders(data_preparer, df_train, df_val, PREDICTOR_BATCH_SIZE)
    trained_predictor = train_predictor(trained_ae, train_loader, val_loader) # Reuse AE loaders

    # 7. Run Outlier Detection (Example: on validation set)
    logger.info("Running outlier detection on validation set for demonstration.")
    run_outlier_detection(trained_ae, df_train, df_val, data_preparer) # Use df_val for OD eval

    # 8. Run Prediction (on test set)
    run_prediction(trained_predictor, df_test, data_preparer)

    logger.info("========== Workflow Finished ==========")