# main_pytorch_outlier.py
import pandas as pd
import numpy as np
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports from project ---
from config import ( # Your config file
    RAW_DATA_PATH, NO_MISSINGS_ENCODED_PATH, DIAG_EMBEDDINGS_PATH, DIAG_LABEL_ENCODER_PATH, LABEL_ENCODERS_PATH,
    ICD9_HIERARCHY_PATH, ICD9_CHAPTERS_PATH, SPACY_MODEL_NAME, MISSING_VALUES,
    DROP_COLUMNS, ONE_HOT_COLUMNS, ORDINAL_MAPPINGS, TREATMENT_COLUMNS,
    TREATMENT_MAPPING
)
# Assuming LABEL_ENCODE_COLUMNS_EXAMPLE is defined or loaded
LABEL_ENCODING= ['discharge_disposition_id', 'admission_source_id']

from preprocessing import FirstPhasePreprocessor
from preprocessing.second_phase import SecondPhasePreprocessor
from recurrent_data import SequenceDataPreparer 
from outlier_detection import SequenceOutlierDetectorPyTorch 

if __name__ == "__main__":
    # --- Preprocessing  ---
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

    # --- Feature Selection (same logic as before) ---
    logger.info("--- Selecting Features ---")
    exclude_cols = ['encounter_id', 'patient_nbr', 'readmitted']
    exclude_cols.extend(phase2_processor.label_encode_columns) # Original names
    # Decide whether to keep original diag codes if needed for other analysis
    # For the model, we use the label encoded 'diag_1', 'diag_2', 'diag_3'
    # Ensure these label encoded columns are NOT in exclude_cols if they are features

    numerical_cols = df_phase2.select_dtypes(include=np.number).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    categorical_cols = [col for col in df_phase2.columns if col.startswith(tuple(ONE_HOT_COLUMNS)) or col in TREATMENT_COLUMNS]
    # Add the final names of label-encoded columns (assuming they are just the original names after transform)
    categorical_cols.extend(phase2_processor.label_encode_columns)
    categorical_cols.extend(['diag_1','diag_2','diag_3']) # Add encoded diag columns

    all_features = list(set(numerical_cols + categorical_cols))
    all_features = [col for col in all_features if col in df_phase2.columns and col not in exclude_cols] # Final check

    numerical_features_final = [col for col in numerical_cols if col in all_features]
    categorical_features_final = [col for col in categorical_cols if col in all_features] # These are already numeric (encoded)

    logger.info(f"Identified Numerical Features ({len(numerical_features_final)}): {numerical_features_final[:5]}...")
    logger.info(f"Identified Categorical Features ({len(categorical_features_final)}): {categorical_features_final[:5]}...") # Already numeric labels/OHE
    logger.info(f"Total Features for Sequence Model: {len(all_features)}")

    # --- Outlier Detection Setup (PyTorch) ---
    logger.info("--- Setting up PyTorch Outlier Detection ---")

    # Use PyTorch Data Preparer
    seq_preparer = SequenceDataPreparer(
        patient_id_col='patient_nbr',
        sort_col='encounter_id',
        numerical_features=numerical_features_final,
        categorical_features=categorical_features_final,
        max_seq_length=30 # Optional: Set max length for truncation, else uses max in batch
    )

    # Use PyTorch Outlier Detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outlier_detector = SequenceOutlierDetectorPyTorch(
        data_preparer=seq_preparer,
        patient_id_col='patient_nbr',
        # input_dim will be set during fit
        hidden_dim=128,          # Example: Increase hidden dim
        encoder_layers=1,
        decoder_layers=1,
        attention_dim=64,        # Example: Increase attention dim
        dropout=0.15,
        use_gru=False,           # Use LSTM
        epochs=10,               # Low epochs for demo
        batch_size=64,          # Smaller batch size might be needed
        learning_rate=0.0005,    # Example: Lower LR
        optimizer_name='adam',
        weight_decay=1e-6,
        validation_split=0.15,
        early_stopping_patience=5, # Increased patience
        lr_scheduler_patience=3,
        lr_scheduler_factor=0.2,
        device=device,
        error_percentile_threshold=98.0
    )

    # --- Fit the detector ---
    # sample_df = df_phase2.sample(n=20000, random_state=42)
    # outlier_detector.fit(sample_df)
    outlier_detector.fit(df_phase2) # Use full data

    # --- Detect outliers ---
    logger.info("--- Detecting Outliers (PyTorch) ---")
    results_df, patient_embeddings = outlier_detector.detect_outliers(df_phase2, return_embeddings=True)

    logger.info("Outlier Detection Results DataFrame:")
    logger.info(results_df.head())
    logger.info(results_df['is_outlier_visit'].value_counts())
    if 'reconstruction_error' in results_df.columns:
         logger.info(f"Mean reconstruction error: {results_df['reconstruction_error'].mean():.4f}")
         logger.info(f"Median reconstruction error: {results_df['reconstruction_error'].median():.4f}")


    logger.info("Patient Embeddings DataFrame:")
    logger.info(patient_embeddings.head())

    # --- (Optional) Save/Load Example ---
    # outlier_detector.save_model("pytorch_ae_outlier_model.pth")
    # # To load later:
    # detector_loaded = SequenceOutlierDetectorPyTorch(...) # Initialize with same preparer and basic config
    # detector_loaded.load_model("pytorch_ae_outlier_model.pth")
    # results_loaded = detector_loaded.detect_outliers(df_phase2)

    # --- Next Steps ---
    # - Analyze results_df
    # - Apply IsolationForest to patient_embeddings
    # - Use the encoder part of the loaded model for prediction
    logger.info("--- PyTorch Example Usage Complete ---")