import pandas as pd

RAW_DATA_PATH = "data/diabetic_data.csv"
MID_PROCESSING_PATH = "data/diabetic_data_mid.csv"
NO_MISSINGS_ENCODED_PATH="data/diabetic_data_no_na_diag.csv"
FINAL_ENCODED_DATA_PATH="data/diabetic_data_no_na_diag.csv"

DIAG_EMBEDDINGS_PATH = "data/diag_embeddings.npy"
DIAG_LABEL_ENCODER_PATH = "data/diag_label_encoder.json"
LABEL_ENCODERS_PATH = "data/diabetic_data_label_encoders.json"
ICD9_HIERARCHY_PATH = "data/icd9Hierarchy.json"
ICD9_CHAPTERS_PATH = "data/icd9Chapters.json"
SPACY_MODEL_NAME = "en_core_sci_md"

MISSING_VALUES = {'?': pd.NA}

DROP_COLUMNS = [
    'weight',
    'payer_code', 'medical_specialty',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
]

ONE_HOT_COLUMNS = [
    'gender', 'admission_type_id', 
]

ORDINAL_MAPPINGS = {
    'age': {
        '[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4,
        '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8,
        '[80-90)': 9, '[90-100)': 10
    },
    'readmitted': {
        'NO': 0, '>30': 1, '<30': 2
    }
}

TREATMENT_MAPPING = {
    'No': 0,
    'Down': 1,
    'Steady': 2,
    'Up': 3
}

TREATMENT_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
    'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
    'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

LABEL_ENCODING= ['discharge_disposition_id', 'admission_source_id']

# ----------------------------

# src/config.py
"""
Central configuration file for the project.
Define paths, hyperparameters, and feature lists here.
"""
import pandas as pd
import os

# --- Paths ---
# Suggest using os.path.join for better cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root assuming src is top-level
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Input Data
RAW_DATA_PATH = os.path.join(DATA_DIR, "diabetic_data.csv")

# Embeddings and Mappings
DIAG_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "diag_embeddings.npy")
DIAG_MAPPING_PATH = os.path.join(DATA_DIR, "diag_mapping.json") # Renamed from label_encoder
OTHER_MAPPINGS_PATH = os.path.join(DATA_DIR, "other_mappings.json") # Renamed from label_encoders
ICD9_HIERARCHY_PATH = os.path.join(DATA_DIR, "icd9Hierarchy.json")
ICD9_CHAPTERS_PATH = os.path.join(DATA_DIR, "icd9Chapters.json")

# Model Artifacts
AE_MODEL_PATH = os.path.join(MODELS_DIR, "autoencoder.pth")
ENCODER_MODEL_PATH = os.path.join(MODELS_DIR, "encoder.pth") # Optionally save separately
PREDICTOR_MODEL_PATH = os.path.join(MODELS_DIR, "predictor.pth")
ISOLATION_FOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl") # Path to save the fitted scaler

# Log file
LOG_FILE = os.path.join(LOGS_DIR, "workflow.log")

DIAG_COLS = ['diag_1', 'diag_2', 'diag_3'] # Columns for diagnosis embedding lookup

# Embedding Dimensions
DIAG_EMBEDDING_DIM = 16 # Dimension *after* t-SNE if used, else Spacy dim
TSNE_N_COMPONENTS = 16 # Make consistent with DIAG_EMBEDDING_DIM if TSNE used
OTHER_EMBEDDING_DIM = 10 # Example dimension for learned embeddings

# --- Feature Definitions for Sequence Model ---
# Define these based on the output columns AFTER preprocessing Phase 2
# Important: These names must match columns in the final DataFrame fed to SequenceDataPreparer
PATIENT_ID_COL = 'patient_nbr' # Use original ID if dropped during processing
ENCOUNTER_ID_COL = 'encounter_id'
TARGET_COL = 'readmitted' # The name of the target column AFTER potential encoding

# Example feature lists (adjust based on actual columns post-preprocessing)
NUMERICAL_FEATURES = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'age'] # Ensure 'age' is numerical here
OHE_FEATURES_PREFIX = ['gender', 'admission_type_id', 'race'] # Prefixes from Phase 1 OHE

# Categorical features needing LEARNED embeddings
LEARNED_EMB_COLS = {
    'discharge_disposition_id': 26, # Example Vocab Size (get from data)
    'admission_source_id': 17       # Example Vocab Size
}

# Categorical features using PRE-COMPUTED embeddings (Diagnosis codes)
PRECOMPUTED_EMB_COLS = DIAG_COLS # These use DIAG_MAPPING_PATH and DIAG_EMBEDDINGS_PATH

# --- Data Preparation Config ---
MAX_SEQ_LENGTH = 50 # Max number of visits per patient sequence

# --- Model Hyperparameters ---
# General
HIDDEN_DIM = 128
NUM_RNN_LAYERS = 1 # For Encoder and Decoder
DROPOUT = 0.2
USE_GRU = False # Use LSTM if False
USE_ATTENTION = True
ATTENTION_DIM = 64 # For Additive Attention mechanism

# Embeddings
FINETUNE_DIAG_EMBEDDINGS = True # Whether to train pre-computed diag embeddings

# --- Training Hyperparameters ---
# AE Training
AE_EPOCHS = 20 # Adjust based on convergence
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-6
AE_OPTIMIZER = 'adam'
AE_EARLY_STOPPING_PATIENCE = 5
AE_SCHEDULER_PATIENCE = 3
AE_SCHEDULER_FACTOR = 0.1

# Predictor Training
PREDICTOR_EPOCHS = 30
PREDICTOR_BATCH_SIZE = 64
PREDICTOR_LEARNING_RATE = 5e-4 # Often lower for fine-tuning
PREDICTOR_WEIGHT_DECAY = 1e-6
PREDICTOR_OPTIMIZER = 'adam'
PREDICTOR_EARLY_STOPPING_PATIENCE = 7
PREDICTOR_SCHEDULER_PATIENCE = 4
PREDICTOR_SCHEDULER_FACTOR = 0.2
PREDICTOR_FINETUNE_ENCODER = True # Fine-tune encoder during prediction training?

# --- Analysis Config ---
# Outlier Detection
OUTLIER_MODE = 'visit' # 'visit' or 'patient'
VISIT_ERROR_PERCENTILE = 98.0 # For visit-level outlier threshold
# Isolation Forest parameters (if using patient mode)
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 'auto' # Or a float like 0.05

# --- Runtime ---
VALIDATION_SPLIT_SIZE = 0.15 # Fraction of patients for validation
TEST_SPLIT_SIZE = 0.15 # Fraction of patients for testing (after train/val)
RANDOM_SEED = 42

# Add checks for directory existence
for dir_path in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)