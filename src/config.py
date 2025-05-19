# src/config.py
"""
Configuration file for the Diabetes Readmission Project.
All paths, hyperparameters, and constants are defined here.
"""
import os
import pandas as pd

# ==============================================================================
# 1. DIRECTORY PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "static/results")

for dir_path in [DATA_DIR, LOGS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==============================================================================
# 2. DATA FILE PATHS & NAMES
# ==============================================================================
RAW_DATA_PATH = os.path.join(DATA_DIR, "diabetic_data.csv")
# MID_PROCESSING_PATH = os.path.join(DATA_DIR, "diabetic_data_mid.csv") # Potentially unused by current pipeline.py
FINAL_ENCODED_DATA_PATH = os.path.join(DATA_DIR, "diabetic_data_no_na_diag.csv")

DIAG_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "diag_embeddings.npy")
DIAG_LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "diag_label_encoder.json")
LABEL_ENCODERS_PATH = os.path.join(DATA_DIR, "diabetic_data_label_encoders.json")
ICD9_HIERARCHY_PATH = os.path.join(DATA_DIR, "icd9Hierarchy.json")
ICD9_CHAPTERS_PATH = os.path.join(DATA_DIR, "icd9Chapters.json")
SPACY_MODEL_NAME = "en_core_sci_md"

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ISOLATION_FOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")

AE_MODEL_LOAD_PATH = os.path.join(MODELS_DIR, "autoencoder_best.pth")
PREDICTOR_MODEL_LOAD_PATH = os.path.join(MODELS_DIR, "predictor_best.pth")
LOG_FILE = os.path.join(LOGS_DIR, "workflow.log")

# --- Preprocessing Phase 1 Artifacts ---
PHASE1_ARTIFACTS_DIR = os.path.join(MODELS_DIR, "phase1_preprocessor") # A directory for these
PHASE1_OHE_ENCODER_PATH = os.path.join(PHASE1_ARTIFACTS_DIR, "ohe_encoder.joblib")
PHASE1_OHE_FEATURE_NAMES_PATH = os.path.join(PHASE1_ARTIFACTS_DIR, "ohe_feature_names.json")
PHASE1_LOW_VAR_COLS_PATH = os.path.join(PHASE1_ARTIFACTS_DIR, "low_variance_cols.json")

# ==============================================================================
# 3. PREPROCESSING DIRECTIVES & COLUMN DEFINITIONS
# ==============================================================================
MISSING_VALUES = {'?': pd.NA}

DROP_COLUMNS = [
    'weight', 'payer_code', 'medical_specialty',
    'max_glu_serum', 'A1Cresult',
    'change', 'diabetesMed' # These were in your original full config
]

RAW_ENCOUNTER_ID_COL_IN_RAW_FILE = 'encounter_id'
RAW_PATIENT_ID_COL_IN_RAW_FILE = 'patient_nbr'

PATIENT_ID_COL = 'patient_nbr'
ENCOUNTER_ID_COL = 'encounter_id'
TARGET_COL = 'readmitted' # Name of the target column AFTER ordinal encoding

DIAG_COLS = ['diag_1', 'diag_2', 'diag_3']

ONE_HOT_COLUMNS = [
    'gender', 'admission_type_id', 'race'
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

TREATMENT_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]
TREATMENT_MAPPING = {
    'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3
}

LABEL_ENCODING = [
    'discharge_disposition_id', 'admission_source_id'
]

# ==============================================================================
# 4. FEATURE DEFINITIONS FOR SequenceDataPreparer
# ==============================================================================
# These names MUST match the column names in the DataFrame AFTER all preprocessing
# as expected by SequenceDataPreparer based on the original working config.

NUMERICAL_FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses',
    'age' # CRITICAL FIX: Reverted from 'age_ordinal' to 'age'
]

LEARNED_EMB_COLS = {
    # CRITICAL FIX: Reverted keys from 'xxx_label' to original names
    'discharge_disposition_id': 26,
    'admission_source_id': 17
}

# CRITICAL FIX: Reverted from using 'xxx_label' to using DIAG_COLS directly
PRECOMPUTED_EMB_COLS = DIAG_COLS # ['diag_1', 'diag_2', 'diag_3']

# ==============================================================================
# 5. DATA PREPARATION & LOADER CONFIG
# ==============================================================================
MAX_SEQ_LENGTH = 50
DATALOADER_NUM_WORKERS = 2
DATALOADER_PIN_MEMORY = True


# ==============================================================================
# 6. MODEL ARCHITECTURE HYPERPARAMETERS
# ==============================================================================
HIDDEN_DIM = 128
NUM_RNN_LAYERS = 1
DROPOUT = 0.2
USE_GRU = False
USE_ATTENTION = True
ATTENTION_DIM = 64

DIAGNOSIS_EMBEDDING_DIM = 16
DIAGNOSIS_TSNE_COMPONENTS = 16
OTHER_EMBEDDING_DIM = 10
FINETUNE_DIAG_EMBEDDINGS = True
NUM_CLASSES = 3
MODEL_INPUT_NUM_OHE_DIM = 26

# ==============================================================================
# 7. TRAINING HYPERPARAMETERS
# ==============================================================================
GRADIENT_CLIP_VALUE = 1.0

AE_EPOCHS = 20
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-6
AE_OPTIMIZER = 'AdamW'
AE_SCHEDULER_PATIENCE = 3
AE_SCHEDULER_FACTOR = 0.1
AE_EARLY_STOPPING_PATIENCE = 5
AE_MODEL_CHECKPOINT_NAME = 'autoencoder_best.pth'

PREDICTOR_EPOCHS = 30
PREDICTOR_BATCH_SIZE = 64
PREDICTOR_LEARNING_RATE = 5e-4
PREDICTOR_WEIGHT_DECAY = 1e-6
PREDICTOR_OPTIMIZER = 'AdamW' 
PREDICTOR_SCHEDULER_PATIENCE = 4
PREDICTOR_SCHEDULER_FACTOR = 0.2
PREDICTOR_EARLY_STOPPING_PATIENCE = 7
PREDICTOR_FINETUNE_ENCODER = True
PREDICTOR_MODEL_CHECKPOINT_NAME = 'predictor_best.pth'

# ==============================================================================
# 8. ANALYSIS & EVALUATION CONFIG
# ==============================================================================
OUTLIER_MODE = 'visit'
VISIT_ERROR_PERCENTILE = 98.0
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 'auto'
INFERENCE_BATCH_SIZE = 64

# ==============================================================================
# 9. DATA SPLITTING & RUNTIME SETTINGS
# ==============================================================================
VALIDATION_SPLIT_SIZE = 0.15
TEST_SPLIT_SIZE = 0.15
RANDOM_SEED = 42

# ==============================================================================
# 10. WORKFLOW CONTROL FLAGS
# ==============================================================================
TRAIN_AE = False
TRAIN_PREDICTOR = False

LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.0
RAG_NUM_CHUNKS_TO_RETRIEVE = 10
RAG_NUM_DOCS_TO_RETRIEVE = 5