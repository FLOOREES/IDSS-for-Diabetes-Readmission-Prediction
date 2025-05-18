# scripts/export_training_samples_for_background.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from sklearn.model_selection import GroupShuffleSplit

# Adjust this import based on your project structure
import sys
# Assuming this script is in 'scripts/' at the project root, and 'src' is also at root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from src import config as AppConfig
from src.preprocessing.first_phase import FirstPhasePreprocessor # For its load_data static method

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Number of unique training patients to include in the background sample CSV
NUM_BACKGROUND_PATIENTS_TO_EXPORT = 50 
# MODIFIED: Output subdirectory changed to 'training_samples'
OUTPUT_SUBDIR = "training_samples" 
BACKGROUND_CSV_FILENAME = "sample_raw_patients_for_background.csv" # Standard filename

def generate_and_export_background_samples():
    logger.info(f"--- Starting Export of Raw Training Samples for SHAP Background ({NUM_BACKGROUND_PATIENTS_TO_EXPORT} patients) ---")
    logger.info(f"Output target directory: {Path(AppConfig.DATA_DIR) / OUTPUT_SUBDIR}")

    # --- 1. Load Fully Preprocessed Data (this is what the split is based on) ---
    final_processed_data_path = Path(AppConfig.FINAL_ENCODED_DATA_PATH)
    if not final_processed_data_path.exists():
        logger.error(f"Fully preprocessed data file not found at: {final_processed_data_path}. Run main pipeline first.")
        return
    try:
        df_final_processed = pd.read_csv(final_processed_data_path, low_memory=False)
        logger.info(f"Loaded fully preprocessed data. Shape: {df_final_processed.shape}")
    except Exception as e:
        logger.error(f"Failed to load preprocessed data from {final_processed_data_path}: {e}", exc_info=True)
        return

    if AppConfig.PATIENT_ID_COL not in df_final_processed.columns:
        logger.error(f"Patient ID column '{AppConfig.PATIENT_ID_COL}' not found in preprocessed data.")
        return

    # --- 2. Perform Full Train/Validation/Test Split (Replicating pipeline._split_data logic) ---
    logger.info(f"Performing full train/val/test split using RANDOM_SEED: {AppConfig.RANDOM_SEED} to identify training patients.")
    
    patient_groups = df_final_processed[AppConfig.PATIENT_ID_COL]
    all_indices = np.arange(len(df_final_processed))

    gss_tv_t = GroupShuffleSplit(n_splits=1, test_size=AppConfig.TEST_SPLIT_SIZE, random_state=AppConfig.RANDOM_SEED)
    try:
        train_val_indices, test_indices = next(gss_tv_t.split(all_indices, groups=patient_groups))
    except Exception as e:
        logger.error(f"Error during initial train/test split: {e}", exc_info=True)
        return

    df_test_processed = df_final_processed.iloc[test_indices] 
    df_train_val_processed = df_final_processed.iloc[train_val_indices]

    actual_test_pids_set = set(df_test_processed[AppConfig.PATIENT_ID_COL].unique())
    actual_train_pids_set = set()
    actual_val_pids_set = set()

    if not df_train_val_processed.empty:
        train_val_patient_groups = df_train_val_processed[AppConfig.PATIENT_ID_COL]
        train_val_all_indices = np.arange(len(df_train_val_processed))
        
        denominator_for_val_split = (1.0 - AppConfig.TEST_SPLIT_SIZE)
        if denominator_for_val_split <= 0: 
            logger.warning("TEST_SPLIT_SIZE is >= 100%, no data left for train/validation. Train/Val sets will be empty.")
            df_train_processed = pd.DataFrame(columns=df_train_val_processed.columns)
        else:
            val_proportion_of_train_val = AppConfig.VALIDATION_SPLIT_SIZE / denominator_for_val_split
            if not (0 < val_proportion_of_train_val < 1): 
                logger.warning(f"Adjusted validation proportion ({val_proportion_of_train_val:.4f}) for train_val set is not between 0 and 1.")
                if val_proportion_of_train_val <= 0:
                    df_train_processed = df_train_val_processed.copy()
                    df_val_processed = pd.DataFrame(columns=df_train_val_processed.columns)
                else: 
                    df_val_processed = df_train_val_processed.copy()
                    df_train_processed = pd.DataFrame(columns=df_train_val_processed.columns)
            else:
                gss_t_v = GroupShuffleSplit(n_splits=1, test_size=val_proportion_of_train_val, random_state=AppConfig.RANDOM_SEED)
                try:
                    train_indices, val_indices = next(gss_t_v.split(train_val_all_indices, groups=train_val_patient_groups))
                    df_train_processed = df_train_val_processed.iloc[train_indices]
                    df_val_processed = df_train_val_processed.iloc[val_indices]
                except ValueError as ve:
                    logger.error(f"Error during train/validation split (ValueError: {ve}). Assigning all train_val to train.")
                    df_train_processed = df_train_val_processed.copy()
                    df_val_processed = pd.DataFrame(columns=df_train_val_processed.columns)
                except Exception as e:
                    logger.error(f"Error during train/validation split: {e}", exc_info=True)
                    return
            
            actual_train_pids_set = set(df_train_processed[AppConfig.PATIENT_ID_COL].unique())
            actual_val_pids_set = set(df_val_processed[AppConfig.PATIENT_ID_COL].unique())
    
    logger.info(f"Identified {len(actual_train_pids_set)} unique patients in the 'actual_train_set'.")
    logger.info(f"Identified {len(actual_val_pids_set)} unique patients in the 'actual_val_set'.")
    logger.info(f"Identified {len(actual_test_pids_set)} unique patients in the 'actual_test_set'.")

    if not (actual_train_pids_set.isdisjoint(actual_val_pids_set) and \
            actual_train_pids_set.isdisjoint(actual_test_pids_set) and \
            actual_val_pids_set.isdisjoint(actual_test_pids_set)):
        logger.error("CRITICAL: Patient ID overlap detected between generated train/val/test sets! Halting export.")
        return
    else:
        logger.info("Patient ID overlap check passed for the generated splits.")

    # --- 3. Select Training Patients for Background Sample Export ---
    if not actual_train_pids_set:
        logger.warning("Actual training set is empty. Cannot export background samples.")
        return

    num_to_select = min(NUM_BACKGROUND_PATIENTS_TO_EXPORT, len(actual_train_pids_set))
    if num_to_select <= 0:
        logger.warning(f"Number of background patients to select is {num_to_select}. No background samples will be exported.")
        return
        
    selected_background_patient_ids = np.random.choice(
        list(actual_train_pids_set), 
        num_to_select, 
        replace=False
    ).tolist()
    logger.info(f"Selected {len(selected_background_patient_ids)} training patients for SHAP background export: {selected_background_patient_ids[:5]}...")

    # --- 4. Load Original Raw Data ---
    logger.info(f"Loading original raw data from: {AppConfig.RAW_DATA_PATH}")
    try:
        df_raw_full = FirstPhasePreprocessor.load_data(AppConfig.RAW_DATA_PATH, AppConfig.MISSING_VALUES)
    except FileNotFoundError:
        logger.error(f"Original raw data file not found at: {AppConfig.RAW_DATA_PATH}")
        return
    except Exception as e:
        logger.error(f"Failed to load original raw data: {e}", exc_info=True)
        return
    
    for col_check in [AppConfig.PATIENT_ID_COL, AppConfig.ENCOUNTER_ID_COL]:
        if col_check not in df_raw_full.columns:
            logger.error(f"Required column '{col_check}' not found in original raw data. Cannot proceed.")
            return

    # --- 5. Filter Raw Data for Selected Training Patients and Save as a Single CSV ---
    df_background_raw_sample = df_raw_full[df_raw_full[AppConfig.PATIENT_ID_COL].isin(selected_background_patient_ids)].copy()
    
    if df_background_raw_sample.empty:
        logger.warning(f"No raw data found for the selected background patient IDs: {selected_background_patient_ids}. Background CSV will not be saved.")
        return

    # MODIFIED: Output path uses new OUTPUT_SUBDIR
    output_path_base = Path(AppConfig.DATA_DIR) / OUTPUT_SUBDIR 
    output_path_base.mkdir(parents=True, exist_ok=True)
    background_output_file_path = output_path_base / BACKGROUND_CSV_FILENAME
    
    try:
        df_background_raw_sample = df_background_raw_sample.sort_values(
            by=[AppConfig.PATIENT_ID_COL, AppConfig.ENCOUNTER_ID_COL]
        )
        df_background_raw_sample.to_csv(background_output_file_path, index=False)
        logger.info(f"Successfully saved RAW data for {len(selected_background_patient_ids)} background (training) patients to {background_output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save background CSV to {background_output_file_path}: {e}", exc_info=True)
    
    logger.info(f"--- Background sample export completed. ---")

if __name__ == '__main__':
    generate_and_export_background_samples() # Call the correctly named function