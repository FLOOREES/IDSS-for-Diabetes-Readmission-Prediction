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
    format='[%(asctime)s] %(levelname)s: %(name)s - %(message)s', # Corrected module for name
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- Configuration ---
NUM_TEST_PATIENTS_TO_EXPORT = 5  # How many random verified test patients to export
OUTPUT_SUBDIR = "test_raw_samples" # Subdirectory within AppConfig.DATA_DIR

def generate_and_export_verified_test_samples():
    logger.info("--- Starting Export of Verified Raw Samples from Test Split ---")

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
    logger.info(f"Performing full train/val/test split using RANDOM_SEED: {AppConfig.RANDOM_SEED}")
    
    patient_groups = df_final_processed[AppConfig.PATIENT_ID_COL]
    all_indices = np.arange(len(df_final_processed))

    # First split: (train_val) and (test)
    gss_tv_t = GroupShuffleSplit(n_splits=1, test_size=AppConfig.TEST_SPLIT_SIZE, random_state=AppConfig.RANDOM_SEED)
    try:
        train_val_indices, test_indices = next(gss_tv_t.split(all_indices, groups=patient_groups))
    except Exception as e:
        logger.error(f"Error during initial train/test split: {e}", exc_info=True)
        return

    df_test_processed = df_final_processed.iloc[test_indices]
    df_train_val_processed = df_final_processed.iloc[train_val_indices]

    actual_test_pids_set = set(df_test_processed[AppConfig.PATIENT_ID_COL].unique())
    logger.info(f"Identified {len(actual_test_pids_set)} unique patients in the 'actual_test_set'.")

    actual_train_pids_set = set()
    actual_val_pids_set = set()

    if not df_train_val_processed.empty:
        train_val_patient_groups = df_train_val_processed[AppConfig.PATIENT_ID_COL]
        train_val_all_indices = np.arange(len(df_train_val_processed))
        
        denominator_for_val_split = (1.0 - AppConfig.TEST_SPLIT_SIZE)
        if denominator_for_val_split <= 0: # Avoid division by zero or negative if TEST_SPLIT_SIZE >= 1
            logger.warning("TEST_SPLIT_SIZE is >= 100%, no data left for train/validation. Train/Val sets will be empty.")
            # df_train_processed and df_val_processed will remain empty or be empty DataFrames
            df_train_processed = pd.DataFrame(columns=df_train_val_processed.columns)
            df_val_processed = pd.DataFrame(columns=df_train_val_processed.columns)

        else:
            val_proportion_of_train_val = AppConfig.VALIDATION_SPLIT_SIZE / denominator_for_val_split
            
            if not (0 < val_proportion_of_train_val < 1):
                logger.warning(f"Adjusted validation proportion ({val_proportion_of_train_val:.4f}) for train_val set is not between 0 and 1. This means either validation or train will be empty from this subset.")
                if val_proportion_of_train_val <= 0: # No validation set from remaining
                    df_train_processed = df_train_val_processed.copy()
                    df_val_processed = pd.DataFrame(columns=df_train_val_processed.columns) # Empty
                else: # All remaining is validation, no train
                    df_val_processed = df_train_val_processed.copy()
                    df_train_processed = pd.DataFrame(columns=df_train_val_processed.columns) # Empty
            else:
                gss_t_v = GroupShuffleSplit(n_splits=1, test_size=val_proportion_of_train_val, random_state=AppConfig.RANDOM_SEED)
                try:
                    train_indices, val_indices = next(gss_t_v.split(train_val_all_indices, groups=train_val_patient_groups))
                    df_train_processed = df_train_val_processed.iloc[train_indices]
                    df_val_processed = df_train_val_processed.iloc[val_indices]
                except ValueError as ve: # Handle case where split might not be possible (e.g. too few groups)
                    logger.error(f"Error during train/validation split (ValueError: {ve}). This can happen with small datasets/groups for the given split size. Assigning all to train.")
                    df_train_processed = df_train_val_processed.copy()
                    df_val_processed = pd.DataFrame(columns=df_train_val_processed.columns) # Empty
                except Exception as e:
                    logger.error(f"Error during train/validation split: {e}", exc_info=True)
                    return
            
            actual_train_pids_set = set(df_train_processed[AppConfig.PATIENT_ID_COL].unique())
            actual_val_pids_set = set(df_val_processed[AppConfig.PATIENT_ID_COL].unique())
    
    logger.info(f"Identified {len(actual_train_pids_set)} unique patients in the 'actual_train_set'.")
    logger.info(f"Identified {len(actual_val_pids_set)} unique patients in the 'actual_val_set'.")

    # Overlap Sanity Check (Essential)
    if not (actual_train_pids_set.isdisjoint(actual_val_pids_set) and \
            actual_train_pids_set.isdisjoint(actual_test_pids_set) and \
            actual_val_pids_set.isdisjoint(actual_test_pids_set)):
        logger.error("CRITICAL: Patient ID overlap detected between generated train/val/test sets! Halting export.")
        logger.error(f"  Train-Val intersection: {actual_train_pids_set.intersection(actual_val_pids_set)}")
        logger.error(f"  Train-Test intersection: {actual_train_pids_set.intersection(actual_test_pids_set)}")
        logger.error(f"  Val-Test intersection: {actual_val_pids_set.intersection(actual_test_pids_set)}")
        return
    else:
        logger.info("Patient ID overlap check passed for the generated splits.")

    # --- 3. Select Patients to Export (from actual_test_pids_set) ---
    selected_patient_ids_for_export = []
    if not actual_test_pids_set:
        logger.warning("Actual test set is empty. No patient samples will be exported from test set.")
    else:
        if len(actual_test_pids_set) <= NUM_TEST_PATIENTS_TO_EXPORT:
            selected_patient_ids_for_export = list(actual_test_pids_set)
        else:
            selected_patient_ids_for_export = np.random.choice(
                list(actual_test_pids_set), 
                NUM_TEST_PATIENTS_TO_EXPORT, 
                replace=False
            ).tolist()
        logger.info(f"Will attempt to export raw data for {len(selected_patient_ids_for_export)} selected test patients: {selected_patient_ids_for_export}")

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

    # --- 5. Filter Raw Data, VERIFY, and Save ---
    output_path_base = Path(AppConfig.DATA_DIR) / OUTPUT_SUBDIR
    output_path_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for samples: {output_path_base}")

    exported_count = 0
    for patient_id in selected_patient_ids_for_export:
        # --- VERIFICATION LOGIC ---
        is_in_test = patient_id in actual_test_pids_set
        is_in_train = patient_id in actual_train_pids_set
        is_in_val = patient_id in actual_val_pids_set

        if is_in_test and not is_in_train and not is_in_val:
            logger.info(f"Verification PASSED for patient ID {patient_id}: Correctly in test set and not in train/val.")
        else:
            logger.error(f"VERIFICATION FAILED for patient ID {patient_id}! This patient will NOT be exported.")
            logger.error(f"  Is in actual_test_pids_set: {is_in_test}")
            logger.error(f"  Is in actual_train_pids_set: {is_in_train}")
            logger.error(f"  Is in actual_val_pids_set: {is_in_val}")
            continue # Skip to the next patient

        df_patient_raw = df_raw_full[df_raw_full[AppConfig.PATIENT_ID_COL] == patient_id].copy()
        
        if df_patient_raw.empty:
            logger.warning(f"No raw data found for verified test patient ID {patient_id}. Skipping.")
            continue
            
        df_patient_raw = df_patient_raw.sort_values(by=AppConfig.ENCOUNTER_ID_COL)
        
        file_name = f"patient_{patient_id}.csv" # Using patient_id directly in filename
        output_file_path = output_path_base / file_name
        
        try:
            df_patient_raw.to_csv(output_file_path, index=False)
            logger.info(f"Successfully saved raw data for verified test patient {patient_id} to {output_file_path}")
            exported_count += 1
        except Exception as e:
            logger.error(f"Failed to save CSV for patient {patient_id} to {output_file_path}: {e}", exc_info=True)
    
    logger.info(f"--- Export completed. {exported_count} verified test patient CSVs saved to {output_path_base} ---")

if __name__ == '__main__':
    generate_and_export_verified_test_samples()