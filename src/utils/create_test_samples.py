import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from sklearn.model_selection import GroupShuffleSplit

# Adjust this import based on your project structure and how config is accessed
# This assumes your config.py is in 'src' and can be imported as AppConfig
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root to path
from src import config as AppConfig
from src.preprocessing.first_phase import FirstPhasePreprocessor # For its load_data static method

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- Configuration ---
NUM_TEST_PATIENTS_TO_EXPORT = 5  # How many random patients from the test set to export
OUTPUT_SUBDIR = "test_split_raw_samples" # Subdirectory within AppConfig.DATA_DIR

def generate_and_export_test_samples():
    """
    Identifies patients in the test split (based on pipeline's logic)
    and exports their original raw data to individual CSV files.
    """
    logger.info("--- Starting Export of Raw Samples from Test Split ---")

    # --- 1. Load Fully Preprocessed Data (this is what the split is based on) ---
    final_processed_data_path = Path(AppConfig.FINAL_ENCODED_DATA_PATH)
    if not final_processed_data_path.exists():
        logger.error(f"Fully preprocessed data file not found at: {final_processed_data_path}")
        logger.error("Please run the main pipeline to generate this file first.")
        return
    try:
        df_final_processed = pd.read_csv(final_processed_data_path, low_memory=False)
        logger.info(f"Loaded fully preprocessed data. Shape: {df_final_processed.shape}")
    except Exception as e:
        logger.error(f"Failed to load preprocessed data from {final_processed_data_path}: {e}", exc_info=True)
        return

    # Ensure necessary columns are present for splitting
    if AppConfig.PATIENT_ID_COL not in df_final_processed.columns:
        logger.error(f"Patient ID column '{AppConfig.PATIENT_ID_COL}' not found in preprocessed data.")
        return

    # --- 2. Perform the Test Split to Identify Test Patient IDs ---
    logger.info(f"Performing test split on preprocessed data using RANDOM_SEED: {AppConfig.RANDOM_SEED}")
    patient_groups = df_final_processed[AppConfig.PATIENT_ID_COL]
    all_indices = np.arange(len(df_final_processed))

    gss_test_split = GroupShuffleSplit(
        n_splits=1, 
        test_size=AppConfig.TEST_SPLIT_SIZE, 
        random_state=AppConfig.RANDOM_SEED
    )
    
    try:
        # Get indices for train_val and test sets
        # The split is on all_indices, grouped by patient_groups from df_final_processed
        _, test_indices = next(gss_test_split.split(all_indices, groups=patient_groups))
        df_test_subset_processed = df_final_processed.iloc[test_indices]
        test_patient_ids = df_test_subset_processed[AppConfig.PATIENT_ID_COL].unique()
    except Exception as e:
        logger.error(f"Error during data splitting: {e}", exc_info=True)
        return

    if len(test_patient_ids) == 0:
        logger.warning("No patients found in the generated test split. Cannot export samples.")
        return
    logger.info(f"Identified {len(test_patient_ids)} unique patients in the test split.")

    # --- 3. Select Patients to Export ---
    if len(test_patient_ids) <= NUM_TEST_PATIENTS_TO_EXPORT:
        selected_patient_ids_for_export = test_patient_ids
    else:
        selected_patient_ids_for_export = np.random.choice(
            test_patient_ids, 
            NUM_TEST_PATIENTS_TO_EXPORT, 
            replace=False
        )
    logger.info(f"Selected {len(selected_patient_ids_for_export)} test patients for raw data export: {selected_patient_ids_for_export.tolist()}")

    # --- 4. Load Original Raw Data ---
    logger.info(f"Loading original raw data from: {AppConfig.RAW_DATA_PATH}")
    try:
        # Using the static method from FirstPhasePreprocessor for consistency in loading
        df_raw_full = FirstPhasePreprocessor.load_data(AppConfig.RAW_DATA_PATH, AppConfig.MISSING_VALUES)
    except FileNotFoundError:
        logger.error(f"Original raw data file not found at: {AppConfig.RAW_DATA_PATH}")
        return
    except Exception as e:
        logger.error(f"Failed to load original raw data: {e}", exc_info=True)
        return
    
    if AppConfig.PATIENT_ID_COL not in df_raw_full.columns:
        logger.error(f"Patient ID column '{AppConfig.PATIENT_ID_COL}' not found in original raw data.")
        return
    if AppConfig.ENCOUNTER_ID_COL not in df_raw_full.columns:
        logger.error(f"Encounter ID column '{AppConfig.ENCOUNTER_ID_COL}' not found in original raw data (needed for sorting).")
        return

    # --- 5. Filter Raw Data and Save ---
    output_path_base = Path(AppConfig.DATA_DIR) / OUTPUT_SUBDIR
    output_path_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring output directory exists: {output_path_base}")

    exported_count = 0
    for patient_id in selected_patient_ids_for_export:
        df_patient_raw = df_raw_full[df_raw_full[AppConfig.PATIENT_ID_COL] == patient_id].copy()
        
        if df_patient_raw.empty:
            logger.warning(f"No raw data found for patient ID {patient_id}. Skipping.")
            continue
            
        # Ensure chronological order by encounter ID
        df_patient_raw = df_patient_raw.sort_values(by=AppConfig.ENCOUNTER_ID_COL)
        
        file_name = f"patient_{patient_id}.csv"
        output_file_path = output_path_base / file_name
        
        try:
            df_patient_raw.to_csv(output_file_path, index=False)
            logger.info(f"Successfully saved raw data for patient {patient_id} to {output_file_path}")
            exported_count += 1
        except Exception as e:
            logger.error(f"Failed to save CSV for patient {patient_id} to {output_file_path}: {e}", exc_info=True)

    logger.info(f"--- Export completed. {exported_count} patient CSVs saved to {output_path_base} ---")

if __name__ == '__main__':
    generate_and_export_test_samples()