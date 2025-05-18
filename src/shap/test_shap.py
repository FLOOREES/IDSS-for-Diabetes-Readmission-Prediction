import argparse
import pandas as pd
import torch
import logging
import os
import sys
import numpy as np # For np.abs().mean() in example logging
from pathlib import Path # For constructing default background_csv path

# --- Setup Project Path ---
# This assumes the script might be run from different locations or is in a subfolder.
# Adjust if your project structure guarantees 'src' is always in PYTHONPATH.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Imports from your 'src' modules ---
try:
    from src import config as AppConfig # Your main config
    from src.inference.predictor_engine import SinglePatientPredictorEngine
    from src.shap.shap_explainer import generate_shap_explanation
    from src.shap.shap_utils import prepare_shap_background_data
    from src.preprocessing.first_phase import FirstPhasePreprocessor # For consistent raw data loading
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Ensure PYTHONPATH is set correctly or run script from project root if modules are not found.")
    print(f"Attempted to add project root: {PROJECT_ROOT} to sys.path")
    sys.exit(1)

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def perform_shap_analysis(
    patient_csv_path: str, 
    background_csv_path: str, 
    num_background_samples_aim: int
    ):
    """
    Orchestrates loading data, initializing the engine, and generating SHAP explanations.
    """
    logger.info(f"--- Starting SHAP Analysis for Patient CSV: {patient_csv_path} ---")
    logger.info(f"Using background CSV: {background_csv_path}")
    logger.info(f"Aiming for approximately {num_background_samples_aim} sequences for SHAP background.")

    # --- 1. Initialize SinglePatientPredictorEngine ---
    # This engine loads all necessary models and preprocessor artifacts.
    try:
        engine = SinglePatientPredictorEngine(cfg=AppConfig)
        logger.info("SinglePatientPredictorEngine initialized successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize SinglePatientPredictorEngine: {e}", exc_info=True)
        return

    # --- 2. Load Raw Data for the Patient to Explain ---
    logger.info(f"Loading patient data to explain from: {patient_csv_path}")
    try:
        # Use FirstPhasePreprocessor.load_data for consistent NA handling from raw CSVs
        raw_patient_df_to_explain = FirstPhasePreprocessor.load_data(
            patient_csv_path, 
            AppConfig.MISSING_VALUES # Uses {'?': pd.NA} which load_data converts for read_csv
        )
        if raw_patient_df_to_explain.empty:
            logger.error(f"No data loaded from patient CSV: {patient_csv_path}")
            return
        # Ensure chronological order for the patient's visits
        raw_patient_df_to_explain = raw_patient_df_to_explain.sort_values(by=AppConfig.ENCOUNTER_ID_COL)
        logger.info(f"Loaded data for patient to explain. Shape: {raw_patient_df_to_explain.shape}")
    except FileNotFoundError:
        logger.error(f"Patient CSV not found: {patient_csv_path}")
        return
    except Exception as e:
        logger.error(f"Failed to load patient data to explain: {e}", exc_info=True)
        return

    # --- 3. Prepare Background Data Tensor for SHAP ---
    logger.info(f"Loading raw background data from: {background_csv_path}")
    try:
        sample_df_for_background_raw = FirstPhasePreprocessor.load_data(
            background_csv_path,
            AppConfig.MISSING_VALUES
        )
        if sample_df_for_background_raw.empty:
            logger.error(f"No data loaded from background CSV: {background_csv_path}")
            return

        background_data_tensor = prepare_shap_background_data(
            engine, 
			df_background_raw_all_patients=sample_df_for_background_raw,
            num_background_sequences_aim=num_background_samples_aim
        )
        logger.info(f"SHAP background data tensor prepared. Target shape for explainer: {background_data_tensor.shape}")
    except FileNotFoundError:
        logger.error(f"Background sample CSV not found: {background_csv_path}")
        return
    except ValueError as ve: 
        logger.error(f"Error preparing background data (e.g., no sequences): {ve}", exc_info=True)
        return
    except Exception as e:
        logger.error(f"Failed to prepare SHAP background data: {e}", exc_info=True)
        return
        
    # --- 4. Get Feature Name Lists (needed by generate_shap_explanation for plots) ---
    config_numerical_features = AppConfig.NUMERICAL_FEATURES
    # actual_ohe_columns are loaded by SinglePatientPredictorEngine's data_preparer
    if not hasattr(engine, 'data_preparer') or not engine.data_preparer or \
       not hasattr(engine.data_preparer, 'actual_ohe_columns') or not engine.data_preparer.actual_ohe_columns:
        logger.error("OHE actual columns not available from engine's data_preparer. Cannot determine feature names for SHAP plots.")
        return
    config_ohe_actual_cols = engine.data_preparer.actual_ohe_columns

    # --- 5. Call generate_shap_explanation from shap_utils ---
    logger.info(f"Generating SHAP explanation for all classes...")
    try:
        shap_results = generate_shap_explanation( # MODIFIED: Call no longer passes target_class_idx
            engine=engine,
            raw_patient_df_to_explain=raw_patient_df_to_explain,
            background_data_tensor_for_shap=background_data_tensor,
            config_numerical_features=config_numerical_features,
            config_ohe_actual_cols=config_ohe_actual_cols
        )
        
        if shap_results and "all_class_shap_values_sequence" in shap_results:
            logger.info("SHAP explanation generation completed successfully for all classes!")
            
            all_shaps = shap_results['all_class_shap_values_sequence'] # List of arrays
            feature_names = shap_results['encoder_input_feature_names']
            
            for i, class_shaps in enumerate(all_shaps):
                avg_abs_shap = pd.Series(
                    np.abs(class_shaps).mean(axis=0), 
                    index=feature_names
                ).sort_values(ascending=False)
                logger.info(f"Class {i} - Top 10 features by average absolute SHAP value:\n{avg_abs_shap.head(10)}")
            
            shap_output_dir = os.path.join(AppConfig.RESULTS_DIR, "shap_explanations")
            logger.info(f"SHAP summary plot (multi-class) should be saved in: {shap_output_dir}")
        else:
            logger.error("SHAP explanation generation failed or returned unexpected results.")

    except Exception as e:
        logger.error(f"An error occurred during SHAP explanation: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for a single patient's raw data CSV.")
    parser.add_argument(
        "patient_csv", 
        type=str, 
        help="Path to the CSV file containing raw visits for a single patient."
    )
    parser.add_argument(
        "--background_csv", 
        type=str, 
        # Default path assumes the output structure from export_verified_test_samples.py
        default=str(Path(AppConfig.DATA_DIR) / "training_samples" / "sample_raw_patients_for_background.csv"),
        help="Path to the CSV file containing raw visits for SHAP background data."
    )
    parser.add_argument(
        "--num_bg_samples", 
        type=int, 
        default=50, # Number of sequences to try and use for background
        help="Approximate number of sequences to use for the SHAP background dataset. Default: 50"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.patient_csv):
        logger.error(f"Provided patient CSV path does not exist: {args.patient_csv}")
        sys.exit(1)
    if not os.path.exists(args.background_csv):
        logger.error(f"Provided background CSV path does not exist: {args.background_csv}. "
                     "You can generate this using 'export_verified_test_samples.py'.")
        sys.exit(1)

    perform_shap_analysis(
        patient_csv_path=args.patient_csv, 
        background_csv_path=args.background_csv,
        num_background_samples_aim=args.num_bg_samples
    )