# In your SHAP analysis script (e.g., run_shap.py or a Jupyter Notebook)
import shap
import pandas as pd
import torch
import matplotlib.pyplot as plt # For plotting
import os

# Assuming these are correctly set up and imported
from src.inference.predictor_engine import SinglePatientPredictorEngine
from src.shap.shap_utils import get_encoder_input_for_shap, prepare_shap_background_data, RNNWrapperForSHAP
from src import config as AppConfig # Your main config
# Ensure logging is set up if you want to see logs from the engine
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

def explain_patient_prediction_with_shap(engine: SinglePatientPredictorEngine, 
                                         raw_patient_df: pd.DataFrame, 
                                         background_data_tensor: torch.Tensor,
                                         target_class_index: Optional[int] = None, # Index of class to explain, e.g., 2 for '<30'
                                         feature_names: Optional[List[str]] = None): # For plot labeling
    """
    Generates SHAP explanations for a single patient's prediction.
    """
    logger.info(f"Generating SHAP explanation for patient.")

    # 1. Prepare the instance to explain
    instance_encoder_input = get_encoder_input_for_shap(engine, raw_patient_df)
    instance_on_device = instance_encoder_input.to(engine.device)

    # 2. Define the SHAP model (RNN part + Head part)
    # Make sure these are the actual nn.LSTM/GRU layer and the PredictionHead layer
    shap_model = RNNWrapperForSHAP(
        engine.predictor_model.encoder.rnn, # Accessing the rnn submodule of EncoderRNN
        engine.predictor_model.head
    ).to(engine.device)
    shap_model.eval()

    # 3. Create DeepExplainer
    # Background data should also be on the same device as the model
    background_on_device = background_data_tensor.to(engine.device)
    explainer = shap.DeepExplainer(shap_model, background_on_device)

    # 4. Get SHAP values
    # shap_values will be a list of tensors (one per output class) if explaining all classes
    # Each tensor: (batch_size_of_instance (1), seq_len, num_encoder_input_features)
    shap_values_output = explainer.shap_values(instance_on_device)
    
    logger.info("SHAP values generated.")

    # 5. Post-process and Visualize (example for one class and one instance)
    # If explaining all classes, shap_values_output is a list.
    # If target_class_index is specified, DeepExplainer might return just for that class,
    # or we select it: shap_values_for_class = shap_values_output[target_class_index]
    # If shap_values is for all classes (list of arrays):
    if isinstance(shap_values_output, list):
        if target_class_index is not None and target_class_index < len(shap_values_output):
            shap_values_for_one_class = shap_values_output[target_class_index]
        else: # Default to explaining the first class or sum/avg if not specified
            logger.warning(f"Target class index {target_class_index} invalid or not provided. Using class 0.")
            shap_values_for_one_class = shap_values_output[0] 
    else: # single output model
        shap_values_for_one_class = shap_values_output

    # shap_values_for_one_class is (num_explained_instances (1), seq_len, num_features)
    # Squeeze the batch dimension if explaining one instance
    shap_values_single_instance = shap_values_for_one_class[0] # (seq_len, num_features)
    
    # Get actual sequence length (from mask, for example)
    # We can get this from the engine's prediction output for this patient
    prediction_details = engine.predict_for_patient(raw_patient_df) # Re-predict to get mask/length easily or pass it
    processed_input_for_mask = prediction_details["processed_model_input"]
    actual_seq_len = processed_input_for_mask['lengths'][0].item()
    
    valid_shap_values = shap_values_single_instance[:actual_seq_len, :] # (actual_seq_len, num_features)
    valid_instance_data = instance_on_device[0, :actual_seq_len, :] # (actual_seq_len, num_features)

    # Basic summary plot (might need adjustment for sequences)
    # For sequence data, a global summary plot might average over timesteps, or you can plot per timestep.
    # If feature_names are not provided, SHAP will use default names.
    # Feature names should correspond to the columns of `instance_encoder_input`
    # (num_ohe features + embedding dimensions)
    
    # Creating feature names for the encoder_input_tensor:
    if feature_names is None:
        num_ohe_cols = engine.data_preparer.actual_ohe_columns
        numerical_cols = engine.data_preparer.numerical_features
        # For embedding dimensions, create generic names
        total_emb_dim = engine.predictor_model.encoder.embedding_manager.get_total_embedding_dim()
        emb_feature_names = [f"emb_dim_{i}" for i in range(total_emb_dim)]
        # The order in encoder_input was num_ohe (which includes numerical after scaling) + embeddings
        # This needs to precisely match how `encoder_input_tensor` was constructed.
        # In `get_encoder_input_for_shap` it's `num_ohe_tensor` then `embeddings_output`.
        # `num_ohe_tensor` comes from `batch_on_device['num_ohe']` which was created by SequenceDataPreparer
        # using `num_ohe_cols = self.numerical_features + self.actual_ohe_columns`.
        # So the order is numerical features then actual_ohe_columns.
        encoder_input_feature_names = numerical_cols + engine.data_preparer.actual_ohe_columns + emb_feature_names
        
        # Ensure the length matches
        if len(encoder_input_feature_names) != valid_shap_values.shape[1]:
            logger.error(f"Mismatch in generated feature names ({len(encoder_input_feature_names)}) and SHAP values features ({valid_shap_values.shape[1]})")
            # Fallback to no feature names if there's a mismatch
            final_feature_names = None
        else:
            final_feature_names = encoder_input_feature_names
    else:
        final_feature_names = feature_names

    # Plotting (this will require matplotlib)
    # SHAP plots might work directly if you provide numpy arrays.
    # Convert to numpy if they are torch tensors.
    valid_shap_values_np = valid_shap_values.cpu().numpy()
    valid_instance_data_np = valid_instance_data.cpu().numpy()

    logger.info("Generating SHAP summary plot (bar) averaged over timesteps.")
    shap.summary_plot(valid_shap_values_np, features=valid_instance_data_np, feature_names=final_feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance (Avg over sequence, Class {target_class_index or 0})")
    plt.savefig("shap_summary_bar.png")
    plt.show()
    logger.info("SHAP bar plot saved to shap_summary_bar.png")

    # For a specific timestep (e.g., the last one)
    last_timestep_idx = actual_seq_len - 1
    if last_timestep_idx >= 0:
        logger.info(f"Generating SHAP waterfall plot for last timestep ({last_timestep_idx}).")
        # explainer.expected_value is often a list for multi-output models
        expected_value_for_class = explainer.expected_value
        if isinstance(explainer.expected_value, list):
            expected_value_for_class = explainer.expected_value[target_class_index if target_class_index is not None else 0]
        
        shap.plots.waterfall(shap.Explanation(
            values=valid_shap_values_np[last_timestep_idx, :],
            base_values=expected_value_for_class, # This might need adjustment based on DeepExplainer output format
            data=valid_instance_data_np[last_timestep_idx, :],
            feature_names=final_feature_names
        ), show=False, max_display=15)
        plt.title(f"SHAP Waterfall for Last Timestep (Class {target_class_index or 0})")
        plt.savefig("shap_waterfall_last_step.png")
        plt.show()
        logger.info("SHAP waterfall plot saved to shap_waterfall_last_step.png")

    return shap_values_output # Return the raw SHAP values

# Example main execution for the SHAP script
if __name__ == '__main__':
    # --- Setup (Logging, Load Patient Data, Engine) ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Initialize Engine
    # Ensure AppConfig is correctly imported and configured with all paths
    # from src import config as AppConfig 
    engine = SinglePatientPredictorEngine(cfg=AppConfig) 

    # 2. Load raw patient data from CSV (example for one patient)
    # This needs to be the CSV format the user uploads
    # For testing, use a slice of your original raw data for one patient
    try:
        # Example: replace with actual path or user upload mechanism
        path_to_patient_csv = "path_to_single_patient_visits.csv" # User provides this
        if not os.path.exists(path_to_patient_csv):
             logger.error(f"Patient CSV not found: {path_to_patient_csv}. Please provide a valid path.")
             # Create a dummy one for demonstration if needed, ensuring all columns are present
             # For a real run, this path should be valid.
             # As a fallback for now, let's try to grab one patient from the full raw dataset
             logger.info("Attempting to grab a sample patient from the full raw dataset for SHAP demo.")
             full_raw_df = pd.read_csv(AppConfig.RAW_DATA_PATH, na_values=list(AppConfig.MISSING_VALUES.keys()))
             example_patient_id = full_raw_df[AppConfig.PATIENT_ID_COL].unique()[0]
             raw_patient_df_to_explain = full_raw_df[full_raw_df[AppConfig.PATIENT_ID_COL] == example_patient_id].copy()
             logger.info(f"Using patient {example_patient_id} with {len(raw_patient_df_to_explain)} visits for SHAP explanation.")
        else:
             raw_patient_df_to_explain = pd.read_csv(path_to_patient_csv, na_values=list(AppConfig.MISSING_VALUES.keys()))
        
        # Ensure it's sorted by encounter/timestamp if not already
        raw_patient_df_to_explain = raw_patient_df_to_explain.sort_values(by=AppConfig.ENCOUNTER_ID_COL)


        # 3. Prepare Background Data (Using a sample of training data)
        # This is a simplified way to get some background data.
        # Ideally, you'd load df_train, sample some patients, and process them.
        # For MVP, using a few patients from the raw data might suffice.
        logger.info("Preparing SHAP background data (using a few sample patients from raw data)...")
        # Let's take first N patients from raw data as sample for background
        # This is a placeholder. A more robust approach involves using actual df_train.
        num_background_patients = 5 # Small number for speed
        background_patient_ids = full_raw_df[AppConfig.PATIENT_ID_COL].unique()[:num_background_patients]
        sample_background_df = full_raw_df[full_raw_df[AppConfig.PATIENT_ID_COL].isin(background_patient_ids)].copy()
        
        background_data_tensor = prepare_shap_background_data(
            engine, 
            sample_background_df, 
            AppConfig.PATIENT_ID_COL, 
            AppConfig.ENCOUNTER_ID_COL
        )
        logger.info(f"SHAP background data prepared. Shape: {background_data_tensor.shape}")


        # 4. Run SHAP explanation for a target class (e.g., class 2 for '<30 days')
        TARGET_READMISSION_CLASS_INDEX = 2 # Example: '<30 days'
        
        shap_output = explain_patient_prediction_with_shap(
            engine, 
            raw_patient_df_to_explain, 
            background_data_tensor,
            target_class_index=TARGET_READMISSION_CLASS_INDEX
        )
        # Raw SHAP values are in shap_output for further processing if needed

    except Exception as e:
        logger.error(f"Error in SHAP analysis script: {e}", exc_info=True)

# In main SHAP script, for background data:
# pipeline_instance = Pipeline(AppConfig) # Assuming Pipeline class is accessible
# df_final = pipeline_instance._run_preprocessing_and_load() # Or load df_final if it exists
# df_train, _, _ = pipeline_instance._split_data(df_final)
# ohe_list = ... # load ohe list
# data_preparer = SequenceDataPreparer(..., actual_ohe_columns=ohe_list, ...)
# data_preparer.fit_scaler(df_train) # Fit scaler if not already done and saved
#
# train_feat_seqs, train_tgt_seqs, train_pids = data_preparer.transform(df_train.head(AppConfig.INFERENCE_BATCH_SIZE * 2)) # Sample
# train_dataset = PatientSequenceDataset(train_feat_seqs, train_tgt_seqs, train_pids)
# bg_loader = DataLoader(train_dataset, batch_size=AppConfig.INFERENCE_BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
# background_batch_collated = next(iter(bg_loader))
# background_batch_on_device = engine._move_batch_to_device(background_batch_collated)
# num_ohe_bg = background_batch_on_device['num_ohe']
# emb_manager_bg = engine.predictor_model.encoder.embedding_manager
# embeddings_output_bg = emb_manager_bg(background_batch_on_device)
# background_encoder_inputs = torch.cat([num_ohe_bg, embeddings_output_bg], dim=-1)
# Now use this `background_encoder_inputs` for DeepExplainer.