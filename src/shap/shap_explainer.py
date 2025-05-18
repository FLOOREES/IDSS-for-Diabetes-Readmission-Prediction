# src/shap/shap_explainer.py
import shap
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, Any, List, Optional

from src.inference.predictor_engine import SinglePatientPredictorEngine
from src.shap.shap_utils import RNNWrapperForSHAP, get_encoder_input_and_collated_batch 
from src import config as AppConfig

logger = logging.getLogger(__name__)

def generate_shap_explanation(
    engine: SinglePatientPredictorEngine,
    raw_patient_df_to_explain: pd.DataFrame,
    background_data_tensor_for_shap: torch.Tensor,
    config_numerical_features: List[str],
    config_ohe_actual_cols: List[str]
) -> Optional[Dict[str, Any]]:
    if raw_patient_df_to_explain.empty or engine.cfg.PATIENT_ID_COL not in raw_patient_df_to_explain:
        logger.error("Input raw_patient_df_to_explain is empty or missing patient ID column.")
        return None
    patient_id_for_log = raw_patient_df_to_explain[engine.cfg.PATIENT_ID_COL].iloc[0]
    logger.info(f"Generating SHAP explanations for all classes for patient ID: {patient_id_for_log}")

    instance_encoder_input, instance_collated_batch_cpu, actual_seq_len = \
        get_encoder_input_and_collated_batch(engine, raw_patient_df_to_explain)
    instance_on_device = instance_encoder_input.to(engine.device)

    if not hasattr(engine.predictor_model.encoder, 'rnn'):
        logger.error("EncoderRNN class does not have an 'rnn' attribute. SHAP setup needs adjustment.")
        return None
    shap_model_to_explain = RNNWrapperForSHAP(
        engine.predictor_model.encoder.rnn,
        engine.predictor_model.head
    ).to(engine.device)
    original_model_mode_is_training = shap_model_to_explain.training
    shap_model_to_explain.eval() # Start in eval, then switch for SHAP call

    background_on_device = background_data_tensor_for_shap.to(engine.device)
    explainer = shap.DeepExplainer(shap_model_to_explain, background_on_device)

    logger.info("Temporarily setting SHAP model to train() mode for DeepExplainer gradient calculation.")
    shap_model_to_explain.train()
    
    shap_values_raw_output = None
    try:
        shap_values_raw_output = explainer.shap_values(instance_on_device, check_additivity=False)
    except Exception as e:
        logger.error(f"Error during explainer.shap_values(): {e}", exc_info=True)
    finally:
        shap_model_to_explain.train(original_model_mode_is_training)
        logger.info(f"Restored SHAP model to original mode (training={original_model_mode_is_training}) post SHAP values attempt.")

    if shap_values_raw_output is None:
        logger.error("SHAP value calculation failed or returned None.")
        return None

    # --- MODIFIED HANDLING OF shap_values_raw_output ---
    logger.info(f"Raw SHAP output type: {type(shap_values_raw_output)}")
    list_of_shap_arrays_for_classes = [] # Will store arrays of shape (1, S, F)

    if isinstance(shap_values_raw_output, np.ndarray):
        logger.info(f"SHAP values returned as a single NumPy array. Shape: {shap_values_raw_output.shape}")
        # Expected shape (Batch_instance, SeqLen, Features_input, Num_Classes_output) -> (1, S, F, C)
        if shap_values_raw_output.ndim == 4:
            num_classes_from_shap = shap_values_raw_output.shape[-1]
            logger.info(f"Single NumPy array has {num_classes_from_shap} classes in the last dimension.")
            for i in range(num_classes_from_shap):
                # Slice out each class: result is (Batch_instance, SeqLen, Features_input)
                list_of_shap_arrays_for_classes.append(shap_values_raw_output[..., i]) 
        elif shap_values_raw_output.ndim == 3: # (Batch_instance, SeqLen, Features_input) -> single class output
            logger.info("Single NumPy array is 3D, assuming SHAP values for a single output class.")
            list_of_shap_arrays_for_classes = [shap_values_raw_output]
        else:
            logger.error(f"Unexpected shape for single NumPy array SHAP output: {shap_values_raw_output.shape}")
            return None
    elif isinstance(shap_values_raw_output, list): # Expected if DeepExplainer returns list per class
        logger.info(f"SHAP values returned as a list with {len(shap_values_raw_output)} elements.")
        list_of_shap_arrays_for_classes = shap_values_raw_output
    else:
        logger.error(f"Unexpected SHAP output format: {type(shap_values_raw_output)}. Cannot proceed.")
        return None
    
    if not list_of_shap_arrays_for_classes:
        logger.error("SHAP explanation yielded no value arrays after processing raw output.")
        return None
    # --- END MODIFIED HANDLING ---
    
    processed_all_class_shap_values = [] # This will store arrays of shape (actual_seq_len, Features)
    for class_shap_item_raw_np in list_of_shap_arrays_for_classes: # Each item is for one class
        if not isinstance(class_shap_item_raw_np, np.ndarray):
            logger.warning(f"Expected NumPy array for class SHAP values, got {type(class_shap_item_raw_np)}. Skipping.")
            processed_all_class_shap_values.append(None)
            continue
        
        # Expecting (batch_size=1, seq_len, features) for each class's SHAP array
        if class_shap_item_raw_np.ndim == 3 and class_shap_item_raw_np.shape[0] == 1:
            class_shap_squeezed_np = class_shap_item_raw_np[0] # Squeeze batch dim -> (seq_len, features)
        elif class_shap_item_raw_np.ndim == 2: # If already (seq_len, features)
            class_shap_squeezed_np = class_shap_item_raw_np
        else:
            logger.error(f"Unexpected shape for class SHAP values array: {class_shap_item_raw_np.shape}. Expected 3D (1,S,F) or 2D (S,F).")
            processed_all_class_shap_values.append(None)
            continue
            
        valid_class_shap_values = class_shap_squeezed_np[:actual_seq_len, :]
        processed_all_class_shap_values.append(valid_class_shap_values)
    
    processed_all_class_shap_values = [s for s in processed_all_class_shap_values if s is not None]
    if not processed_all_class_shap_values:
        logger.error("No SHAP values could be successfully processed for any class after filtering.")
        return None

    # Prepare feature names (logic remains the same)
    num_features_names = config_numerical_features 
    ohe_features_names = config_ohe_actual_cols
    total_emb_dim = engine.predictor_model.encoder.embedding_manager.get_total_embedding_dim()
    emb_feature_names = [f"emb_dim_{i+1}" for i in range(total_emb_dim)]
    encoder_input_feature_names = num_features_names + ohe_features_names + emb_feature_names

    explained_instance_data_np = instance_on_device[0, :actual_seq_len, :].detach().cpu().numpy()

    if len(encoder_input_feature_names) != explained_instance_data_np.shape[1]:
        mismatch_msg = (f"Feature name count ({len(encoder_input_feature_names)}) mismatch with "
                        f"explained instance features ({explained_instance_data_np.shape[1]}). Check feature name generation. "
                        f"NumConfig: {len(config_numerical_features)}, OHEConfig: {len(config_ohe_actual_cols)}, "
                        f"EmbDims: {engine.predictor_model.encoder.embedding_manager.get_total_embedding_dim()}") # Added more detail
        logger.error(mismatch_msg)
        encoder_input_feature_names = [f"Feature_{i}" for i in range(explained_instance_data_np.shape[1])]


    # Visualization (Summary Plot for multi-class)
    logger.info("Generating SHAP summary bar plot for all classes.")
    target_class_names = getattr(engine.cfg, 'TARGET_CLASS_NAMES', 
                                 [f"Class {i}" for i in range(len(processed_all_class_shap_values))])
    if len(target_class_names) != len(processed_all_class_shap_values):
        logger.warning("Mismatch between #TARGET_CLASS_NAMES and #SHAP output classes. Using default plot class names.")
        target_class_names = [f"Class {i}" for i in range(len(processed_all_class_shap_values))]

    try:
        # features argument for summary_plot should match the SHAP values structure
        # For list of SHAP arrays, features can be a single array (instance data) or a list too.
        shap.summary_plot(
            processed_all_class_shap_values, # List of (actual_seq_len, num_features) arrays
            features=explained_instance_data_np, # (actual_seq_len, num_features)
            feature_names=encoder_input_feature_names, 
            plot_type="bar", 
            show=False,
            class_names=target_class_names
        )
        # ... (saving plot logic remains the same) ...
        plt.title(f"SHAP Feature Importance (Patient ID: {patient_id_for_log})")
        shap_output_dir = os.path.join(AppConfig.RESULTS_DIR, "shap_explanations")
        os.makedirs(shap_output_dir, exist_ok=True)
        safe_patient_id_for_filename = str(patient_id_for_log).replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(shap_output_dir, f"patient_{safe_patient_id_for_filename}_summary_bar.png")
        plt.savefig(plot_path, bbox_inches='tight')
        logger.info(f"SHAP multi-class summary bar plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error during SHAP plot generation or saving: {e}", exc_info=True)

    base_values = explainer.expected_value
    if torch.is_tensor(base_values): base_values = base_values.detach().cpu().numpy()
    # explainer.expected_value for DeepExplainer on multi-output is usually a numpy array already
    if hasattr(base_values, 'tolist'): base_values = base_values.tolist()

    return {
        "patient_id": patient_id_for_log,
        "all_class_shap_values_sequence": processed_all_class_shap_values,
        "encoder_input_feature_names": encoder_input_feature_names,
        "explained_instance_encoder_input_np": explained_instance_data_np,
        "base_values_all_classes": base_values
    }