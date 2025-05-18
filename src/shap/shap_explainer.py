import shap
import pandas as pd
import numpy as np # Ensure numpy is imported
import torch
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, Any, List, Optional # Ensure all are imported

from src.inference.predictor_engine import SinglePatientPredictorEngine
# Updated import for get_encoder_input_and_collated_batch
from src.shap.shap_utils import RNNWrapperForSHAP, get_encoder_input_and_collated_batch 
from src import config as AppConfig # For RESULTS_DIR and TARGET_CLASS_NAMES

logger = logging.getLogger(__name__)

def generate_shap_explanation(
    engine: SinglePatientPredictorEngine, 
    raw_patient_df_to_explain: pd.DataFrame, 
    background_data_tensor_for_shap: torch.Tensor,
    config_numerical_features: List[str], 
    config_ohe_actual_cols: List[str]
):
    patient_id_for_log = raw_patient_df_to_explain[engine.cfg.PATIENT_ID_COL].iloc[0]
    logger.info(f"Generating SHAP explanations for all classes for patient ID: {patient_id_for_log}")

    # 1. Prepare the instance to explain - now also gets the collated batch for length info
    instance_encoder_input, instance_collated_batch_cpu, actual_seq_len = \
        get_encoder_input_and_collated_batch(engine, raw_patient_df_to_explain) # MODIFIED
    
    instance_on_device = instance_encoder_input.to(engine.device)

    # 2. Define the SHAP model 
    if not hasattr(engine.predictor_model.encoder, 'rnn'): # Or however you access the core RNN (LSTM/GRU)
        raise AttributeError("EncoderRNN class does not have an 'rnn' attribute. SHAP setup needs adjustment.")
    shap_model_to_explain = RNNWrapperForSHAP(
        engine.predictor_model.encoder.rnn, 
        engine.predictor_model.head
    ).to(engine.device)
    shap_model_to_explain.eval()

    # 3. Create DeepExplainer
    background_on_device = background_data_tensor_for_shap.to(engine.device)
    explainer = shap.DeepExplainer(shap_model_to_explain, background_on_device)

    # 4. Get SHAP values for ALL classes
    shap_values_all_classes_raw = explainer.shap_values(instance_on_device)
    logger.info(f"SHAP values generated for {len(shap_values_all_classes_raw)} classes.")

    # 5. Process SHAP values for all classes
    # actual_seq_len is now available from get_encoder_input_and_collated_batch
    
    processed_all_class_shap_values = []
    for class_shap_values_tensor in shap_values_all_classes_raw:
        class_shap_values_single_instance = class_shap_values_tensor[0].cpu().numpy() 
        valid_class_shap_values = class_shap_values_single_instance[:actual_seq_len, :]
        processed_all_class_shap_values.append(valid_class_shap_values)
    
    # Prepare feature names
    num_features_cfg = config_numerical_features 
    ohe_features_cfg = config_ohe_actual_cols
    total_emb_dim = engine.predictor_model.encoder.embedding_manager.get_total_embedding_dim()
    emb_feature_names = [f"emb_dim_{i+1}" for i in range(total_emb_dim)]
    # Order in encoder_input_tensor: num_ohe_tensor (numerical + ohe), then embeddings_output
    # Order from SequenceDataPreparer's 'num_ohe' key: numerical_features then actual_ohe_columns
    encoder_input_feature_names = num_features_cfg + ohe_features_cfg + emb_feature_names

    instance_encoder_input_np = instance_encoder_input[0, :actual_seq_len, :].cpu().numpy()

    if len(encoder_input_feature_names) != processed_all_class_shap_values[0].shape[1]:
        mismatch_msg = (f"Feature name count ({len(encoder_input_feature_names)}) mismatch with SHAP features "
                        f"({processed_all_class_shap_values[0].shape[1]}). Check feature name generation carefully. "
                        f"NumFeats: {len(num_features_cfg)}, OHEFeats: {len(ohe_features_cfg)}, EmbDims: {total_emb_dim}")
        logger.error(mismatch_msg)
        # Fallback to generic names if there's an error, to prevent crash during plotting
        encoder_input_feature_names = [f"Feature_{i}" for i in range(processed_all_class_shap_values[0].shape[1])]


    # 6. Visualization (Summary Plot for multi-class)
    logger.info("Generating SHAP summary bar plot for all classes.")
    
    # Get class names from config if available, else generate default
    target_class_names = getattr(engine.cfg, 'TARGET_CLASS_NAMES', 
                                 [f"Class {i}" for i in range(len(processed_all_class_shap_values))])
    if len(target_class_names) != len(processed_all_class_shap_values):
        logger.warning(f"Mismatch between #TARGET_CLASS_NAMES ({len(target_class_names)}) and #output classes ({len(processed_all_class_shap_values)}). Using default class names for plot.")
        target_class_names = [f"Class {i}" for i in range(len(processed_all_class_shap_values))]

    shap.summary_plot(
        processed_all_class_shap_values, 
        features=instance_encoder_input_np, 
        feature_names=encoder_input_feature_names, 
        plot_type="bar", 
        show=False,
        class_names=target_class_names
    )
    plt.title(f"SHAP Feature Importance (Patient ID: {patient_id_for_log})")
    
    shap_output_dir = os.path.join(AppConfig.RESULTS_DIR, "shap_explanations")
    os.makedirs(shap_output_dir, exist_ok=True)
    plot_path = os.path.join(shap_output_dir, f"patient_{str(patient_id_for_log).replace('/', '_')}_summary_bar.png") # Sanitize patient_id if it can have slashes
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        logger.info(f"SHAP multi-class summary bar plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save SHAP summary plot: {e}")
    plt.close()

    # Base values for DeepExplainer (typically one per class)
    base_values = explainer.expected_value
    if hasattr(base_values, 'cpu'): base_values = base_values.cpu()
    if hasattr(base_values, 'numpy'): base_values = base_values.numpy()
    if hasattr(base_values, 'tolist'): base_values = base_values.tolist()


    return {
        "patient_id": patient_id_for_log,
        "all_class_shap_values_sequence": processed_all_class_shap_values,
        "encoder_input_feature_names": encoder_input_feature_names,
        "explained_instance_encoder_input_np": instance_encoder_input_np,
        "base_values_all_classes": base_values
    }