import torch
import torch.nn as nn # Ensure nn is imported for RNNWrapperForSHAP
import pandas as pd
from typing import Dict, Any, List, Tuple # Added Tuple
from src.inference.predictor_engine import SinglePatientPredictorEngine # Your engine
from src.data_preparation.datasets import PatientSequenceDataset # For background data
from src.data_preparation.collators import pad_collate_fn      # For background data
from torch.utils.data import DataLoader                       # For background data
import logging

logger = logging.getLogger(__name__)

def get_encoder_input_and_collated_batch(engine: SinglePatientPredictorEngine, 
                                         raw_patient_df: pd.DataFrame) -> Tuple[torch.Tensor, Dict[str, Any], int]:
    """
    Processes a single patient's raw data to get:
    1. The concatenated tensor input for the EncoderRNN's core RNN layers (on device).
    2. The collated batch (on CPU) which contains lengths and other info.
    3. The actual sequence length for this patient.
    """
    logger.debug(f"Getting encoder input and collated batch for patient data shape: {raw_patient_df.shape}")
    # These are helper methods within the engine:
    # _preprocess_raw_patient_df applies Phase1 and Phase2
    processed_df = engine._preprocess_raw_patient_df(raw_patient_df.copy())
    
    # _prepare_model_input_batch applies SequenceDataPreparer.transform and pad_collate_fn
    # This collated_batch_cpu is on CPU and contains all keys like 'num_ohe', 'learned_labels', etc.
    # as well as 'lengths', 'mask', 'patient_id'.
    collated_batch_cpu = engine._prepare_model_input_batch(processed_df) 
    
    # Now, create the encoder_input_tensor using this collated_batch, moving necessary parts to device
    batch_on_device = engine._move_batch_to_device(collated_batch_cpu) 

    num_ohe_tensor_device = batch_on_device['num_ohe']
    # Ensure embedding_manager and its components are on the correct device (usually handled by engine init)
    embedding_manager = engine.predictor_model.encoder.embedding_manager
    
    # EmbeddingManager's forward pass expects the batch dictionary with labels on the correct device
    embeddings_output_device = embedding_manager(batch_on_device) 
    
    encoder_input_tensor_device = torch.cat([num_ohe_tensor_device, embeddings_output_device], dim=-1)
    
    # Get the actual sequence length for this single patient from the collated_batch (which is on CPU)
    # The collated_batch['lengths'] is a tensor of shape (1,) for a single patient.
    actual_seq_len = collated_batch_cpu['lengths'][0].item() 
    
    logger.debug(f"encoder_input_tensor shape: {encoder_input_tensor_device.shape}, actual_seq_len: {actual_seq_len}")
    
    # Return: 1. Tensor for SHAP (on device), 2. Original collated batch (on CPU), 3. Length
    return encoder_input_tensor_device, collated_batch_cpu, actual_seq_len 

def prepare_shap_background_data(engine: SinglePatientPredictorEngine, 
                                 df_background_raw_all_patients: pd.DataFrame, # Combined raw data for ALL bg patients
                                 num_background_sequences_aim: int = 50 
                                ) -> torch.Tensor:
    """
    Prepares a background dataset tensor for SHAP's DeepExplainer.
    All sequences in the output tensor will have consistent padding.
    """
    logger.info(f"Preparing SHAP background data from combined raw DataFrame (num visits: {len(df_background_raw_all_patients)}), "
                f"aiming for ~{num_background_sequences_aim} patient sequences.")
    
    # 1. Preprocess all background patients' raw data together (Phase 1 & 2)
    # This uses the engine's preprocessors with their loaded artifacts
    processed_background_df = engine._preprocess_raw_patient_df(df_background_raw_all_patients.copy())

    # 2. Use SequenceDataPreparer to get all sequences
    temp_target_col_name = engine.data_preparer.target_col
    # Work on a copy for adding dummy target
    df_for_sdp = processed_background_df.copy()
    if temp_target_col_name not in df_for_sdp.columns:
        df_for_sdp[temp_target_col_name] = 0 # Add dummy target
    
    all_feature_seqs, all_target_seqs, all_pids = engine.data_preparer.transform(df_for_sdp)

    if not all_feature_seqs:
        raise ValueError("No sequences generated from the background raw data for SHAP.")

    # Take up to num_background_sequences_aim sequences
    num_actual_bg_sequences = min(num_background_sequences_aim, len(all_feature_seqs))
    if num_actual_bg_sequences == 0:
        raise ValueError("Zero background sequences selected. Cannot proceed with SHAP.")

    logger.info(f"Using {num_actual_bg_sequences} patient sequences for the background set.")
    
    bg_feature_sequences_sample = all_feature_seqs[:num_actual_bg_sequences]
    bg_target_sequences_sample = all_target_seqs[:num_actual_bg_sequences]
    bg_pids_sample = all_pids[:num_actual_bg_sequences]

    # 3. Create a Dataset and a single DataLoader to get one consistently padded batch
    # This ensures all sequences in background_collated_batch are padded to the same length (max_len within this combined set)
    background_dataset = PatientSequenceDataset(bg_feature_sequences_sample, bg_target_sequences_sample, bg_pids_sample)
    
    # Batch size = len(dataset) to process all as one batch for consistent padding
    background_dataloader = DataLoader(background_dataset, batch_size=len(background_dataset), 
                                       shuffle=False, collate_fn=pad_collate_fn) 
    background_collated_batch_cpu = next(iter(background_dataloader))

    # 4. Get the encoder_input_tensor for this entire background batch
    background_batch_on_device = engine._move_batch_to_device(background_collated_batch_cpu)
    
    bg_num_ohe_tensor = background_batch_on_device['num_ohe']
    embedding_manager = engine.predictor_model.encoder.embedding_manager
    bg_embeddings_output = embedding_manager(background_batch_on_device)
    
    final_background_encoder_inputs = torch.cat([bg_num_ohe_tensor, bg_embeddings_output], dim=-1)
    
    logger.info(f"SHAP background data tensor prepared. Shape: {final_background_encoder_inputs.shape}")
    return final_background_encoder_inputs

class RNNWrapperForSHAP(nn.Module): # This wrapper is good as is.
    def __init__(self, encoder_rnn_module: nn.Module, prediction_head: nn.Module):
        super().__init__()
        self.encoder_rnn_module = encoder_rnn_module 
        self.prediction_head = prediction_head   
    
    def forward(self, encoder_input_tensor: torch.Tensor) -> torch.Tensor:
        rnn_outputs, _ = self.encoder_rnn_module(encoder_input_tensor) 
        logits = self.prediction_head(rnn_outputs)
        return logits