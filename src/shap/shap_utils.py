# In src/analysis/shap_utils.py
import torch
from typing import Dict, Any
from src.inference.predictor_engine import SinglePatientPredictorEngine
import pandas as pd
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

def get_encoder_input_for_shap(engine: SinglePatientPredictorEngine, raw_patient_df: pd.DataFrame) -> torch.Tensor:
    """
    Processes raw patient data to get the concatenated tensor input for the EncoderRNN's core RNN layers.
    This tensor includes numerical/OHE features and the outputs of the EmbeddingManager.
    """
    # Use engine's methods to get the collated batch
    # _preprocess_raw_patient_df and _prepare_model_input_batch are helper methods within the engine
    processed_df = engine._preprocess_raw_patient_df(raw_patient_df.copy())
    collated_batch = engine._prepare_model_input_batch(processed_df)
    batch_on_device = engine._move_batch_to_device(collated_batch) # Use the engine's device

    # Extract components needed to form the encoder_input_tensor
    num_ohe_tensor = batch_on_device['num_ohe']

    # Get the embedding_manager from the loaded model within the engine
    embedding_manager = engine.predictor_model.encoder.embedding_manager
    # EmbeddingManager expects the batch dict to find 'learned_labels' and 'precomputed_labels'
    embeddings_output = embedding_manager(batch_on_device) 

    encoder_input_tensor = torch.cat([num_ohe_tensor, embeddings_output], dim=-1)
    return encoder_input_tensor

def prepare_shap_background_data(engine: SinglePatientPredictorEngine, 
                                 df_train_sample: pd.DataFrame, # A sample from your training data
                                 patient_id_col: str, 
                                 timestamp_col: str) -> torch.Tensor:
    """Prepares background data for SHAP DeepExplainer."""
    background_tensors = []
    # Group df_train_sample by patient if it contains multiple patients,
    # or assume it's a list of single-patient DataFrames.
    # For simplicity, let's assume df_train_sample is iterated patient by patient.
    # This part needs to be adapted based on how df_train_sample is structured.
    # If df_train_sample contains multiple patients, group it first.

    unique_patients = df_train_sample[patient_id_col].unique()
    logger.info(f"Preparing SHAP background data from {len(unique_patients)} sample patients...")

    for pid in unique_patients:
        try:
            patient_df_raw = df_train_sample[df_train_sample[patient_id_col] == pid]
            # Ensure patient_df_raw is sorted correctly by timestamp if not already
            patient_df_raw = patient_df_raw.sort_values(by=timestamp_col) 

            encoder_input = get_encoder_input_for_shap(engine, patient_df_raw)
            background_tensors.append(encoder_input)
        except Exception as e:
            logger.warning(f"Could not process patient {pid} for SHAP background: {e}")
            continue

    if not background_tensors:
        raise ValueError("No background data could be prepared for SHAP.")

    return torch.cat(background_tensors, dim=0) # Shape: (num_bg_samples * 1, seq_len, features_for_rnn)
                                                # Note: each patient is a "sample" of batch size 1 here.
                                                # DeepExplainer usually expects (num_samples, *data_shape)
                                                # So if each patient has varying seq_len, this direct cat might be an issue
                                                # if not all sequences are padded to the same length *before* this cat.
                                                # pad_collate_fn already handles padding to max_len *within that batch*.
                                                # For a background set, all encoder_input_tensors should have the same seq_len (max_len of that patient).
                                                # It's better if all background samples are padded to a common max_seq_length.
                                                # The current get_encoder_input_for_shap will return varying seq_len.
                                                #
                                                # Simpler for background: ensure all background samples have same seq_len
                                                # by processing them through pad_collate_fn in batches or one by one
                                                # and then stacking their encoder_input parts.
                                                #
                                                # Let's assume for MVP, background_tensors will be a list of tensors,
                                                # and DeepExplainer might take a list of tensors if sizes vary, or one tensor.
                                                # The standard is one tensor: (N, S, F)
                                                # This needs all sequences to be padded to the SAME max_seq_length.
                                                # The current pad_collate_fn pads per batch.
                                                #
                                                # Easiest way: process a batch of background patients using the DataLoader
                                                # and then extract the encoder_input parts.
                                                # For now, let's assume we get a representative tensor.
                                                # A common approach is to select a subset of training data, process it
                                                # through the same data prep and collate, and then form the background tensor.
                                                # Let's defer precise background padding strategy for a moment.
                                                # For DeepExplainer, it's often just one tensor.
    # Placeholder: this needs robust handling of sequence lengths for background set.
    # For now, let's assume we take a single, representative background_encoder_input.
    # Or a small number of them if they are all padded to the same max_seq_length.
    # A common method is to use a subset of the training data. For example, one batch.

    # Let's load one batch from train_loader and get its encoder_input as background
    # This requires access to train_loader or recreation of it.
    # For MVP, a small, fixed background set is easier.
    # We'll use the first background_tensor for now for simplicity,
    # or stack them if get_encoder_input_for_shap ensures fixed seq len.
    # The collate_fn output will have consistent seq_len within its batch.
    #
    # Ideal way for background:
    # 1. Load df_train.
    # 2. Get a sample (e.g., 100 rows for a few patients).
    # 3. Preprocess this sample fully (Phase1, Phase2).
    # 4. Use data_preparer.transform() on this processed sample.
    # 5. Use PatientSequenceDataset and DataLoader (with pad_collate_fn) to get a few batches.
    # 6. For each batch, compute the encoder_input_tensor.
    # 7. Concatenate these to form a larger background tensor.
    # This is too complex for here.
    # For now, let's assume `background_tensors` contains one or more tensors of shape (1, S_i, F)
    # and we will use `torch.cat(background_tensors, dim=0)` if S_i is the same, or just use one.
    
class RNNWrapperForSHAP(nn.Module):
    def __init__(self, encoder_rnn_cell: nn.Module, prediction_head: nn.Module):
        super().__init__()
        self.encoder_rnn_cell = encoder_rnn_cell # This is engine.predictor_model.encoder.rnn
        self.prediction_head = prediction_head   # This is engine.predictor_model.head

    def forward(self, encoder_input_tensor: torch.Tensor) -> torch.Tensor:
        # encoder_input_tensor is (batch, seq_len, features_for_rnn)
        # The nn.LSTM/GRU layer can take this directly. 
        # It initializes hidden state to zeros if not provided.
        rnn_outputs, _ = self.encoder_rnn_cell(encoder_input_tensor) 
        logits = self.prediction_head(rnn_outputs) # (batch, seq_len, num_classes)
        return logits