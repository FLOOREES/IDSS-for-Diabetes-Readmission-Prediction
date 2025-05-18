# src/data_preparation/collators.py
import torch
from typing import List, Dict, Any, Optional
import logging

# It's good practice to have a logger available, even if not heavily used here.
logger = logging.getLogger(__name__)

def pad_collate_fn(batch: List[Dict[str, Any]], 
                   enforced_max_len: Optional[int] = None) -> Dict[str, Any]:
    """
    Collate function to pad sequences within a batch and restructure features.

    Args:
        batch: A list of dictionaries, where each dictionary is the output
               of PatientSequenceDataset.__getitem__.
        enforced_max_len (Optional[int]): If provided, all sequences in the batch
                                           will be padded to this specific length.
                                           Original lengths are still used for the mask.
                                           If None, pads to the max length within the current batch.
    Returns:
        A dictionary containing padded tensors for each feature type,
        padded targets, the mask, original lengths, and patient IDs.
        Output structure:
        {
            'num_ohe': torch.Tensor (batch_size, padding_target_len, num_ohe_dim),
            'learned_labels': Dict[str, torch.Tensor] (col_name -> (batch_size, padding_target_len)),
            'precomputed_labels': Dict[str, torch.Tensor] (col_name -> (batch_size, padding_target_len)),
            'targets': torch.Tensor (batch_size, padding_target_len),
            'mask': torch.Tensor (batch_size, padding_target_len), (True for actual data)
            'lengths': torch.Tensor (batch_size,), (Original sequence lengths)
            'patient_id': List[Any] (batch_size,)
        }
    """
    if not batch:
        logger.warning("pad_collate_fn received an empty batch.")
        return {} # Or raise error, depending on desired behavior

    # Original lengths are crucial for masking and knowing actual content
    original_lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    # Determine the target length for padding for this batch
    if enforced_max_len is not None:
        padding_target_len = enforced_max_len
        # We assume SequenceDataPreparer has already truncated sequences if they were longer
        # than enforced_max_len, so original_lengths should be <= enforced_max_len.
        # If any original_length > enforced_max_len, it implies an issue upstream.
        # For safety, if an original_length is somehow greater, the mask will still be correct up to padding_target_len.
    else:
        padding_target_len = torch.max(original_lengths).item() if original_lengths.numel() > 0 else 0
    
    # If all sequences in the batch are empty (original_lengths all 0) 
    # and no enforced_max_len, padding_target_len could be 0.
    # Models usually expect a sequence dimension. If so, set a minimum, e.g., 1.
    # However, if enforced_max_len is set (e.g. 50), padding_target_len will be 50.
    if padding_target_len == 0 and batch and enforced_max_len is None:
        logger.debug("All sequences in batch have original length 0 and no enforced_max_len. Setting padding_target_len to 1.")
        padding_target_len = 1


    # --- Initialize lists/dicts to hold data for each feature type before stacking ---
    num_ohe_sequences = []
    # Safely get keys for label dicts from the first item that has features
    # (assuming all items in batch have the same feature structure if they have features)
    first_valid_item_features_visit = None
    for item_check in batch:
        if item_check['length'] > 0 and item_check['features']:
            first_valid_item_features_visit = item_check['features'][0] # First visit of first valid patient
            break
    
    learned_label_keys = list(first_valid_item_features_visit['learned_labels'].keys()) \
        if first_valid_item_features_visit and 'learned_labels' in first_valid_item_features_visit else []
    precomputed_label_keys = list(first_valid_item_features_visit['precomputed_labels'].keys()) \
        if first_valid_item_features_visit and 'precomputed_labels' in first_valid_item_features_visit else []

    learned_label_sequences = {col_name: [] for col_name in learned_label_keys}
    precomputed_label_sequences = {col_name: [] for col_name in precomputed_label_keys}
    
    target_sequences = []
    patient_ids = [item['patient_id'] for item in batch]

    # Determine expected_num_ohe_dim for padding completely empty sequences
    # This should be consistent for all visits if features exist.
    expected_num_ohe_dim = 0
    if first_valid_item_features_visit and 'num_ohe' in first_valid_item_features_visit:
        expected_num_ohe_dim = first_valid_item_features_visit['num_ohe'].shape[0]
    elif batch and batch[0]['features'] and batch[0]['features'][0]['num_ohe'] is not None: # Fallback to first item even if empty sequence
        expected_num_ohe_dim = batch[0]['features'][0]['num_ohe'].shape[0]
    # If still 0, it implies no num_ohe features configured or an issue.
    # MODEL_INPUT_NUM_OHE_DIM from config is the ultimate truth for this.
    # A collator usually infers. For robustness, this could be passed if it's problematic.


    # --- Iterate through the batch, extract data up to original_len, and pad to padding_target_len ---
    for i, item in enumerate(batch):
        feature_seq = item['features']    # List of visit dicts for this patient
        target_seq_orig = item['targets'] # Original target Tensor for this patient
        original_len = original_lengths[i].item() # Original length of this patient's sequence
        
        # Length of actual content to take from original sequence (cannot exceed padding_target_len)
        len_to_take = min(original_len, padding_target_len)
        padding_needed = padding_target_len - len_to_take
        
        # --- Pad numerical/OHE features ---
        if len_to_take > 0:
            num_ohe_data_tensors = [visit['num_ohe'] for visit in feature_seq[:len_to_take]]
            # All tensors in num_ohe_data_tensors must have the same feature dimension
            current_num_ohe_dim = num_ohe_data_tensors[0].shape[0] if num_ohe_data_tensors else expected_num_ohe_dim
            stacked_num_ohe = torch.stack(num_ohe_data_tensors, dim=0) if num_ohe_data_tensors else \
                              torch.empty(0, current_num_ohe_dim, dtype=torch.float32)
        else: # This patient's sequence part to consider is empty
            current_num_ohe_dim = expected_num_ohe_dim
            stacked_num_ohe = torch.empty(0, current_num_ohe_dim, dtype=torch.float32)
        
        padding_tensor_num_ohe = torch.zeros(padding_needed, current_num_ohe_dim, dtype=torch.float32)
        padded_num_ohe = torch.cat([stacked_num_ohe, padding_tensor_num_ohe], dim=0)
        num_ohe_sequences.append(padded_num_ohe)

        # --- Pad learned embedding labels ---
        for col_name in learned_label_keys:
            labels = [visit['learned_labels'][col_name] for visit in feature_seq[:len_to_take]] if len_to_take > 0 else []
            padded_labels = torch.cat(
                [torch.tensor(labels, dtype=torch.long)] + \
                [torch.zeros(padding_needed, dtype=torch.long)] # Pad with 0
            , dim=0)
            learned_label_sequences[col_name].append(padded_labels)

        # --- Pad precomputed embedding labels ---
        for col_name in precomputed_label_keys:
            labels = [visit['precomputed_labels'][col_name] for visit in feature_seq[:len_to_take]] if len_to_take > 0 else []
            padded_labels = torch.cat(
                [torch.tensor(labels, dtype=torch.long)] + \
                [torch.zeros(padding_needed, dtype=torch.long)] # Pad with 0
            , dim=0)
            precomputed_label_sequences[col_name].append(padded_labels)

        # --- Pad targets ---
        padding_tensor_targets = torch.zeros(padding_needed, dtype=torch.long) # Pad with 0
        padded_targets = torch.cat([target_seq_orig[:len_to_take], padding_tensor_targets], dim=0)
        target_sequences.append(padded_targets)

    # --- Stack sequences into batch tensors ---
    collated_batch = {}
    if num_ohe_sequences:
        try:
            collated_batch['num_ohe'] = torch.stack(num_ohe_sequences, dim=0)
        except RuntimeError as e:
            logger.error(f"Error stacking num_ohe_sequences (likely inconsistent feature dims). Shapes: {[s.shape for s in num_ohe_sequences]}. Error: {e}")
            # Fallback to a consistently shaped empty tensor or raise error
            # This indicates a problem in how expected_num_ohe_dim was determined or in data itself.
            collated_batch['num_ohe'] = torch.empty(len(batch), padding_target_len, expected_num_ohe_dim, dtype=torch.float32)

    else: # Should only happen if batch was empty initially
        collated_batch['num_ohe'] = torch.empty(len(batch), padding_target_len, expected_num_ohe_dim, dtype=torch.float32)

    collated_batch['learned_labels'] = {
        k: torch.stack(v, dim=0) for k, v in learned_label_sequences.items() if v # Ensure list is not empty
    }
    collated_batch['precomputed_labels'] = {
        k: torch.stack(v, dim=0) for k, v in precomputed_label_sequences.items() if v # Ensure list is not empty
    }
    collated_batch['targets'] = torch.stack(target_sequences, dim=0)
    collated_batch['lengths'] = original_lengths # CRITICAL: Use original_lengths for the mask
    collated_batch['patient_id'] = patient_ids

    # --- Create mask based on original_lengths and the final padding_target_len ---
    # Mask is True for actual data, False for padding.
    mask = torch.arange(padding_target_len)[None, :] < original_lengths[:, None]
    collated_batch['mask'] = mask

    return collated_batch