# outlier_detection_pytorch.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
import logging
from typing import List, Tuple, Dict, Optional, Union
import time
import os

# Import from our new modules
from recurrent_data import SequenceDataPreparer, PatientSequenceDataset, collate_fn_pad
from model import EncoderRNN, AdditiveAttention, DecoderRNN, Seq2SeqAE

logger = logging.getLogger(__name__)

class SequenceOutlierDetectorPyTorch:
    """
    Detects outliers using a PyTorch LSTM/GRU Attention Autoencoder.
    """
    def __init__(
        self,
        data_preparer: SequenceDataPreparer,
        patient_id_col: str = 'patient_nbr',
        # Model Hyperparameters
        input_dim: int = -1, # Will be determined from data
        hidden_dim: int = 64,
        encoder_layers: int = 1,
        decoder_layers: int = 1, # Often same as encoder
        attention_dim: int = 32,
        dropout: float = 0.2,
        use_gru: bool = False,
        # Training Hyperparameters
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        optimizer_name: str = 'adam', # 'adam' or 'rmsprop'
        weight_decay: float = 1e-5, # L2 regularization
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        lr_scheduler_patience: int = 5,
        lr_scheduler_factor: float = 0.1,
        device: Optional[torch.device] = None,
        # Outlier Detection Parameters
        error_percentile_threshold: float = 95.0,
        logger: logging.Logger = None
    ):
        self.data_preparer = data_preparer
        self.patient_id_col = patient_id_col
        # Model params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.use_gru = use_gru
        # Training params
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        # Outlier params
        self.error_percentile_threshold = error_percentile_threshold
        self.logger = logger or logging.getLogger(__name__)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.model: Optional[Seq2SeqAE] = None
        self.history: Dict[str, List] = {'train_loss': [], 'val_loss': []}
        self.visit_error_threshold: Optional[float] = None
        self.training_patient_ids: Optional[list] = None
        self.validation_patient_ids: Optional[list] = None
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def _patient_train_val_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List, List]:
        """Performs patient-level train/validation split."""
        # (Same logic as Keras version)
        self.logger.info(f"Performing patient-level train/validation split...")
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.validation_split, random_state=42)
        patient_ids = df[self.patient_id_col].unique()
        indices = np.arange(len(df))
        train_idx_indices, val_idx_indices = next(splitter.split(indices, groups=df[self.patient_id_col]))
        train_df = df.iloc[train_idx_indices]
        val_df = df.iloc[val_idx_indices]
        train_patients = train_df[self.patient_id_col].unique()
        val_patients = val_df[self.patient_id_col].unique()
        self.logger.info(f"Split: {len(train_patients)} train patients, {len(val_patients)} validation patients.")
        overlap = set(train_patients) & set(val_patients)
        if overlap: self.logger.warning(f"Overlap detected: {len(overlap)} patients.")
        return train_df, val_df, list(train_patients), list(val_patients)

    def _build_model(self):
        """Builds the PyTorch Seq2SeqAE model."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be set (usually from data) before building model.")

        encoder = EncoderRNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.encoder_layers,
            dropout=self.dropout,
            use_gru=self.use_gru
        )
        attention = AdditiveAttention(
            encoder_hidden_dim=self.hidden_dim, # Assuming encoder output dim is hidden_dim
            decoder_hidden_dim=self.hidden_dim, # Assuming decoder hidden dim is same
            attention_dim=self.attention_dim
        )
        decoder = DecoderRNN(
            output_dim=self.input_dim, # Reconstruct original features
            hidden_dim=self.hidden_dim,
            encoder_hidden_dim=self.hidden_dim,
            attention=attention,
            n_layers=self.decoder_layers,
            dropout=self.dropout,
            use_gru=self.use_gru
        )
        self.model = Seq2SeqAE(encoder, decoder).to(self.device)
        self.logger.info("PyTorch Seq2SeqAE model built.")
        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,}")


    def fit(self, df: pd.DataFrame):
        """Trains the PyTorch Autoencoder."""
        self.logger.info("Starting outlier detector fitting process (PyTorch)...")

        # 1. Train/Validation Split
        df_train, df_val, self.training_patient_ids, self.validation_patient_ids = self._patient_train_val_split(df)

        # 2. Fit Scaler on Training Data
        self.data_preparer.fit_scaler(df_train)

        # 3. Prepare Sequences (Lists of Tensors)
        train_sequences, train_pids = self.data_preparer.create_sequences_and_ids(df_train)
        val_sequences, val_pids = self.data_preparer.create_sequences_and_ids(df_val)

        # Determine input_dim from data
        if not train_sequences:
            raise ValueError("No training sequences generated. Check data and preparation steps.")
        self.input_dim = train_sequences[0].shape[1]
        self.logger.info(f"Determined input_dim: {self.input_dim}")

        # 4. Create Datasets and DataLoaders
        train_dataset = PatientSequenceDataset(train_sequences, train_pids)
        val_dataset = PatientSequenceDataset(val_sequences, val_pids)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn_pad)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn_pad)

        # 5. Build Model
        self._build_model()

        # 6. Setup Optimizer, Loss, Scheduler
        if self.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == 'rmsprop':
             optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
             raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        criterion = nn.MSELoss(reduction='none') # Calculate loss per element initially for masking
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience, verbose=True
        )

        # 7. Training Loop
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        start_time_total = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss_epoch = 0.0
            total_train_steps = 0

            for batch_idx, (sequences, lengths, masks) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                masks = masks.to(self.device) # (batch, seq_len) boolean mask

                optimizer.zero_grad()
                reconstructions = self.model(sequences, attention_mask=masks) # Pass mask to model

                # Calculate masked loss
                unmasked_loss = criterion(reconstructions, sequences) # (batch, seq_len, features)
                # Expand mask for features dim: (batch, seq_len, 1)
                mask_expanded = masks.unsqueeze(-1).float()
                masked_loss = unmasked_loss * mask_expanded # Zero out loss for padded steps

                # Average loss over non-padded elements in the batch
                # Sum loss per batch, divide by number of non-padded elements
                batch_loss = masked_loss.sum() / mask_expanded.sum()

                batch_loss.backward()
                # Optional: Gradient Clipping
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_epoch += batch_loss.item() * sequences.size(0) # Loss * batch size
                total_train_steps += sequences.size(0) # Count samples

                if batch_idx % 100 == 0: # Log progress
                    self.logger.debug(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {batch_loss.item():.4f}")


            avg_train_loss = train_loss_epoch / total_train_steps if total_train_steps > 0 else 0.0
            self.history['train_loss'].append(avg_train_loss)

            # Validation Step
            self.model.eval()
            val_loss_epoch = 0.0
            total_val_steps = 0
            with torch.no_grad():
                for sequences, lengths, masks in val_loader:
                    sequences = sequences.to(self.device)
                    masks = masks.to(self.device)
                    reconstructions = self.model(sequences, attention_mask=masks)
                    unmasked_loss = criterion(reconstructions, sequences)
                    mask_expanded = masks.unsqueeze(-1).float()
                    masked_loss = unmasked_loss * mask_expanded
                    batch_loss = masked_loss.sum() / mask_expanded.sum() # Average loss

                    val_loss_epoch += batch_loss.item() * sequences.size(0)
                    total_val_steps += sequences.size(0)

            avg_val_loss = val_loss_epoch / total_val_steps if total_val_steps > 0 else 0.0
            self.history['val_loss'].append(avg_val_loss)
            epoch_duration = time.time() - epoch_start_time

            self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {epoch_duration:.2f}s")

            # LR Scheduling and Early Stopping
            scheduler.step(avg_val_loss)

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                # Save best model checkpoint
                self.save_model("best_model_checkpoint.pth")
                self.logger.info(f"Validation loss improved. Saved checkpoint.")
            else:
                self.epochs_no_improve += 1
                self.logger.info(f"Validation loss did not improve for {self.epochs_no_improve} epoch(s).")

            if self.epochs_no_improve >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        total_training_time = time.time() - start_time_total
        self.logger.info(f"Training completed in {total_training_time:.2f} seconds.")

        # Load best model weights
        self.load_model("best_model_checkpoint.pth")
        self.logger.info("Loaded best model weights based on validation loss.")

        # Calculate Outlier Threshold on Training Reconstruction Errors
        self._calculate_thresholds(train_loader)


    def _calculate_thresholds(self, train_loader: DataLoader):
        """Calculates reconstruction error thresholds on training data."""
        self.logger.info("Calculating reconstruction error threshold on training data...")
        if self.model is None:
            raise RuntimeError("Model must be trained before calculating thresholds.")

        self.model.eval()
        all_step_errors = []
        criterion = nn.L1Loss(reduction='none') # Use MAE per element for threshold

        with torch.no_grad():
            for sequences, lengths, masks in train_loader:
                sequences = sequences.to(self.device)
                masks = masks.to(self.device) # (batch, seq_len)
                reconstructions = self.model(sequences, attention_mask=masks)

                # Calculate MAE per step, sum across features
                mae_per_element = criterion(reconstructions, sequences) # (batch, seq_len, features)
                mae_per_step = torch.sum(mae_per_element, dim=2) # (batch, seq_len)

                # Apply mask: Keep only errors for valid steps
                # masked_select returns a 1D tensor of valid errors
                valid_errors = torch.masked_select(mae_per_step, masks)
                all_step_errors.extend(valid_errors.cpu().numpy())

        if not all_step_errors:
            self.logger.warning("No valid errors found in training data to calculate threshold.")
            self.visit_error_threshold = float('inf')
        else:
            self.visit_error_threshold = np.percentile(all_step_errors, self.error_percentile_threshold)
            self.logger.info(f"Visit-level MAE threshold ({self.error_percentile_threshold}th percentile): {self.visit_error_threshold:.4f}")


    def detect_outliers(self, df: pd.DataFrame, return_embeddings: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Detects outliers using the trained PyTorch model."""
        if self.model is None or self.visit_error_threshold is None:
            raise RuntimeError("Model must be trained (`fit`) before detecting outliers.")

        self.logger.info(f"Detecting outliers in DataFrame with shape {df.shape}...")

        # 1. Prepare Sequences and DataLoader
        sequences_list, patient_ids_list = self.data_preparer.create_sequences_and_ids(df)
        if not sequences_list:
             self.logger.warning("No sequences generated from the input DataFrame.")
             df['reconstruction_error'] = np.nan
             df['is_outlier_visit'] = False
             if return_embeddings:
                 # Return empty embeddings DF?
                 empty_embed_df = pd.DataFrame(index=df[self.patient_id_col].unique(), columns=[f'emb_{j}' for j in range(self.hidden_dim)])
                 empty_embed_df.index.name = self.patient_id_col
                 return df, empty_embed_df
             else:
                 return df

        dataset = PatientSequenceDataset(sequences_list, patient_ids_list)
        # Use a large batch size for inference if memory allows
        eval_batch_size = self.batch_size * 2
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn_pad)

        # 2. Get Predictions and Calculate Errors
        self.model.eval()
        all_reconstruction_errors = [] # Store errors mapped back to original visit order
        all_patient_embeddings = []
        processed_indices = [] # Track original df indices
        original_indices_map = df.index.tolist() # Get original indices

        criterion = nn.L1Loss(reduction='none') # MAE for outlier score
        encoder = self.model.get_encoder() # Get encoder for embeddings

        df_idx_counter = 0 # Counter to map back to original df

        with torch.no_grad():
            batch_start_idx = 0
            for sequences, lengths, masks in data_loader:
                batch_size_current = sequences.shape[0]
                sequences = sequences.to(self.device)
                masks = masks.to(self.device)

                # Get reconstructions
                reconstructions = self.model(sequences, attention_mask=masks)

                # Calculate MAE per step
                mae_per_element = criterion(reconstructions, sequences) # (batch, seq_len, features)
                mae_per_step = torch.sum(mae_per_element, dim=2).cpu().numpy() # (batch, seq_len)

                # Get embeddings (e.g., final encoder hidden state)
                if return_embeddings:
                    encoder_outputs, encoder_final_hidden = encoder(sequences)
                    # If LSTM, take hidden state h
                    if isinstance(encoder_final_hidden, tuple):
                        final_state = encoder_final_hidden[0][-1] # Last layer hidden state (batch, hidden_dim)
                    else: # GRU
                        final_state = encoder_final_hidden[-1] # Last layer hidden state (batch, hidden_dim)
                    all_patient_embeddings.append(final_state.cpu().numpy())

                # Map errors back to individual visits, respecting padding/truncation
                numpy_masks = masks.cpu().numpy() # (batch, seq_len)
                original_lengths = lengths.cpu().numpy() # (batch,)

                for i in range(batch_size_current): # Iterate through samples in batch
                    seq_errors = mae_per_step[i] # Errors for this sequence (seq_len,)
                    seq_mask = numpy_masks[i] # Mask for this sequence (seq_len,)
                    orig_len = original_lengths[i] # Original length before padding

                    # Errors corresponding to actual visits (unpadded portion)
                    valid_errors = seq_errors[seq_mask] # Select errors where mask is True

                    if len(valid_errors) != orig_len:
                         self.logger.warning(f"Length mismatch in error mapping: expected {orig_len}, got {len(valid_errors)}. Using NaN.")
                         visit_errors_for_patient = [np.nan] * orig_len
                    else:
                         visit_errors_for_patient = valid_errors.tolist()

                    # Get the original DataFrame indices for this patient's visits
                    current_patient_id = patient_ids_list[batch_start_idx + i]
                    # Find original indices (assuming df is sorted by patient then encounter)
                    num_visits_in_df = len(df[df[self.patient_id_col] == current_patient_id])

                    # This direct mapping assumes df order matches sequence creation order
                    # Need a safer way if df wasn't pre-sorted exactly as sequences were made
                    patient_indices = df.index[df[self.patient_id_col] == current_patient_id].tolist()

                    if len(patient_indices) != orig_len:
                         self.logger.error(f"CRITICAL: Index count mismatch for patient {current_patient_id}. Expected {orig_len}, found {len(patient_indices)} in original df slice. Error mapping compromised.")
                         # Assign NaN to avoid incorrect mapping
                         visit_errors_for_patient = [np.nan] * len(patient_indices) # Match length of indices found
                    else:
                         # Truncation check (if max_seq_len was used in preparer)
                         if num_visits_in_df > orig_len: # Truncation happened
                             # We have errors for the LAST `orig_len` visits
                             patient_indices = patient_indices[-orig_len:] # Match indices to errors

                    # Store errors with corresponding original index
                    for visit_idx, error in zip(patient_indices, visit_errors_for_patient):
                         all_reconstruction_errors.append({'original_index': visit_idx, 'reconstruction_error': error})

                batch_start_idx += batch_size_current # Move to next batch start index in patient_ids_list


        # Create results DataFrame
        error_df = pd.DataFrame(all_reconstruction_errors).set_index('original_index')
        # Merge errors back into the original DataFrame
        outlier_df = df.join(error_df)

        # Check for NaNs introduced by mapping issues
        if outlier_df['reconstruction_error'].isnull().any():
            num_nans = outlier_df['reconstruction_error'].isnull().sum()
            self.logger.warning(f"{num_nans} visits have NaN reconstruction error due to potential mapping issues.")

        # Determine outlier flag
        outlier_df['is_outlier_visit'] = outlier_df['reconstruction_error'] > self.visit_error_threshold
        num_outliers = outlier_df['is_outlier_visit'].sum()
        self.logger.info(f"Outlier detection complete. Found {num_outliers} potential outlier visits ({num_outliers / len(outlier_df):.2%}).")

        if return_embeddings:
            if not all_patient_embeddings:
                 # Handle case where no embeddings were generated
                 embeddings_df = pd.DataFrame(index=patient_ids_list, columns=[f'emb_{j}' for j in range(self.hidden_dim)])
            else:
                 embeddings_array = np.concatenate(all_patient_embeddings, axis=0)
                 # Ensure index matches the unique patient IDs associated with the embeddings
                 unique_patient_ids_for_embeddings = patient_ids_list[:embeddings_array.shape[0]] # Use the list corresponding to processed sequences
                 embeddings_df = pd.DataFrame(embeddings_array, index=unique_patient_ids_for_embeddings)

            embeddings_df.index.name = self.patient_id_col
            embeddings_df.columns = [f'emb_{j}' for j in range(embeddings_array.shape[1])]
            self.logger.info(f"Generated embeddings DataFrame with shape {embeddings_df.shape}")
            return outlier_df, embeddings_df
        else:
            return outlier_df


    def save_model(self, path: str):
        """Saves the trained model's state dictionary."""
        if self.model:
            self.logger.info(f"Saving model state dictionary to {path}")
            # Include relevant attributes needed for reloading architecture
            save_obj = {
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'encoder_layers': self.encoder_layers,
                'decoder_layers': self.decoder_layers,
                'attention_dim': self.attention_dim,
                'dropout': self.dropout,
                'use_gru': self.use_gru,
                'visit_error_threshold': self.visit_error_threshold # Also save the threshold
            }
            torch.save(save_obj, path)
        else:
            self.logger.warning("No model to save.")

    def load_model(self, path: str):
        """Loads a model from a saved state dictionary."""
        if not os.path.exists(path):
            self.logger.error(f"Model checkpoint not found at {path}")
            return

        self.logger.info(f"Loading model state dictionary from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # Restore hyperparameters needed to build the model
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.encoder_layers = checkpoint.get('encoder_layers', 1) # Default if missing
        self.decoder_layers = checkpoint.get('decoder_layers', 1) # Default if missing
        self.attention_dim = checkpoint['attention_dim']
        self.dropout = checkpoint['dropout']
        self.use_gru = checkpoint['use_gru']
        self.visit_error_threshold = checkpoint.get('visit_error_threshold', None) # Load threshold

        # Build model architecture with loaded parameters
        self._build_model()

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device) # Ensure model is on correct device
        self.logger.info("Model loaded successfully.")