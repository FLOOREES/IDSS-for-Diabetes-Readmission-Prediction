import pandas as pd
import json
import numpy as np
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE, trustworthiness # Keep trustworthiness
# from sklearn.manifold import continuity # If you want continuity back, uncomment
from typing import Dict, List, Optional, Union
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

class SecondPhasePreprocessor:
    """
    Performs the second phase of preprocessing: embedding encoding for diagnosis and other categorical variables.

    This class handles label encoding for diagnosis codes and other specified categorical features,
    and generates/loads SpaCy embeddings for diagnosis codes. It reuses pre-computed resources
    granularly: diagnosis embeddings/encoder and other label encoders are checked and processed
    independently to enhance efficiency and maintain consistency.

    Key improvements:
        - Independent checking and processing for diagnosis resources (embeddings, label encoder)
          and other categorical label encoders.
        - Avoids recomputing embeddings if only other label encoders are missing.
        - Ensures SpaCy model is loaded only when strictly necessary.
    """

    def __init__(
        self,
        diag_embeddings_path: str,
        diag_label_encoder_path: str,
        label_encoders_path: str,
        icd9_hierarchy_path: str,
        icd9_chapters_path: str, # Keep for potential future use, even if not used in embedding now
        spacy_model_name: str,
        label_encode_columns: List[str] = None,
        embedding_dim: int = 8,
        tsne_n_components: int = 8,
        tsne_method: str = 'exact',
        embedding_weight_specific: float = 0.7,
        embedding_weight_general: float = 0.3,
        logger: logging.Logger = None
    ):
        # ... (Initialization parameters remain the same) ...
        self.diag_embeddings_path = diag_embeddings_path
        self.diag_label_encoder_path = diag_label_encoder_path
        self.label_encoders_path = label_encoders_path
        self.icd9_hierarchy_path = icd9_hierarchy_path
        self.icd9_chapters_path = icd9_chapters_path
        self.spacy_model_name = spacy_model_name
        self.label_encode_columns = label_encode_columns or []
        self.embedding_dim = embedding_dim
        self.tsne_n_components = tsne_n_components
        self.tsne_method = tsne_method
        self.embedding_weight_specific = embedding_weight_specific
        self.embedding_weight_general = embedding_weight_general
        self.logger = logger or logging.getLogger(__name__)

        # Internal state
        self.label_encoder = LabelEncoder() # For diagnosis codes
        self.other_label_encoders = {} # For other specified categorical columns
        self.diag_embeddings = None # To store loaded/calculated embeddings
        self._nlp = None # Lazy load SpaCy model

        self.logger.info(f"SecondPhasePreprocessor initialized. Diag Emb: '{os.path.basename(self.diag_embeddings_path)}', "
                         f"Diag LE: '{os.path.basename(self.diag_label_encoder_path)}', Other LE: '{os.path.basename(self.label_encoders_path)}'. "
                         f"Additional LE cols: {self.label_encode_columns or 'None'}.")


    # --- Resource Checking ---
    def _check_diag_resources(self) -> bool:
        """Checks if diagnosis embedding and label encoder files exist."""
        exists = os.path.exists(self.diag_embeddings_path) and os.path.exists(self.diag_label_encoder_path)
        self.logger.info(f"Checking diagnosis resources (Embeddings, Label Encoder): {'Found' if exists else 'Missing'}")
        return exists

    def _check_other_le_resources(self) -> bool:
        """Checks if the file for other label encoders exists."""
        exists = os.path.exists(self.label_encoders_path)
        self.logger.info(f"Checking other label encoder resources: {'Found' if exists else 'Missing'}")
        return exists

    # --- Diagnosis Processing Methods ---
    def _load_and_apply_diag_resources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loads existing diagnosis embeddings and label encoder, applies encoding,
        AND VALIDATES loaded label encoder and diagnosis data."""
        try:
            self.logger.info(f"Loading diagnosis label encoder mapping from: {self.diag_label_encoder_path}")
            with open(self.diag_label_encoder_path, 'r') as f:
                code_to_label = json.load(f)
            # Ensure keys are strings if loaded from JSON
            code_to_label = {str(k): v for k, v in code_to_label.items()}
            loaded_classes = list(code_to_label.keys())
            self.label_encoder.classes_ = np.array(loaded_classes) # Set classes directly

            # --- [NEW] Validation: Check for "0" category in loaded label encoder ---
            if '0' not in code_to_label:
                self.logger.error("CRITICAL: '0' category MISSING from LOADED diagnosis label encoder mapping! "
                                  "The loaded encoder is invalid. Check saved resources.")
                raise ValueError("'0' category missing in loaded diagnosis label encoder - invalid resource.")
            else:
                self.logger.info("'0' category FOUND in LOADED diagnosis label encoder mapping - validation passed.")
            # --- [END] Validation: "0" category check ---


            self.logger.info("Applying loaded diagnosis label encoding.")
            # Ensure diagnosis columns are strings for mapping, handle NAs
            for col in ['diag_1', 'diag_2', 'diag_3']:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('0') # Convert potential numbers/NAs to string '0'
                    # --- [NEW] Validation: Check for unmapped values BEFORE mapping ---
                    unmapped_codes = df[col][~df[col].isin(code_to_label.keys())].unique()
                    if len(unmapped_codes) > 0:
                        self.logger.error(f"CRITICAL: Unmapped diagnosis codes found BEFORE applying loaded label encoder in column '{col}': {list(unmapped_codes)}. "
                                          f"Loaded label encoder is incompatible with current data. Check preprocessing steps.")
                        raise ValueError(f"Unmapped diagnosis codes found before applying loaded label encoder in column '{col}'. Incompatible data/encoder.")
                    # --- [END] Validation: Unmapped values check ---

                    df[col] = df[col].map(code_to_label).fillna(-1).astype(int) # Apply mapping, fill remaining unknown with -1
                else:
                     self.logger.warning(f"Diagnosis column '{col}' not found in DataFrame.")


            self.logger.info(f"Loading diagnosis embeddings from: {self.diag_embeddings_path}")
            self.diag_embeddings = np.load(self.diag_embeddings_path)
            self.logger.info(f"Diagnosis resources loaded successfully. Embeddings shape: {self.diag_embeddings.shape}")
            return df

        except FileNotFoundError as e:
            self.logger.error(f"Diagnosis resource file not found during load: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from diagnosis label encoder file: {e}")
            raise
        except ValueError as e:
             self.logger.error(f"ValueError loading diagnosis resources (corrupted file or data incompatibility?): {e}")
             raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading diagnosis resources: {e}")
            raise

    def _process_diagnosis_resources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates, saves, and applies diagnosis label encoding and embeddings."""
        self.logger.info("Calculating diagnosis label encoder and embeddings.")
        _nlp_managed_locally = False # Flag to track if SpaCy was loaded here
        try:
            # 1. Label Encode Diagnosis Columns
            self.logger.info("Fitting diagnosis label encoder.")
            diag_cols = ['diag_1', 'diag_2', 'diag_3']
            present_diag_cols = [col for col in diag_cols if col in df.columns]
            if not present_diag_cols:
                 self.logger.warning("No diagnosis columns found. Skipping diagnosis processing.")
                 return df # Return df as is if no diag columns

            # Handle NaNs and ensure string type before finding unique codes
            unique_codes_list = []
            for col in present_diag_cols:
                 df[col] = df[col].astype(str).fillna('0') # Convert NaNs and ensure string
                 unique_codes_list.append(df[col])

            unique_codes = pd.concat(unique_codes_list).unique()
            self.label_encoder.fit(unique_codes)
            code_to_label = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
            self.logger.info(f"Found {len(unique_codes)} unique diagnosis codes.")

            # --- [NEW] Validation: Check for "0" category in label encoder ---
            if '0' not in code_to_label:
                self.logger.error("CRITICAL: '0' category MISSING from diagnosis label encoder mapping! Check data preprocessing.")
                raise ValueError("'0' category missing in diagnosis label encoder.")
            else:
                self.logger.info("'0' category FOUND in diagnosis label encoder mapping as expected.")
            # --- [END] Validation ---

            self.logger.info("Applying diagnosis label encoding.")
            for col in present_diag_cols:
                df[col] = df[col].map(code_to_label).fillna(-1).astype(int) # Apply mapping

            # Save diagnosis label encoder mapping
            code_to_label_int_values = {str(k): int(v) for k, v in code_to_label.items()} # Ensure string keys for JSON
            self.logger.info(f"Saving diagnosis label encoder mapping to: {self.diag_label_encoder_path}")
            os.makedirs(os.path.dirname(self.diag_label_encoder_path), exist_ok=True) # Ensure directory exists
            with open(self.diag_label_encoder_path, 'w') as f:
                json.dump(code_to_label_int_values, f, indent=4)
            self.logger.info("Diagnosis label encoder mapping saved.")

            # 2. Generate Embeddings
            self.logger.info(f"Loading SpaCy model: {self.spacy_model_name} for embedding generation.")
            if self._nlp is None:
                # Only load SpaCy if we need to generate embeddings
                self._nlp = spacy.load(self.spacy_model_name)
                _nlp_managed_locally = True # Mark that we loaded it here
            else:
                self.logger.warning("SpaCy model was already loaded unexpectedly.") # Should usually be None

            idc9_hierarchy = self._load_icd9_hierarchy_data(self.icd9_hierarchy_path)

            label_to_embed = dict()
            self.logger.info(f"Generating SpaCy embeddings for {len(unique_codes)} unique diagnosis codes.")
            for code in unique_codes: # Use the unique codes found earlier
                if code in code_to_label: # Ensure code is in our mapping
                    label = code_to_label[code]
                    desc = self._find_diag_description(idc9_hierarchy, code) # Static method call
                    label_to_embed[label] = self._create_embedding(self._nlp, desc, self.embedding_weight_specific, self.embedding_weight_general) # Static method call
                else:
                    self.logger.warning(f"Code '{code}' not found in label encoder mapping during embedding generation.")

            # Ensure embeddings are ordered by label index for correct alignment
            num_labels = len(self.label_encoder.classes_)
            embedding_size = self._nlp.vector_size # Get embedding size from SpaCy model
            unique_emb_array = np.zeros((num_labels, embedding_size), dtype=np.float32) # Initialize with zeros

            for label, embedding in label_to_embed.items():
                 if 0 <= label < num_labels: # Check if label index is valid
                      unique_emb_array[label] = embedding
                 else:
                      self.logger.warning(f"Invalid label index '{label}' encountered during embedding array creation.")


            self.logger.info(f"SpaCy embeddings generated, shape: {unique_emb_array.shape}.")

            # 3. Dimensionality Reduction (t-SNE)
            self.logger.info(f"Applying t-SNE for dimensionality reduction to {self.embedding_dim} dimensions.")
            tsne = TSNE(n_components=self.embedding_dim, method=self.tsne_method, random_state=42) # Add random_state for reproducibility
            red_emb = tsne.fit_transform(unique_emb_array)
            self.diag_embeddings = red_emb # Store reduced embeddings
            self.logger.info(f"t-SNE completed, reduced embeddings shape: {self.diag_embeddings.shape}.")

            # Optional: Validation
            try:
                 self.logger.info("Calculating Trustworthiness for t-SNE embeddings...")
                 # Ensure n_neighbors is less than number of samples if dataset is very small
                 n_neighbors_val = min(5, unique_emb_array.shape[0] - 1)
                 if n_neighbors_val > 0:
                     trustworthiness_score = trustworthiness(unique_emb_array, red_emb, n_neighbors=n_neighbors_val)
                     self.logger.info(f"  Trustworthiness (n_neighbors={n_neighbors_val}): {trustworthiness_score:.4f}")
                 else:
                      self.logger.warning("Not enough samples to calculate trustworthiness.")
            except Exception as val_err:
                 self.logger.warning(f"Could not calculate trustworthiness: {val_err}")


            # 4. Save Embeddings
            self.logger.info(f"Saving diagnosis embeddings to: {self.diag_embeddings_path}")
            os.makedirs(os.path.dirname(self.diag_embeddings_path), exist_ok=True) # Ensure directory exists
            np.save(self.diag_embeddings_path, self.diag_embeddings)
            self.logger.info("Diagnosis embeddings saved.")

            return df

        except FileNotFoundError as e:
            self.logger.error(f"Data file not found during diagnosis processing: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error calculating diagnosis resources: {e}")
            raise
        finally:
            # Unload SpaCy only if it was loaded in this method
            if _nlp_managed_locally and self._nlp is not None:
                self.logger.info("Unloading SpaCy model.")
                self._nlp = None


    # --- Other Label Encoding Methods ---
    def _load_and_apply_other_le_resources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loads existing mappings for other columns and applies encoding."""
        if not self.label_encode_columns:
            self.logger.info("No additional columns specified for label encoding. Skipping load.")
            return df
        try:
            self.logger.info(f"Loading label encoder mappings for additional columns from: {self.label_encoders_path}")
            with open(self.label_encoders_path, 'r') as f:
                other_mappings_json = json.load(f)

            self.logger.info("Applying loaded other label encoding mappings.")
            for column in self.label_encode_columns:
                 if column in df.columns:
                    if column in other_mappings_json:
                        # Load the mapping directly
                        mapping = other_mappings_json[column]
                        # Ensure keys are strings for compatibility with .map() on stringified column
                        mapping = {str(k): int(v) for k, v in mapping.items()}

                        # Apply mapping
                        df[column] = df[column].astype(str).fillna('NaN_placeholder') # Handle NaN before map
                        df[column] = df[column].map(mapping).fillna(-1).astype(int) # Fill unmapped with -1
                        self.logger.debug(f"Mapping loaded and applied for column: {column}")
                    else:
                        self.logger.warning(f"No mapping found in file for column: {column}. Column remains unchanged.")
                 else:
                     self.logger.warning(f"Column '{column}' for label encoding not found in DataFrame.")

            self.logger.info("Other label encoder mappings loaded and applied.")
            return df

        except FileNotFoundError as e:
            self.logger.error(f"Other label encoder file not found during load: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from other label encoder file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading other label encoders: {e}")
            raise


    def _process_other_label_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits LEncoder, saves (mapping only), and applies encoding for other specified columns."""
        if not self.label_encode_columns:
            self.logger.info("No additional columns specified for label encoding. Skipping calculation.")
            return df

        self.logger.info("Calculating label encoder mappings for other specified columns.")
        mappings_to_save = {}

        for column in self.label_encode_columns:
            if column in df.columns:
                self.logger.info(f"Fitting LabelEncoder and applying mapping for column: {column}")
                df[column] = df[column].fillna('NaN_placeholder') # Handle NaN

                temp_le = LabelEncoder()
                # Fit and transform in one step
                df[column] = temp_le.fit_transform(df[column].astype(str)).astype(int)

                # Get the mapping
                mapping = dict(zip(temp_le.classes_, temp_le.transform(temp_le.classes_)))
                # Ensure keys are str, values are int for JSON
                mappings_to_save[column] = {str(k): int(v) for k, v in mapping.items()}
                self.logger.debug(f"Label encoding applied and mapping prepared for column: {column}")
            else:
                self.logger.warning(f"Column '{column}' for label encoding not found in DataFrame. Skipping.")


        # Save the mappings
        if mappings_to_save:
             self.logger.info(f"Saving label encoder mappings for additional columns to: {self.label_encoders_path}")
             os.makedirs(os.path.dirname(self.label_encoders_path), exist_ok=True)
             with open(self.label_encoders_path, 'w') as f:
                json.dump(mappings_to_save, f, indent=4)
             self.logger.info("Label encoder mappings for additional columns saved.")
        else:
             self.logger.info("No additional label encoder mappings were calculated or saved.")

        return df


    # --- Main Transform Method ---
    def transform(self, df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Applies all second phase preprocessing steps, checking/using existing
        resources granularly. Optionally saves the transformed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame after first phase preprocessing.
        output_path : Optional[str], default=None
            If provided, the path (including filename) where the fully transformed
            DataFrame will be saved as a CSV file.

        Returns
        -------
        pd.DataFrame
            The DataFrame with diagnosis and other specified columns label encoded.
            Diagnosis embeddings are available in `self.diag_embeddings`.
        """
        self.logger.info("Starting comprehensive preprocessing transformation (Phase 2)...")
        df_processed = df.copy() # Work on a copy

        # --- Step 1: Process Diagnosis Resources ---
        if self._check_diag_resources():
            df_processed = self._load_and_apply_diag_resources(df_processed)
        else:
            df_processed = self._process_diagnosis_resources(df_processed)

        # --- Step 2: Process Other Label Encoders ---
        if self.label_encode_columns:
            if self._check_other_le_resources():
                df_processed = self._load_and_apply_other_le_resources(df_processed)
            else:
                df_processed = self._process_other_label_encoders(df_processed)
        else:
            self.logger.info("No additional columns specified for label encoding.")


        # --- Step 3: Optional Saving ---
        # Note: While convenient, saving arbitrary files can be seen as mixing
        # responsibilities. Often, saving outputs is handled by the calling script/pipeline.
        if output_path:
            self.logger.info(f"Saving transformed DataFrame to: {output_path}")
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
                df_processed.to_csv(output_path, index=False)
                self.logger.info("DataFrame saved successfully.")
            except Exception as e:
                self.logger.error(f"Failed to save DataFrame to {output_path}: {e}")
                # Continue execution even if saving fails


        # --- Finalization ---
        self.logger.info("Comprehensive preprocessing transformation (Phase 2) completed.")
        # Ensure embeddings are available if needed (check added previously)
        if self.diag_embeddings is None and self._check_diag_resources():
            self.logger.warning("Diagnosis embeddings were expected but are not loaded. Attempting load again.")
            try:
                self.diag_embeddings = np.load(self.diag_embeddings_path)
                self.logger.info("Diagnosis embeddings loaded successfully on second attempt.")
            except Exception as e:
                 self.logger.error(f"Failed to load diagnosis embeddings on second attempt: {e}")

        return df_processed

    # --- Static Helper Methods ---
    @staticmethod
    @lru_cache(maxsize=None) # Unbounded cache for descriptions
    def _find_diag_description(idc9_hierarchy: List[Dict], code: str) -> Union[Dict, str]:
        # ... (implementation remains the same) ...
        if code == '0' or pd.isna(code):
            return 'No diagnosis' # Simplified check

        code_str = str(code)
        # Clean code (remove dots, potentially add leading zeros if needed based on hierarchy format)
        # This part depends heavily on how codes are stored in your hierarchy JSON
        cleaned_code = code_str.replace('.', '')
        is_three_digit = '.' not in code_str and len(cleaned_code) <= 3

        # Add leading zeros if necessary for matching common formats (e.g., '1' -> '001')
        if is_three_digit and len(cleaned_code) < 3:
            cleaned_code = cleaned_code.zfill(3)

        for d in idc9_hierarchy:
            # Check against different possible keys and formats in the hierarchy data
            if ('icd9' in d and (d['icd9'] == cleaned_code or d['icd9'] == cleaned_code + '0')) or \
               ('threedigit' in d and is_three_digit and d['threedigit'] == cleaned_code):

                specific_desc = d.get('descLong', d.get('major', 'Unknown Specific Description'))
                general_desc_parts = []
                if 'subchapter' in d: general_desc_parts.append(d['subchapter'])
                if 'chapter' in d: general_desc_parts.append(d['chapter'])
                general_desc = '. '.join(filter(None, general_desc_parts)) or 'Unknown General Description'

                return {'specific': specific_desc, 'general': general_desc}

        logger.warning(f"Diagnosis code '{code}' (cleaned: '{cleaned_code}') not found in ICD-9 hierarchy.")
        return 'Not found' # Return consistent 'Not found' string

    @staticmethod
    def _create_embedding(nlp, desc: Union[Dict, str], weight_specific: float, weight_general: float) -> np.ndarray:
        # ... (implementation remains the same) ...
        try:
            if isinstance(desc, dict):
                emb_specific = nlp(desc.get('specific', '')).vector
                emb_general = nlp(desc.get('general', '')).vector
                # Handle potential zero vectors if descriptions were empty/missing
                norm_specific = np.linalg.norm(emb_specific)
                norm_general = np.linalg.norm(emb_general)

                if norm_specific > 0 and norm_general > 0:
                     emb = weight_specific * emb_specific + weight_general * emb_general
                elif norm_specific > 0:
                     emb = emb_specific # Use only specific if general is zero
                elif norm_general > 0:
                     emb = emb_general # Use only general if specific is zero
                else:
                     emb = np.zeros(nlp.vector_size, dtype=np.float32) # Return zero vector if both are zero

                return emb
            else:
                # Handle 'No diagnosis' or 'Not found' - return zero vector or embedding of the string?
                # Let's embed the string itself.
                return nlp(str(desc)).vector # Ensure input is string
        except Exception as e:
            logger.error(f"Error creating embedding for description '{desc}': {e}")
            # Return a zero vector of the correct size in case of error
            return np.zeros(nlp.vector_size, dtype=np.float32)


    @staticmethod
    def _load_icd9_hierarchy_data(path: str) -> List[Dict]:
        # ... (implementation remains the same) ...
        logger.info(f"Loading ICD-9 hierarchy data from: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"ICD-9 hierarchy file not found at: {path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from ICD-9 hierarchy file at: {path}")
            raise

    @staticmethod
    def _load_icd9_chapters_data(path: str) -> Dict:
        # ... (implementation remains the same) ...
        logger.info(f"Loading ICD-9 chapters data from: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"ICD-9 chapters file not found at: {path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from ICD-9 chapters file at: {path}")
            raise


# --- Main execution block remains the same for demonstration ---
if __name__ == "__main__":
    # ... (Keep the existing __main__ block for testing) ...
    # Example: Ensure paths in config.py are correct
    from config import (
        RAW_DATA_PATH, MID_PROCESSING_PATH, NO_MISSINGS_PATH, NO_MISSINGS_COD_PATH,
        DIAG_EMBEDDINGS_PATH, DIAG_LABEL_ENCODER_PATH, LABEL_ENCODERS_PATH,
        ICD9_HIERARCHY_PATH, ICD9_CHAPTERS_PATH, SPACY_MODEL_NAME, MISSING_VALUES
    )
    from preprocessing import FirstPhasePreprocessor

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/preprocessing_phase2_demo.log", mode='w') # Overwrite log for demo
        ]
    )
    main_script_logger = logging.getLogger(__name__)
    main_script_logger.info("=== DEMO SCRIPT - SECOND PHASE PREPROCESSING STARTED ===")

    # --- Simulate Missing Files ---
    # To test, you can temporarily rename or delete one of the files, e.g.:
    # if os.path.exists(LABEL_ENCODERS_PATH):
    #     os.rename(LABEL_ENCODERS_PATH, LABEL_ENCODERS_PATH + ".bak")
    #     main_script_logger.info(f"Temporarily renamed {LABEL_ENCODERS_PATH} to test calculation.")
    # -----------------------------

    try:
        # Load data (assuming NO_MISSINGS_PATH exists from a previous run or script)
        if not os.path.exists(NO_MISSINGS_PATH):
             main_script_logger.error(f"Input data file {NO_MISSINGS_PATH} not found. Run Phase 1 first.")
             exit()
        df_input_phase2 = pd.read_csv(NO_MISSINGS_PATH) # Simpler load if CSV is clean
        main_script_logger.info(f"Data loaded for Phase 2, shape: {df_input_phase2.shape}")

        LABEL_ENCODE_COLUMNS_EXAMPLE = ['discharge_disposition_id', 'admission_source_id'] # From config or define here

        phase2_preprocessor = SecondPhasePreprocessor(
            diag_embeddings_path=DIAG_EMBEDDINGS_PATH,
            diag_label_encoder_path=DIAG_LABEL_ENCODER_PATH,
            label_encoders_path=LABEL_ENCODERS_PATH,
            icd9_hierarchy_path=ICD9_HIERARCHY_PATH,
            icd9_chapters_path=ICD9_CHAPTERS_PATH,
            spacy_model_name=SPACY_MODEL_NAME,
            label_encode_columns=LABEL_ENCODE_COLUMNS_EXAMPLE,
            embedding_dim=16, tsne_n_components=16, tsne_method='exact',
            embedding_weight_specific=0.8, embedding_weight_general=0.2,
            logger=main_script_logger
        )

        df_encoded_phase2 = phase2_preprocessor.transform(df_input_phase2.copy())

        main_script_logger.info(f"Data shape after Phase 2 preprocessing: {df_encoded_phase2.shape}")
        main_script_logger.info(f"Columns after Phase 2: {df_encoded_phase2.columns.tolist()}")
        if phase2_preprocessor.diag_embeddings is not None:
             main_script_logger.info(f"Diagnosis embeddings are available in memory, shape: {phase2_preprocessor.diag_embeddings.shape}")
        else:
             main_script_logger.warning("Diagnosis embeddings are not loaded in memory after transform.")


        # Optional: Save processed data
        # df_encoded_phase2.to_csv(NO_MISSINGS_COD_PATH, index=False)
        # main_script_logger.info(f"Processed data potentially saved to: {NO_MISSINGS_COD_PATH}")

    except Exception as e:
         main_script_logger.error(f"An error occurred during the demo: {e}", exc_info=True)
    finally:
        # --- Restore Missing Files (if renamed) ---
        # if os.path.exists(LABEL_ENCODERS_PATH + ".bak"):
        #     os.rename(LABEL_ENCODERS_PATH + ".bak", LABEL_ENCODERS_PATH)
        #     main_script_logger.info(f"Restored {LABEL_ENCODERS_PATH}.")
        # ---------------------------------------
        main_script_logger.info("=== DEMO SCRIPT - SECOND PHASE PREPROCESSING COMPLETED ===")