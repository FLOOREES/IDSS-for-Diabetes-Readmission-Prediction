import pandas as pd
import json
import numpy as np
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE, trustworthiness, continuity
from typing import Dict, List, Optional, Union
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

class SecondPhasePreprocessor:
    """
    Performs the second phase of preprocessing: embedding encoding for diagnosis and other categorical variables.

    This class extends the functionality of the SecondPhasePreprocessor to handle label encoding
    for additional categorical features beyond diagnosis codes. It efficiently manages label encoding
    for both diagnosis and specified categorical variables, and generates or loads SpaCy embeddings
    for diagnosis codes. The class reuses pre-computed resources when available to enhance
    performance and maintain consistency across runs.

    The preprocessing steps include:
        - Checking for pre-existing diagnosis embeddings, diagnosis label encoder mapping,
          and label encoders for other specified categorical variables.
        - If resources exist, load them for efficient label encoding and embedding application.
        - If resources are absent, compute diagnosis embeddings using SpaCy, reduce dimensionality with t-SNE,
          perform label encoding for diagnoses and other categorical variables, and persist all generated
          embeddings and label encoder mappings for future reuse.

    Configurable aspects include:
        - Embedding dimensions for reduced diagnosis embeddings.
        - t-SNE parameters for dimensionality reduction.
        - Weights for combining specific and general diagnosis descriptions in embedding creation.
        - Specification of additional categorical columns to be label encoded.
    """

    def __init__(
        self,
        diag_embeddings_path: str,
        diag_label_encoder_path: str,
        label_encoders_path: str, # New parameter: Path for other label encoders
        icd9_hierarchy_path: str,
        icd9_chapters_path: str,
        spacy_model_name: str,
        label_encode_columns: List[str] = None, # New parameter: Columns to label encode
        embedding_dim: int = 8,
        tsne_n_components: int = 8,
        tsne_method: str = 'exact',
        embedding_weight_specific: float = 0.7,
        embedding_weight_general: float = 0.3,
        logger: logging.Logger = None
    ):
        """
        Initializes the SecondPhasePreprocessor with configurations for file paths, SpaCy model,
        embedding dimensions, t-SNE, logging, and additional label encoding.

        Parameters
        ----------
        diag_embeddings_path : str
            File path to save/load diagnosis embeddings (.npy file).
        diag_label_encoder_path : str
            File path to save/load diagnosis label encoder mapping (JSON file).
        label_encoders_path : str # New parameter documentation
            File path to save/load label encoders for additional categorical columns (JSON file).
        icd9_hierarchy_path : str
            File path to the ICD-9 hierarchy JSON file, used for diagnosis description lookup.
        icd9_chapters_path : str
            File path to the ICD-9 chapters JSON file (currently not utilized in embedding calculation).
        spacy_model_name : str
            Name of the SpaCy model to be loaded for generating initial diagnosis embeddings (e.g., "en_core_sci_md").
        label_encode_columns : List[str], optional # New parameter documentation
            List of column names to be label encoded in addition to diagnosis codes. Defaults to None.
        embedding_dim : int, default=8
            Desired dimensionality of the diagnosis embeddings after t-SNE reduction.
        tsne_n_components : int, default=8
            Number of components for t-SNE dimensionality reduction.
        tsne_method : str, default='exact'
            Method parameter for t-SNE, either 'exact' or 'barnes_hut'.
        embedding_weight_specific : float, default=0.7
            Weight assigned to the 'specific' diagnosis description when creating combined embeddings.
        embedding_weight_general : float, default=0.3
            Weight assigned to the 'general' diagnosis description when creating combined embeddings.
        logger : logging.Logger, optional
            Logger instance for logging messages. If None, a default module-level logger is used.
        """
        self.diag_embeddings_path = diag_embeddings_path
        self.diag_label_encoder_path = diag_label_encoder_path
        self.label_encoders_path = label_encoders_path # Store path for other label encoders
        self.icd9_hierarchy_path = icd9_hierarchy_path
        self.icd9_chapters_path = icd9_chapters_path
        self.spacy_model_name = spacy_model_name
        self.label_encode_columns = label_encode_columns or [] # Store label encode columns, default to empty list
        self.embedding_dim = embedding_dim
        self.tsne_n_components = tsne_n_components
        self.tsne_method = tsne_method
        self.embedding_weight_specific = embedding_weight_specific
        self.embedding_weight_general = embedding_weight_general
        self.logger = logger or logging.getLogger(__name__)
        self.label_encoder = LabelEncoder() # LabelEncoder for diagnosis codes
        self.other_label_encoders = {} # Dictionary to store LabelEncoders for other columns
        self._nlp = None # Lazy load SpaCy model

        self.logger.info(f"SecondPhasePreprocessor initialized with configurations. Additional label encoding columns: {self.label_encode_columns or 'None'}.")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies diagnosis embedding encoding and label encoding for other categorical columns
        to the input DataFrame.

        Orchestrates the preprocessing pipeline, checking for and loading existing resources
        or calculating them if necessary. This includes label encoding for diagnosis codes and
        other specified categorical columns, as well as generating diagnosis embeddings.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame expected to contain diagnosis columns ('diag_1', 'diag_2', 'diag_3')
            and any columns specified for additional label encoding in `label_encode_columns`.

        Returns
        -------
        pd.DataFrame
            DataFrame with diagnosis columns and specified categorical columns label encoded.
            Diagnosis embeddings are calculated and stored as `self.diag_embeddings`.
        """
        self.logger.info("Starting comprehensive preprocessing transformation: Diagnosis embedding and additional label encoding.")

        if self._check_existing_resources():
            self.logger.info("Existing resources found: Diagnosis embeddings, diagnosis label encoder, and additional label encoders. Loading...")
            df_encoded = self._load_existing_embeddings(df.copy()) # Load all existing resources
        else:
            self.logger.info("No existing resources found or incomplete resources. Calculating all embeddings and label encoders.")
            df_encoded = self._calculate_embeddings(df.copy()) # Calculate all resources

        self.logger.info("Comprehensive preprocessing transformation completed.")
        return df_encoded


    def _check_existing_resources(self) -> bool:
        """
        Verifies the existence of all necessary resources: diagnosis embeddings, diagnosis label encoder,
        and label encoders for other specified categorical columns.

        Returns
        -------
        bool
            True if all resource files exist, indicating that all pre-calculated resources are available.
            False otherwise.
        """
        embeddings_exist = os.path.exists(self.diag_embeddings_path)
        diag_label_encoder_exist = os.path.exists(self.diag_label_encoder_path)
        other_label_encoders_exist = os.path.exists(self.label_encoders_path) # Check for other label encoders

        resources_exist = embeddings_exist and diag_label_encoder_exist and other_label_encoders_exist
        resource_status = "found" if resources_exist else "not found"
        self.logger.info(f"Checking for existing resources: Diagnosis Embeddings - {os.path.basename(self.diag_embeddings_path)}: {'exists' if embeddings_exist else 'missing'}, "
                         f"Diagnosis Label Encoder - {os.path.basename(self.diag_label_encoder_path)}: {'exists' if diag_label_encoder_exist else 'missing'}, "
                         f"Additional Label Encoders - {os.path.basename(self.label_encoders_path)}: {'exists' if other_label_encoders_exist else 'missing'}. Resources {resource_status}.")
        return resources_exist


    def _load_existing_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads pre-calculated diagnosis embeddings, diagnosis label encoder mapping,
        and label encoders for other categorical columns from files. Applies label encoding
        to diagnosis columns and other specified categorical columns in the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to be encoded.

        Returns
        -------
        pd.DataFrame
            DataFrame with diagnosis columns and other specified categorical columns label encoded
            using the loaded label encoders and mappings.
        """
        try:
            # Load Diagnosis Label Encoder and Embeddings (as before)
            self.logger.info(f"Loading diagnosis label encoder mapping from: {self.diag_label_encoder_path}")
            with open(self.diag_label_encoder_path, 'r') as f:
                code_to_label = json.load(f)
            loaded_classes = list(code_to_label.keys())
            self.label_encoder.classes_ = np.array(loaded_classes)

            df['diag_1'] = df['diag_1'].apply(lambda x: '0' if pd.isna(x) else x)
            df['diag_2'] = df['diag_2'].apply(lambda x: '0' if pd.isna(x) else x)
            df['diag_3'] = df['diag_3'].apply(lambda x: '0' if pd.isna(x) else x)

            df['diag_1'] = df['diag_1'].map(code_to_label).fillna(-1).astype(int)
            df['diag_2'] = df['diag_2'].map(code_to_label).fillna(-1).astype(int)
            df['diag_3'] = df['diag_3'].map(code_to_label).fillna(-1).astype(int)

            self.diag_embeddings = np.load(self.diag_embeddings_path)
            self.logger.info(f"Diagnosis label encoder and embeddings loaded.")

            # Load and Apply Other Label Encoders (NEW SECTION)
            self.logger.info(f"Loading label encoders for additional columns from: {self.label_encoders_path}")
            with open(self.label_encoders_path, 'r') as f:
                label_encoders_json = json.load(f)

            for column in self.label_encode_columns:
                if column in label_encoders_json:
                    self.other_label_encoders[column] = LabelEncoder() # Initialize LabelEncoder
                    loaded_classes_other = label_encoders_json[column]['classes_'] # Get loaded classes for column
                    mapping = label_encoders_json[column]['mapping'] # Get loaded mapping
                    self.other_label_encoders[column].classes_ = np.array(loaded_classes_other) # Set classes_
                    df[column] = df[column].map(mapping).fillna(-1).astype(int) # Apply mapping
                    self.logger.info(f"Label encoder loaded and applied for column: {column}")
                else:
                    self.logger.warning(f"No label encoder mapping found in file for column: {column}. Skipping label encoding for this column.")
            self.logger.info("Label encoders for additional columns loaded and applied.")

            return df

        except FileNotFoundError as e:
            self.logger.error(f"Resource file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from label encoder file: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"ValueError encountered while loading embeddings (possibly corrupted file): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading existing embeddings and label encoders: {e}")
            raise


    def _calculate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates diagnosis embeddings, performs label encoding for diagnosis and other categorical columns,
        and saves all generated resources to files.

        This method extends `_calculate_embeddings` to also handle label encoding for additional categorical
        columns specified in `self.label_encode_columns`. It fits LabelEncoders for these columns,
        transforms the DataFrame, and saves the fitted encoders along with the diagnosis embeddings and encoder.
        """
        try:
            # Calculate Diagnosis Embeddings and Label Encoding (as before)
            self.logger.info("Calculating diagnosis embeddings and label encoding.")
            idc9_hierarchy = SecondPhasePreprocessor._load_icd9_hierarchy_data(self.icd9_hierarchy_path)
            # idc9_chapters = SecondPhasePreprocessor._load_icd9_chapters_data(self.icd9_chapters_path)

            df['diag_1'] = df['diag_1'].apply(lambda x: '0' if pd.isna(x) else x)
            df['diag_2'] = df['diag_2'].apply(lambda x: '0' if pd.isna(x) else x)
            df['diag_3'] = df['diag_3'].apply(lambda x: '0' if pd.isna(x) else x)

            unique_codes = pd.concat([df['diag_1'], df['diag_2'], df['diag_3']]).unique()
            self.label_encoder.fit(unique_codes)
            code_to_label = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))

            df['diag_1'] = self.label_encoder.transform(df['diag_1'])
            df['diag_2'] = self.label_encoder.transform(df['diag_2'])
            df['diag_3'] = self.label_encoder.transform(df['diag_3'])
            self.logger.info("Diagnosis columns label encoded.")

            code_to_label_int_values = {k: int(v) for k, v in code_to_label.items()}
            self.logger.info(f"Saving diagnosis label encoder mapping to: {self.diag_label_encoder_path}")
            with open(self.diag_label_encoder_path, 'w') as f:
                json.dump(code_to_label_int_values, f)
            self.logger.info("Diagnosis label encoder mapping saved.")

            self.logger.info(f"Loading SpaCy model: {self.spacy_model_name} for embedding generation.")
            if self._nlp is None:
                self._nlp = spacy.load(self.spacy_model_name)
            label_to_embed = dict()
            self.logger.info(f"Generating SpaCy embeddings for {len(unique_codes)} unique diagnosis codes.")
            for code in unique_codes:
                desc = SecondPhasePreprocessor._find_diag_description(idc9_hierarchy, code)
                label_to_embed[code_to_label[code]] = SecondPhasePreprocessor._create_embedding(self._nlp, desc, weight_specific=self.embedding_weight_specific, weight_general=self.embedding_weight_general)

            unique_emb = np.empty((0, self._nlp.vector_size))
            for k in sorted(label_to_embed.keys()):
                unique_emb = np.vstack([unique_emb, label_to_embed[k]])
            self.logger.info(f"SpaCy embeddings generated, shape: {unique_emb.shape}. Applying t-SNE for dimensionality reduction to {self.embedding_dim} dimensions.")

            tsne = TSNE(n_components=self.embedding_dim, method=self.tsne_method)
            red_emb = tsne.fit_transform(unique_emb)
            self.diag_embeddings = red_emb

            # ======================== VALIDATION SECTION - TRUSTWORTHINESS & CONTINUITY ========================
            self.logger.info("Calculating Trustworthiness and Continuity for t-SNE embeddings...")
            trustworthiness_score = trustworthiness(unique_emb, red_emb, n_neighbors=5) # n_neighbors is a parameter, can be adjusted
            continuity_score = continuity(unique_emb, red_emb, n_neighbors=5) # n_neighbors should be consistent with trustworthiness

            self.logger.info(f"t-SNE dimensionality reduction completed, embeddings shape: {red_emb.shape}.")
            self.logger.info(f"Validation Metrics:")
            self.logger.info(f"  Trustworthiness (n_neighbors=5): {trustworthiness_score:.4f} - Measures how well the local neighborhood is preserved from high to low dimension.") # Interpretation added
            self.logger.info(f"  Continuity      (n_neighbors=5): {continuity_score:.4f}      - Measures how well the local neighborhood is preserved from low to high dimension.") # Interpretation added
            self.logger.info("  Interpretation of scores: Scores range from 0 to 1, with higher values indicating better preservation of original data structure.") # General interpretation
            self.logger.info("  - Trustworthiness close to 1 means points close in the embedding are also close in the original space (high confidence in embedding's local structure).") # More detailed interpretation
            self.logger.info("  - Continuity close to 1 means points close in the original space remain close in the embedding (less distortion of original neighborhoods).") # More detailed interpretation
            self.logger.info("  - There is no absolute 'good' score, but scores closer to 1 are generally desirable. Compare scores across different settings or methods.") # Contextual interpretation
            # ======================== END VALIDATION SECTION ========================

            self.logger.info(f"Saving diagnosis embeddings to: {self.diag_embeddings_path}")
            np.save(self.diag_embeddings_path, red_emb)
            self.logger.info("Diagnosis embeddings saved.")

            # Handle Label Encoding for Other Categorical Columns (NEW SECTION)
            label_encoders_json_to_save = {} # Dictionary to store label encoder mappings for JSON

            for column in self.label_encode_columns:
                self.logger.info(f"Fitting and applying LabelEncoder for column: {column}")
                le_other = LabelEncoder() # Create new LabelEncoder instance for each column
                df[column] = le_other.fit_transform(df[column]).astype(int) # Fit and transform column, ensure int type
                self.other_label_encoders[column] = le_other # Store fitted LabelEncoder instance

                # Prepare mapping and classes_ to save to JSON
                mapping_to_save = dict(zip(le_other.classes_.astype(str), le_other.transform(le_other.classes_))) # Mapping to save as JSON needs string keys
                label_encoders_json_to_save[column] = { # Structure to save per column info
                    'classes_': le_other.classes_.tolist(), # classes_ needs to be serializable
                    'mapping': mapping_to_save
                }
                self.logger.info(f"Label encoding applied and encoder prepared for column: {column}")

            # Save all other label encoders' mappings to a single JSON file
            self.logger.info(f"Saving label encoders for additional columns to: {self.label_encoders_path}")
            with open(self.label_encoders_path, 'w') as f:
                json.dump(label_encoders_json_to_save, f, indent=4) # Save with indent for readability
            self.logger.info("Label encoders for additional columns saved.")


            return df

        except FileNotFoundError as e:
            self.logger.error(f"ICD-9 data file not found: {e}")
            raise
        except spacy.util. মডেলNotFoundException as e:
            self.logger.error(f"SpaCy model '{self.spacy_model_name}' not found: {e}")
            self._nlp = None
            raise
        except Exception as e:
            self.logger.error(f"Error calculating diagnosis embeddings or additional label encoding: {e}")
            self._nlp = None
            raise
        finally: # Ensure SpaCy model is unloaded to free memory if loaded in this method
            if self._nlp is not None:
                self._nlp = None # Unload SpaCy model after use in calculation to save memory


    @staticmethod
    @lru_cache(maxsize=1024)
    def _find_diag_description(idc9_hierarchy: List[Dict], code: str) -> Union[Dict, str]:
        """
        Finds diagnosis descriptions (specific and general) from ICD-9 hierarchy data for a given code.
        Implements caching to optimize performance.
        """
        if code == '0':
            return 'No diagnosis'
        if not pd.isna(code):
            code = str(code)
            if len(code) == 1:
                code = '00'+code
            elif len(code) == 2:
                code = '0'+code
            if '.' in code:
                code = code.replace('.','')
                for d in idc9_hierarchy:
                    if d['icd9'] == code or d['icd9'] == code+'0':
                        if 'subchapter' in d.keys():
                            return {'specific':d['descLong'],'general':d['subchapter']+'. '+d['chapter']}
                        else:
                            return {'specific':d['descLong'],'general':d['chapter']}
            else:
                for d in idc9_hierarchy:
                    if d['threedigit'] == code:
                        if 'subchapter' in d.keys():
                            return {'specific':d['major'],'general':d['subchapter']+'. '+d['chapter']}
                        else:
                            return {'specific':d['major'],'general':d['chapter']}
        else:
            return 'No diagnosis'

        logger.warning(f"Diagnosis code '{code}' not found in ICD-9 hierarchy.")
        return 'Not found'


    @staticmethod
    def _create_embedding(nlp, desc: Union[Dict, str], weight_specific: float, weight_general: float) -> np.ndarray:
        """
        Creates a SpaCy embedding vector for a diagnosis description using weighted average.
        """
        if isinstance(desc, dict):
            emb_specific = nlp(desc['specific']).vector
            emb_general = nlp(desc['general']).vector
            emb = weight_specific * emb_specific + weight_general * emb_general
            return emb
        else:
            return nlp(desc).vector

    @staticmethod
    def _load_icd9_hierarchy_data(path: str) -> List[Dict]:
        """Loads ICD-9 hierarchy data from a JSON file."""
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
        """Loads ICD-9 chapters data from a JSON file."""
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


if __name__ == "__main__":
    """
    Main execution block to demonstrate the SecondPhasePreprocessor class.
    """
    from config import ( # Import configurations
        RAW_DATA_PATH,
        MID_PROCESSING_PATH,
        NO_MISSINGS_PATH,
        NO_MISSINGS_COD_PATH, # Assuming you'll define this in config.py
        DIAG_EMBEDDINGS_PATH,
        DIAG_LABEL_ENCODER_PATH,
        LABEL_ENCODERS_PATH, # Assuming you'll define this in config.py
        ICD9_HIERARCHY_PATH,
        ICD9_CHAPTERS_PATH,
        SPACY_MODEL_NAME,
        MISSING_VALUES # Import MISSING_VALUES if load_data is used from FirstPhasePreprocessor
    )
    from preprocessing import FirstPhasePreprocessor # Import FirstPhasePreprocessor

    # ======================== MAIN SCRIPT LOGGER CONFIGURATION ========================
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/preprocessing_phase2_demo.log")
        ]
    )
    main_script_logger = logging.getLogger(__name__)
    main_script_logger.info("=== DEMO SCRIPT - SECOND PHASE PREPROCESSING STARTED ===")


    # Load data using FirstPhasePreprocessor's load_data method (or your own)
    df_no_na_diag = FirstPhasePreprocessor.load_data(NO_MISSINGS_PATH, MISSING_VALUES) # Or however you load your data
    main_script_logger.info(f"Data loaded for Phase 2, shape: {df_no_na_diag.shape}")

    # Example list of columns to label encode (replace with your actual list from config.py)
    LABEL_ENCODE_COLUMNS_EXAMPLE = ['discharge_disposition_id', 'admission_source_id']

    # Initialize SecondPhasePreprocessor with configurable parameters
    phase2_preprocessor = SecondPhasePreprocessor(
        diag_embeddings_path=DIAG_EMBEDDINGS_PATH,
        diag_label_encoder_path=DIAG_LABEL_ENCODER_PATH,
        label_encoders_path=LABEL_ENCODERS_PATH, # Path for saving other label encoders
        icd9_hierarchy_path=ICD9_HIERARCHY_PATH,
        icd9_chapters_path=ICD9_CHAPTERS_PATH,
        spacy_model_name=SPACY_MODEL_NAME,
        label_encode_columns=LABEL_ENCODE_COLUMNS_EXAMPLE, # Pass the list of columns to label encode
        embedding_dim=16, # Example: Configurable embedding dimension
        tsne_n_components=16, # Example: Configurable t-SNE components
        tsne_method='exact', # Example: Configurable t-SNE method
        embedding_weight_specific=0.8, # Example: Configurable weight for specific desc
        embedding_weight_general=0.2, # Example: Configurable weight for general desc
        logger=main_script_logger # Pass main script logger for unified logging
    )

    # Apply the second phase preprocessing (embedding encoding and additional label encoding)
    df_encoded_phase2 = phase2_preprocessor.transform(df_no_na_diag.copy()) # Use .copy()

    main_script_logger.info(f"Data shape after Phase 2 preprocessing: {df_encoded_phase2.shape}")

    # Optional: Save the processed data and embeddings
    df_encoded_phase2.to_csv(NO_MISSINGS_COD_PATH, index=False) # Assuming NO_MISSINGS_COD_PATH is defined in config
    main_script_logger.info(f"Processed data with encoded diagnoses and additional label encoding saved to: {NO_MISSINGS_COD_PATH}")
    # Embeddings and label encoders are saved by the class itself during transform if needed

    main_script_logger.info("=== DEMO SCRIPT - SECOND PHASE PREPROCESSING COMPLETED ===")