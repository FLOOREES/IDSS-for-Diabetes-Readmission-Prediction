# src/preprocessing/first_phase.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib # For saving/loading sklearn objects
import json
from typing import Dict, List, Optional, Union
import logging
import os 

# ======================== LOGGER CONFIG ========================
logger = logging.getLogger(__name__) # Module-level logger

class FirstPhasePreprocessor:
    """
    Performs the first phase of preprocessing on the diabetes dataset.
    Now includes stateful components for OneHotEncoder and Low Variance Column Removal,
    ensuring consistency for batch and single-instance processing.

    When transform is called without pre-loaded state, it will fit components
    on the provided data, save them, and then transform. For single-instance
    inference, load_fitted_state() should be called first.
    """

    def __init__(
        self,
        drop_columns: List[str],
        one_hot_columns: List[str], # Should include 'race' for this refactored version
        ordinal_mappings: Dict[str, Dict[str, int]],
        treatment_columns: List[str],
        treatment_mapping: Dict[str, int],
        missing_values_encoding: Dict[str, str],
        # New: Paths for saving/loading fitted components
        ohe_encoder_path: str,
        ohe_feature_names_path: str,
        low_variance_cols_path: str,
        logger: Optional[logging.Logger] = None
    ):
        self.drop_columns = drop_columns
        # In this refactored version, 'race' will be imputed then OHEd, so it should be in one_hot_columns.
        # The old logic of excluding it in __init__ and adding it later in _one_hot_encode can be simplified.
        self.one_hot_columns_config = one_hot_columns # Store the configured OHE columns
        self.ordinal_mappings = ordinal_mappings
        self.treatment_columns = treatment_columns
        self.treatment_mapping = treatment_mapping
        self.missing_values_encoding = missing_values_encoding
        
        self.ohe_encoder_path = ohe_encoder_path
        self.ohe_feature_names_path = ohe_feature_names_path
        self.low_variance_cols_path = low_variance_cols_path
        
        self.logger = logger or logging.getLogger(__name__)

        # Attributes to store fitted components and state
        self.ohe_encoder_: Optional[OneHotEncoder] = None
        self.ohe_feature_names_: Optional[List[str]] = None
        self.low_variance_columns_to_drop_: Optional[List[str]] = None
        self.fitted_components_loaded = False

        self.logger.info(f"FirstPhasePreprocessor initialized. Configured OHE columns: {self.one_hot_columns_config}")

    def _ensure_dir_exists(self, file_path: str):
        """Ensures the directory for a given file_path exists."""
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            self.logger.info(f"Created directory: {dir_name}")

    def fit_and_save_components(self, df: pd.DataFrame, columns_for_ohe: List[str]):
        """
        Fits OHE, identifies low-variance columns based on df, and saves them.
        This is called by transform if components are not already loaded/fitted.
        """
        self.logger.info("Fitting and saving components (OHE, Low Variance Columns)...")
        
        # --- Fit and Save OneHotEncoder ---
        self.ohe_encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.logger.info(f"Fitting OneHotEncoder on columns: {columns_for_ohe}")
        self.ohe_encoder_.fit(df[columns_for_ohe])
        self.ohe_feature_names_ = self.ohe_encoder_.get_feature_names_out(columns_for_ohe).tolist()
        
        self._ensure_dir_exists(self.ohe_encoder_path)
        joblib.dump(self.ohe_encoder_, self.ohe_encoder_path)
        self.logger.info(f"Saved fitted OneHotEncoder to {self.ohe_encoder_path}")
        
        self._ensure_dir_exists(self.ohe_feature_names_path)
        with open(self.ohe_feature_names_path, 'w') as f:
            json.dump(self.ohe_feature_names_, f)
        self.logger.info(f"Saved OHE feature names to {self.ohe_feature_names_path}")

        # --- Identify and Save Low Variance Columns (post-OHE simulation) ---
        # Temporarily apply OHE to identify low variance columns on the correct feature set
        df_temp_ohe = self.ohe_encoder_.transform(df[columns_for_ohe])
        df_temp_ohe = pd.DataFrame(df_temp_ohe, columns=self.ohe_feature_names_, index=df.index)
        
        # Drop original columns that were OHE'd, keep others
        df_for_low_var_check = df.drop(columns=columns_for_ohe)
        df_for_low_var_check = pd.concat([df_for_low_var_check, df_temp_ohe], axis=1)
        
        self.logger.info("Identifying low-variance columns on (simulated post-OHE) data.")
        cols_to_check = df_for_low_var_check.select_dtypes(exclude=['object']).columns
        nunique = df_for_low_var_check[cols_to_check].nunique()
        self.low_variance_columns_to_drop_ = nunique[nunique < 2].index.tolist()
        
        self._ensure_dir_exists(self.low_variance_cols_path)
        with open(self.low_variance_cols_path, 'w') as f:
            json.dump(self.low_variance_columns_to_drop_, f)
        self.logger.info(f"Identified and saved {len(self.low_variance_columns_to_drop_)} low-variance columns to {self.low_variance_cols_path}: {self.low_variance_columns_to_drop_}")
        
        self.fitted_components_loaded = True # Mark as fitted for this instance

    def load_fitted_state(self) -> bool:
        """
        Loads the fitted OneHotEncoder, its feature names, and low variance columns list.
        Returns True if loading was successful, False otherwise.
        """
        try:
            if not (os.path.exists(self.ohe_encoder_path) and \
                    os.path.exists(self.ohe_feature_names_path) and \
                    os.path.exists(self.low_variance_cols_path)):
                self.logger.warning("One or more fitted component files not found. Cannot load state.")
                return False

            self.logger.info(f"Loading fitted OneHotEncoder from {self.ohe_encoder_path}")
            self.ohe_encoder_ = joblib.load(self.ohe_encoder_path)
            
            self.logger.info(f"Loading OHE feature names from {self.ohe_feature_names_path}")
            with open(self.ohe_feature_names_path, 'r') as f:
                self.ohe_feature_names_ = json.load(f)
            
            self.logger.info(f"Loading low-variance columns list from {self.low_variance_cols_path}")
            with open(self.low_variance_cols_path, 'r') as f:
                self.low_variance_columns_to_drop_ = json.load(f)
            
            self.fitted_components_loaded = True
            self.logger.info("Successfully loaded all fitted components.")
            return True
        except Exception as e:
            self.logger.error(f"Error loading fitted components: {e}", exc_info=True)
            self.ohe_encoder_ = None # Ensure partial loads don't leave inconsistent state
            self.ohe_feature_names_ = None
            self.low_variance_columns_to_drop_ = None
            self.fitted_components_loaded = False
            return False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the first phase preprocessing steps."""
        self.logger.info("Starting data transformation pipeline (First Phase).")
        df_transformed = df.copy()

        df_transformed = self._load_data_handle_missings(df_transformed) # Handles '?' -> pd.NA
        
        diag_cols_to_fill = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols_to_fill:
            if col in df_transformed.columns:
                initial_na_count = df_transformed[col].isna().sum()
                if initial_na_count > 0:
                    self.logger.info(f"Filling {initial_na_count} NA values in '{col}' with string '0'.")
                    df_transformed[col] = df_transformed[col].fillna('0')

        df_transformed = self._drop_columns(df_transformed)
        df_transformed = self._ordinal_encode(df_transformed)
        df_transformed = self._treatment_encode(df_transformed)
        df_transformed = self._impute_missing_race(df_transformed) # 'race' is now clean, ready for OHE

        # Determine columns for OHE - includes 'race' now
        # Ensure configured columns actually exist in the df at this point
        actual_ohe_columns = [col for col in self.one_hot_columns_config if col in df_transformed.columns]
        if 'race' not in actual_ohe_columns and 'race' in df_transformed.columns:
             actual_ohe_columns.append('race') # Ensure imputed 'race' is included if present
        elif 'race' not in df_transformed.columns and 'race' in self.one_hot_columns_config:
            self.logger.warning("'race' configured for OHE but not found in DataFrame. Skipping 'race' for OHE.")
            actual_ohe_columns = [col for col in actual_ohe_columns if col != 'race']


        # --- Fit or Use Loaded OneHotEncoder ---
        if not self.fitted_components_loaded:
            # Attempt to load first if not already done by an explicit call to load_fitted_state
            if not self.load_fitted_state(): # If loading fails or files don't exist
                self.logger.info("Fitted components not loaded and not found on disk. Fitting now.")
                # Pass the actual OHE columns found in the current df
                self.fit_and_save_components(df_transformed, actual_ohe_columns)
            # If load_fitted_state was successful, self.fitted_components_loaded is True
            # If fit_and_save_components was successful, self.fitted_components_loaded is True
        
        if not self.ohe_encoder_ or not self.ohe_feature_names_:
            msg = "OHE encoder or feature names not available after fit/load attempt."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info(f"Applying OneHotEncoder using {'loaded' if self.fitted_components_loaded else 'newly fitted'} components.")
        if not actual_ohe_columns:
            self.logger.warning("No columns to one-hot encode after filtering for existing ones.")
        else:
            # Ensure columns passed to transform match those used in fit
            # This requires careful handling if actual_ohe_columns could be different from what encoder was fit on
            # For simplicity here, assume actual_ohe_columns are what the loaded/fitted encoder expects
            # or that fit_and_save_components used the correct actual_ohe_columns for the current df.
            # The OHE encoder was fitted on `columns_for_ohe` in `fit_and_save_components`.
            # We need to ensure we pass the same set of columns (or compatible) to `transform`.
            # The `self.ohe_encoder_.feature_names_in_` holds the names it was fitted on.
            
            # Columns the encoder was actually fitted on (available after fit or load)
            fitted_on_cols = self.ohe_encoder_.feature_names_in_.tolist()
            
            # Check if all columns the encoder was fitted on are present in the current df
            missing_for_transform = [col for col in fitted_on_cols if col not in df_transformed.columns]
            if missing_for_transform:
                msg = f"Cannot transform. OHE encoder was fit on columns not all present in current DataFrame: {missing_for_transform}"
                self.logger.error(msg)
                raise ValueError(msg)

            ohe_data = self.ohe_encoder_.transform(df_transformed[fitted_on_cols])
            df_ohe_encoded = pd.DataFrame(ohe_data, columns=self.ohe_feature_names_, index=df_transformed.index)
            
            df_transformed = df_transformed.drop(columns=fitted_on_cols)
            df_transformed = pd.concat([df_transformed, df_ohe_encoded], axis=1)
            self.logger.info(f"One-hot encoding applied. New columns: {self.ohe_feature_names_[:5]}...")


        # --- Use Loaded Low Variance Columns List ---
        if not self.low_variance_columns_to_drop_ and not self.fitted_components_loaded:
            # This case should have been handled by fit_and_save_components if we got here without loading
            self.logger.warning("Low variance columns list not available, though components should have been fitted. This might indicate an issue.")
            # Fallback: do nothing, or re-calculate (but re-calculation here is against fit/transform paradigm for transform only)
            # For now, if it's not loaded and not fitted in this call (which it should have been if not loaded), it means an issue.
            # The robust way is that `fit_and_save_components` MUST set this list.
            if self.low_variance_columns_to_drop_ is None: # Still none after all attempts
                 self.low_variance_columns_to_drop_ = []
                 self.logger.error("Low variance column list is still None after fit/load attempts. No low variance columns will be dropped.")


        if self.low_variance_columns_to_drop_:
            # Only drop columns that actually exist in the current df_transformed
            cols_to_actually_drop = [col for col in self.low_variance_columns_to_drop_ if col in df_transformed.columns]
            if cols_to_actually_drop:
                self.logger.info(f"Dropping {len(cols_to_actually_drop)} low-variance columns: {cols_to_actually_drop[:5]}...")
                df_transformed = df_transformed.drop(columns=cols_to_actually_drop)
            else:
                self.logger.info("Configured low-variance columns not found in current DataFrame. Nothing to drop.")
        else:
            self.logger.info("No low-variance columns configured to be dropped or list is empty.")
            
        self.logger.info("First phase preprocessing pipeline completed.")
        return df_transformed

    # --- Helper methods for transformations (largely unchanged, but ensure logging uses self.logger) ---
    def _load_data_handle_missings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Logs initial NA counts after pd.read_csv has converted placeholders.
        The actual replacement of placeholders (e.g., '?') with NA objects
        is expected to have been done by the static `load_data` method using
        `pd.read_csv(na_values=...)`.
        """
        self.logger.info(f"Logging NA counts. Original missing placeholders like '{list(self.missing_values_encoding.keys())[0] if self.missing_values_encoding else 'N/A'}' should now be pd.NA/np.nan.")
        
        if 'race' in df.columns:
            missing_race_initial = df['race'].isnull().sum()
            self.logger.info(f"Initial pd.NA/np.nan count in 'race': {missing_race_initial}")
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                missing_diag = df[col].isnull().sum()
                self.logger.info(f"Initial pd.NA/np.nan count in '{col}': {missing_diag}")
        return df # Return df unchanged by this specific method now

    @staticmethod
    def load_data(path: str, na_values: Union[List[str], Dict[str, List[str]]]) -> pd.DataFrame: # na_values usually just List[str] for read_csv
        """
        Loads data from a CSV file into a Pandas DataFrame.
        `na_values` here is what `pd.read_csv` expects (e.g., ['?']).
        """
        # The config sends {'?': pd.NA} which is for df.replace().
        # For pd.read_csv, na_values should be the actual string markers like ['?']
        actual_na_values_for_read_csv = list(na_values.keys()) if isinstance(na_values, dict) else na_values

        logger.info(f"Loading data: {path} with na_values={actual_na_values_for_read_csv}")
        try:
            df = pd.read_csv(path, na_values=actual_na_values_for_read_csv)
            logger.info(f"Data loaded. Shape: {df.shape}")
            # Logging of initial missing counts based on read_csv's interpretation of na_values
            if 'race' in df.columns:
                missing_race_initial = df['race'].isnull().sum()
                logger.info(f"Initial missing (interpreted by read_csv as NA) count in 'race': {missing_race_initial}")
            for col in ['diag_1', 'diag_2', 'diag_3']:
                if col in df.columns:
                    missing_diag = df[col].isnull().sum()
                    logger.info(f"Initial missing (interpreted by read_csv as NA) count in '{col}': {missing_diag}")
            return df
        except FileNotFoundError: logger.error(f"File not found: {path}"); raise
        except Exception as e: logger.error(f"Error loading data: {e}"); raise

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Dropping columns: {self.drop_columns}")
        cols_to_drop_present = [col for col in self.drop_columns if col in df.columns]
        if len(cols_to_drop_present) < len(self.drop_columns):
            missing_to_drop = set(self.drop_columns) - set(cols_to_drop_present)
            self.logger.warning(f"Columns specified to drop but not found in DataFrame: {list(missing_to_drop)}")
        if cols_to_drop_present:
            df = df.drop(columns=cols_to_drop_present)
        return df

    def _ordinal_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Ordinal encoding columns: {list(self.ordinal_mappings.keys())}")
        for col, mapping in self.ordinal_mappings.items():
            if col in df.columns:
                unmapped = df[col][~df[col].isin(mapping.keys()) & df[col].notna()].unique()
                if len(unmapped) > 0:
                    self.logger.warning(f"Column '{col}' has values not in ordinal mapping: {unmapped}. These will become NaN then -1.")
                df[col] = df[col].map(mapping)
                if df[col].isnull().any():
                    self.logger.info(f"NaNs generated during ordinal encoding for '{col}'. Filling with -1.")
                    df[col] = df[col].fillna(-1)
                df[col] = df[col].astype(int) # Ensure it's int after fillna
            else:
                self.logger.warning(f"Column '{col}' for ordinal encoding not found.")
        return df

    def _treatment_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Encoding treatment columns: {self.treatment_columns}")
        for col in self.treatment_columns:
            if col in df.columns:
                df[col] = df[col].map(self.treatment_mapping).fillna(0).astype(int)
            else:
                self.logger.warning(f"Treatment column '{col}' not found.")
        return df

    def _impute_missing_race(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'race' not in df.columns:
            self.logger.warning("Column 'race' not found, skipping imputation.")
            return df

        initial_missing = df['race'].isnull().sum() # Checks for pd.NA
        self.logger.info(f"Starting 'race' imputation. Initial pd.NA count: {initial_missing}")

        if initial_missing > 0:
            patient_id_col = 'patient_nbr' # Assuming this is available; could pass from config
            encounter_id_col = 'encounter_id' # Assuming this is available

            if patient_id_col not in df.columns or encounter_id_col not in df.columns:
                self.logger.error(f"'{patient_id_col}' or '{encounter_id_col}' required for race ffill/bfill but not found. Filling all missing 'race' with 'Unknown'.")
                df['race'].fillna('Unknown', inplace=True)
            else:
                self.logger.info(f"Attempting ffill/bfill for 'race' based on '{patient_id_col}' and '{encounter_id_col}'.")
                # Sort by patient then encounter to ensure correct fill order
                df_sorted_indices = df.sort_values([patient_id_col, encounter_id_col]).index
                # Apply fill operations on a Series sorted this way, then map back
                # This is safer than df.sort_values().groupby().ffill() which can be tricky with index alignment if not careful
                race_filled = df.loc[df_sorted_indices, 'race'].groupby(df.loc[df_sorted_indices, patient_id_col]).ffill()
                race_filled = race_filled.groupby(df.loc[df_sorted_indices, patient_id_col]).bfill()
                
                # Update the original DataFrame's 'race' column
                # Iterate through original index and new values to assign properly
                for original_idx, filled_value in zip(df_sorted_indices, race_filled):
                    df.loc[original_idx, 'race'] = filled_value

                missing_after_ffill = df['race'].isnull().sum()
                self.logger.info(f"Missing 'race' after ffill/bfill: {missing_after_ffill}")
                if missing_after_ffill > 0:
                    self.logger.info(f"Filling {missing_after_ffill} remaining missing 'race' with 'Unknown'.")
                    df['race'].fillna('Unknown', inplace=True)
        else:
            self.logger.info("No missing 'race' values (pd.NA) found to impute.")
        
        final_missing = df['race'].isnull().sum()
        if final_missing > 0:
             self.logger.error(f"CRITICAL: {final_missing} NaNs / pd.NA remain in 'race' after imputation!")
        else:
             self.logger.info(f"'race' imputation complete. Value counts:\n{df['race'].value_counts(dropna=False)}")
        return df