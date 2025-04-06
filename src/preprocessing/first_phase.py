# preprocessing.py

import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier # Not used, can remove
from typing import Dict, List, Optional
import logging

# ======================== LOGGER CONFIG ========================
logger = logging.getLogger(__name__) # Module-level logger as default

class FirstPhasePreprocessor:
    """
    Performs the first phase of preprocessing on the diabetes dataset.

    Corrected logic for 'race' imputation and encoding.
    Steps:
        - Loading data.
        - Dropping specified columns.
        - Ordinal encoding for specified columns.
        - Encoding treatment-related columns.
        - **Imputing missing 'race' values (ffill/bfill then 'Unknown').**
        - **One-hot encoding ('gender', 'admission_type_id', 'race').**
        - Removing columns with near-zero variance.
    """

    def __init__(
        self,
        drop_columns: List[str],
        one_hot_columns: List[str],
        ordinal_mappings: Dict[str, Dict[str, int]],
        treatment_columns: List[str],
        treatment_mapping: Dict[str, int],
        missing_values_encoding: Dict[str, str],
        logger: logging.Logger = None
    ):
        """
        Initializes the FirstPhasePreprocessor with preprocessing configurations.

        Parameters
        ----------
        drop_columns : List[str]
            List of column names to be dropped.
        one_hot_columns : List[str]
            List of column names for one-hot encoding, *excluding 'race'* (race is handled separately).
        ordinal_mappings : Dict[str, Dict[str, int]]
            Dictionary defining mappings for ordinal encoding.
        treatment_columns : List[str]
            List of column names representing treatment features to be encoded.
        treatment_mapping : Dict[str, int]
            Dictionary defining the mapping for treatment encoding.
        missing_values_encoding : Dict[str, str]
            Dictionary specifying how missing values are encoded in the raw data (e.g., {'?': pd.NA}).
        logger : logging.Logger, optional
            Logger instance for logging messages. If None, a module-level logger is used.
        """
        self.drop_columns = drop_columns
        self.one_hot_columns = [col for col in one_hot_columns if col != 'race'] # Exclude 'race' from general OHE
        self.ordinal_mappings = ordinal_mappings
        self.treatment_columns = treatment_columns
        self.treatment_mapping = treatment_mapping
        self.missing_values_encoding = missing_values_encoding
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"FirstPhasePreprocessor initialized. OHE columns (excluding race): {self.one_hot_columns}")
        if 'race' in one_hot_columns:
            self.logger.warning("'race' column specified in one_hot_columns; will be handled separately.")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the first phase preprocessing steps to the input DataFrame.

        ... (rest of the docstring remains the same) ...
        """
        self.logger.info("Starting data transformation pipeline (First Phase).")
        df = self._load_data_handle_missings(df.copy()) # Load data and handle initial missings
        
        # --- [NEW] Explicitly handle missing values in diagnosis columns ---
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols:
            if col in df.columns: # Check if column exists
                initial_na_count = df[col].isna().sum()
                if initial_na_count > 0:
                    self.logger.info(f"Filling {initial_na_count} missing values (pd.NA) in column '{col}' with '0'.")
                    df[col] = df[col].fillna('0') # Fill pd.NA with string '0'
                else:
                    self.logger.info(f"No missing values (pd.NA) found in column '{col}'.")
        # --- [END] Missing value handling for diagnosis columns ---

        df = self._drop_columns(df)
        df = self._ordinal_encode(df)
        df = self._treatment_encode(df)
        df = self._impute_missing_race(df)
        df = self._one_hot_encode(df)
        df = self._drop_low_variance_columns(df)
        self.logger.info("First phase preprocessing pipeline completed.")
        return df

    def _load_data_handle_missings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles initial data loading and replacement of missing value placeholders.

        This internal method directly replaces missing value placeholders (defined in `self.missing_values_encoding`)
        with `pd.NA` upon loading the data within the transform pipeline. It also logs the initial counts of missing
        values in 'race' and diagnosis columns after replacement.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (raw, directly after reading from CSV).

        Returns
        -------
        pd.DataFrame
            DataFrame with missing value placeholders replaced by `pd.NA`.
        """
        logger.info(f"Replacing missing value placeholders: {self.missing_values_encoding}")
        df_replaced = df.replace(self.missing_values_encoding)

        missing_race_initial = df_replaced['race'].isnull().sum()
        logger.info(f"Initial missing count in 'race' after placeholder replacement: {missing_race_initial}")
        # Log initial missing in diags as well
        for col in ['diag_1', 'diag_2', 'diag_3']:
             if col in df_replaced.columns:
                  missing_diag = df_replaced[col].isnull().sum()
                  logger.info(f"Initial missing count in '{col}' after placeholder replacement: {missing_diag}")
        return df_replaced


    @staticmethod
    def load_data(path: str, na_values: Dict) -> pd.DataFrame:
        """
        Loads data from a CSV file into a Pandas DataFrame.

        This static method provides a utility to load the dataset from a specified
        CSV file path. It handles missing values based on the provided encoding dictionary.

        Parameters
        ----------
        path : str
            The file path to the CSV file.
        na_values : Dict
            Dictionary defining how missing values are represented in the CSV.

        Returns
        -------
        pd.DataFrame
            The loaded Pandas DataFrame.
        """
        logger.info(f"Loading data: {path} with na_values={na_values}")
        try:
            df = pd.read_csv(path, na_values=na_values)
            logger.info(f"Data loaded. Shape: {df.shape}")
            missing_race_initial = df['race'].isnull().sum()
            logger.info(f"Initial missing ('{list(na_values.keys())[0]}') count in 'race': {missing_race_initial}")
            # Log initial missing in diags as well
            for col in ['diag_1', 'diag_2', 'diag_3']:
                 if col in df.columns:
                      missing_diag = df[col].isnull().sum()
                      logger.info(f"Initial missing ('{list(na_values.keys())[0]}') count in '{col}': {missing_diag}")
            return df
        except FileNotFoundError: logger.error(f"File not found: {path}"); raise
        except Exception as e: logger.error(f"Error loading data: {e}"); raise

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.

        This method removes columns listed in `self.drop_columns` from the input DataFrame.
        It logs the columns being dropped and handles cases where specified columns are not found.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with specified columns dropped.
        """
        self.logger.info(f"Dropping columns: {self.drop_columns}")
        cols_to_drop = [col for col in self.drop_columns if col in df.columns]
        if len(cols_to_drop) < len(self.drop_columns):
             missing = set(self.drop_columns) - set(cols_to_drop)
             self.logger.warning(f"Columns specified to drop but not found: {list(missing)}")
        return df.drop(cols_to_drop, axis=1, errors='ignore') # Add errors='ignore' for robustness


    def _ordinal_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies ordinal encoding to specified ordered categorical columns.

        This method maps categories in ordinal columns to numerical values based on
        the mappings defined in `self.ordinal_mappings`. It handles cases where columns
        are not found or when unmapped values are encountered, logging warnings accordingly.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with ordinal encoded columns.
        """
        ordinal_cols = list(self.ordinal_mappings.keys())
        self.logger.info(f"Ordinal encoding columns: {ordinal_cols}")
        for col, mapping in self.ordinal_mappings.items():
            if col in df.columns:
                # Add check for unmapped values before converting type
                unmapped = df[col][~df[col].isin(mapping.keys())].unique()
                if len(unmapped) > 0:
                    self.logger.warning(f"Column '{col}' has values not in ordinal mapping: {unmapped}. These will become NaN.")
                df[col] = df[col].map(mapping)
                # Handle potential NaNs from mapping before converting to int
                if df[col].isnull().any():
                    self.logger.warning(f"NaNs generated during ordinal encoding for '{col}'. Filling with -1.")
                    df[col] = df[col].fillna(-1) # Or choose another placeholder
                df[col] = df[col].astype(int)
            else:
                self.logger.warning(f"Column '{col}' not found, skipping ordinal encoding.")
        return df


    def _treatment_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes treatment columns using the provided treatment mapping.

        This method maps treatment-related columns to numerical values based on
        the mappings defined in `self.treatment_mapping`. It fills any NaN values
        resulting from the mapping (or pre-existing NaNs) with 0, assuming 'No treatment'.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with treatment columns encoded.
        """
        self.logger.info(f"Encoding treatment columns: {self.treatment_columns}")
        for col in self.treatment_columns:
            if col in df.columns:
                # Map known values, fill remaining (likely NaN or maybe 'No') with 0
                df[col] = df[col].map(self.treatment_mapping).fillna(0).astype(int)
            else:
                self.logger.warning(f"Treatment column '{col}' not found, skipping.")
        return df

    def _impute_missing_race(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing 'race' values using a combination of forward-fill, backward-fill, and 'Unknown' category.

        This method first attempts to impute missing 'race' values by propagating known 'race' values
        within each patient's visit history using forward-fill and backward-fill. If missing values still
        remain after this process, they are imputed with the 'Unknown' category.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing 'race' values imputed.
        """
        if 'race' not in df.columns:
            self.logger.warning("Column 'race' not found, skipping imputation.")
            return df

        initial_missing = df['race'].isnull().sum()
        self.logger.info(f"Starting 'race' imputation. Initial missing: {initial_missing}")

        if initial_missing > 0:
            # Ensure encounter_id exists for sorting
            if 'encounter_id' not in df.columns:
                 self.logger.error("Column 'encounter_id' required for race ffill/bfill but not found. Skipping ffill/bfill.")
            else:
                self.logger.info("Attempting ffill/bfill for 'race' based on 'patient_nbr'.")
                # Use .copy() to avoid SettingWithCopyWarning if df is a slice
                df_sorted = df.sort_values(['patient_nbr', 'encounter_id']).copy()
                df_sorted['race'] = df_sorted.groupby('patient_nbr')['race'].ffill()
                df_sorted['race'] = df_sorted.groupby('patient_nbr')['race'].bfill()
                # Assign back using .loc to ensure index alignment
                df.loc[df_sorted.index, 'race'] = df_sorted['race']
                missing_after_ffill = df['race'].isnull().sum()
                self.logger.info(f"Missing 'race' after ffill/bfill: {missing_after_ffill}")

                if missing_after_ffill > 0:
                    self.logger.info(f"Filling {missing_after_ffill} remaining missing 'race' with 'Unknown'.")
                    df['race'].fillna('Unknown', inplace=True) # Directly fill remaining NaNs
                else:
                    self.logger.info("No missing 'race' after ffill/bfill.")
        else:
            self.logger.info("No missing 'race' values found to impute.")

        # Final verification
        final_missing = df['race'].isnull().sum()
        if final_missing > 0:
             self.logger.error(f"CRITICAL: {final_missing} NaNs remain in 'race' after imputation!")
        else:
             self.logger.info("'race' imputation complete. No NaNs remain.")
             # Log value counts after imputation
             self.logger.info(f"Value counts for 'race' after imputation:\n{df['race'].value_counts()}")

        return df


    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to specified nominal categorical columns, including 'race'.

        This method performs one-hot encoding on the columns listed in `self.one_hot_columns`
        and specifically includes the 'race' column (which should have been imputed by this stage).
        It logs the columns being encoded and handles potential errors during the encoding process.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one-hot encoded columns.
        """
        columns_to_encode = self.one_hot_columns + ['race']
        columns_to_encode = [col for col in columns_to_encode if col in df.columns]

        if not columns_to_encode:
             self.logger.warning("No columns found for one-hot encoding.")
             return df

        self.logger.info(f"One-hot encoding columns: {columns_to_encode}")

        # Check for unexpected values like '?' in race just before OHE
        if 'race' in columns_to_encode and '?' in df['race'].unique():
             self.logger.error("CRITICAL: Value '?' still present in 'race' column before OHE. Imputation failed.")
             # Decide how to handle: raise error or try to fix? Let's raise for now.
             raise ValueError("Value '?' found in 'race' column before OHE - check imputation logic.")

        try:
            # dummy_na=False is correct as we handle missing via 'Unknown' category
            df_encoded = pd.get_dummies(df, columns=columns_to_encode, dtype=int, dummy_na=False)
            original_cols = set(df.columns)
            new_cols = set(df_encoded.columns) - original_cols
            self.logger.info(f"Created {len(new_cols)} new columns via OHE.")

            # Explicitly check for 'race_?' and rename 'race_Unknown' if needed
            # Although the imputation should prevent 'race_?', this is a safeguard.
            if 'race_?' in df_encoded.columns:
                 self.logger.warning("Column 'race_?' was created by get_dummies. Renaming to 'race_Unknown'. Check imputation.")
                 # If 'race_Unknown' also exists, maybe combine them? For now, just rename 'race_?'.
                 # This assumes 'race_Unknown' might not exist if no NaNs were left for the 'Unknown' category.
                 df_encoded.rename(columns={'race_?': 'race_Unknown'}, inplace=True)
                 if 'race_Unknown' in original_cols: # Should not happen if OHE worked correctly
                      self.logger.error("Original 'race_Unknown' column detected after OHE created 'race_?'. Investigate.")

            # Verify the expected 'race_Unknown' column exists if there were 'Unknown' values
            if 'Unknown' in df['race'].unique() and 'race_Unknown' not in df_encoded.columns:
                 self.logger.error("Value 'Unknown' was present in 'race', but 'race_Unknown' column was not created by OHE.")
            elif 'race_Unknown' in df_encoded.columns:
                 self.logger.info(f"Column 'race_Unknown' created successfully (count={df_encoded['race_Unknown'].sum()}).")


            return df_encoded
        except Exception as e:
            self.logger.error(f"Error during one-hot encoding: {e}")
            raise

    def _drop_low_variance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns with near-zero variance (nunique < 2).

        This method identifies and removes columns that have very little variance,
        specifically columns where the number of unique values is less than 2.
        It excludes object-type columns from the check to avoid potential issues with `nunique()` calculation on objects.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with low-variance columns dropped.
        """
        self.logger.info("Checking for low-variance columns (nunique < 2).")
        # Exclude object columns explicitly as nunique might be slow/problematic
        cols_to_check = df.select_dtypes(exclude=['object']).columns
        nunique = df[cols_to_check].nunique()
        dropped_cols = nunique[nunique < 2].index.tolist()

        if dropped_cols:
            self.logger.info(f"Dropping low-variance columns: {dropped_cols}")
            return df.drop(dropped_cols, axis=1)
        else:
            self.logger.info("No low-variance columns found to drop.")
            return df


# --- Main execution block (for testing) ---
if __name__ == "__main__":
    from config import ( # Import configuration parameters
        RAW_DATA_PATH, MID_PROCESSING_PATH, NO_MISSINGS_PATH,
        MISSING_VALUES, DROP_COLUMNS, ONE_HOT_COLUMNS,
        ORDINAL_MAPPINGS, TREATMENT_COLUMNS, TREATMENT_MAPPING
    )

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/preprocessing_class_main_demo.log", mode='w')
        ]
    )
    main_script_logger = logging.getLogger(__name__)
    main_script_logger.info("=== DEMO SCRIPT - FIRST PHASE PREPROCESSING (Corrected Race) ===")

    df_raw = FirstPhasePreprocessor.load_data(RAW_DATA_PATH, MISSING_VALUES)
    main_script_logger.info(f"Original data shape: {df_raw.shape}")
    main_script_logger.info(f"Original 'race' value counts:\n{df_raw['race'].value_counts(dropna=False)}")

    # NOTE: 'race' should NOT be in ONE_HOT_COLUMNS from config for this corrected class
    if 'race' in ONE_HOT_COLUMNS:
         main_script_logger.warning("Config issue: 'race' should not be in ONE_HOT_COLUMNS list in config.py for the corrected preprocessor.")
         ONE_HOT_COLUMNS_CORRECTED = [col for col in ONE_HOT_COLUMNS if col != 'race']
    else:
         ONE_HOT_COLUMNS_CORRECTED = ONE_HOT_COLUMNS

    preprocessor = FirstPhasePreprocessor(
        drop_columns=DROP_COLUMNS,
        one_hot_columns=ONE_HOT_COLUMNS_CORRECTED, # Pass list without 'race'
        ordinal_mappings=ORDINAL_MAPPINGS,
        treatment_columns=TREATMENT_COLUMNS,
        treatment_mapping=TREATMENT_MAPPING,
        missing_values_encoding=MISSING_VALUES,
        logger=main_script_logger
    )

    df_processed = preprocessor.transform(df_raw) # Use .copy() inside transform now

    main_script_logger.info(f"Processed data shape: {df_processed.shape}")
    main_script_logger.info(f"Columns after processing: {df_processed.columns.tolist()}")

    # Check race columns specifically
    race_cols_after = [col for col in df_processed.columns if col.startswith('race_')]
    main_script_logger.info(f"Resulting 'race' related columns: {race_cols_after}")
    if not race_cols_after:
         main_script_logger.error("No 'race_*' columns found after OHE - check logic.")
    elif 'race_Unknown' in df_processed.columns:
         main_script_logger.info(f"Count of 'race_Unknown' == 1: {df_processed['race_Unknown'].sum()}")
    else:
         main_script_logger.info("'race_Unknown' column not present (perhaps no missing race values after ffill/bfill).")


    # Check if original 'race' column still exists (it shouldn't after OHE)
    if 'race' in df_processed.columns:
         main_script_logger.error("Original 'race' column still exists after OHE!")

    # Check for NaNs in the final DataFrame (optional but good practice)
    nan_counts = df_processed.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        main_script_logger.warning(f"NaN values found in final DataFrame:\n{nan_cols}")
    else:
        main_script_logger.info("No NaN values found in the final DataFrame.")


    main_script_logger.info("=== DEMO SCRIPT - FIRST PHASE PREPROCESSING COMPLETED ===")