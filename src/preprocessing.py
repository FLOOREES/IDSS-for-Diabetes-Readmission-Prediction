import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional
import logging

# ======================== LOGGER CONFIG ========================
logger = logging.getLogger(__name__) # Module-level logger as default

class FirstPhasePreprocessor:
    """
    Performs the first phase of preprocessing on the diabetes dataset.
    This class is responsible for data loading and initial feature engineering steps,
    preparing the data for further preprocessing and modeling.
    The steps include:
        - Loading data from a CSV file.
        - Dropping specified columns.
        - One-hot encoding for nominal categorical features.
        - Ordinal encoding for ordered categorical features.
        - Encoding treatment-related columns based on a provided mapping.
        - Removing columns with near-zero variance.
        - Imputing missing 'race' values using forward-fill and backward-fill based on patient history.
        - Imputing any remaining missing 'race' values with 'Unknown' and one-hot encoding.
    """

    def __init__(
        self,
        drop_columns: List[str],
        one_hot_columns: List[str],
        ordinal_mappings: Dict[str, Dict[str, int]],
        treatment_columns: List[str],
        treatment_mapping: Dict[str, int],
        missing_values_encoding: Dict[str, str], # Added missing_values_encoding to constructor
        logger: logging.Logger = None
    ):
        """
        Initializes the FirstPhasePreprocessor with preprocessing configurations.

        Parameters
        ----------
        drop_columns : List[str]
            List of column names to be dropped from the DataFrame.
        one_hot_columns : List[str]
            List of column names to be one-hot encoded.
        ordinal_mappings : Dict[str, Dict[str, int]]
            Dictionary defining mappings for ordinal encoding.
            Keys are column names, and values are dictionaries mapping categories to ordinal values.
        treatment_columns : List[str]
            List of column names representing treatment features to be encoded.
        treatment_mapping : Dict[str, int]
            Dictionary defining the mapping for treatment encoding.
            Maps treatment categories (e.g., 'No', 'Steady', 'Up') to numerical values.
        missing_values_encoding : Dict[str, str]
            Dictionary specifying how missing values are encoded in the raw data,
            e.g., {'?': pd.NA} to recognize '?' as missing values.
        logger : logging.Logger, optional
            Logger instance to use for logging messages.
            If None, a module-level logger is used by default.
        """
        self.drop_columns = drop_columns
        self.one_hot_columns = one_hot_columns
        self.ordinal_mappings = ordinal_mappings
        self.treatment_columns = treatment_columns
        self.treatment_mapping = treatment_mapping
        self.missing_values_encoding = missing_values_encoding # Store missing value encoding
        self.logger = logger or logging.getLogger(__name__) # Use provided logger or default to module logger

        self.logger.info("FirstPhasePreprocessor initialized with configurations.")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all first phase preprocessing steps to the input DataFrame.

        This method orchestrates the sequence of preprocessing operations defined
        in the FirstPhasePreprocessor class. It takes a raw DataFrame as input and
        returns a preprocessed DataFrame, ready for subsequent preprocessing phases or modeling.

        Parameters
        ----------
        df : pd.DataFrame
            The raw input DataFrame to be preprocessed.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame after applying all feature engineering steps.
        """
        self.logger.info("Starting data transformation pipeline.")
        df = self._drop_columns(df)
        df = self._one_hot_encode(df)
        df = self._ordinal_encode(df)
        df = self._treatment_encode(df)
        df = self._drop_low_variance_columns(df)
        df = self._impute_missing_race_ffill_bfill(df)
        df = self._impute_remaining_missing_race(df)

        self.logger.info("First phase preprocessing pipeline completed.")
        return df

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
        logger.info(f"Loading data from CSV file: {path}") # Use module level logger for static method
        try:
            df = pd.read_csv(path, na_values=na_values)
            logger.info(f"Data loaded successfully. DataFrame shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found at path: {path}")
            raise # Re-raise the exception after logging
        except Exception as e:
            logger.error(f"Error occurred while loading data: {e}")
            raise # Re-raise other exceptions as well


    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops specified columns from the DataFrame.

        This private method removes columns listed in `self.drop_columns` from the input DataFrame.
        It uses `errors='ignore'` to prevent errors if a column does not exist.

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
        return df.drop(self.drop_columns, axis=1, errors='ignore')

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies one-hot encoding to specified nominal categorical columns.

        This private method performs one-hot encoding on the columns listed in `self.one_hot_columns`.
        It uses `pd.get_dummies` to convert categorical variables into a numerical representation.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one-hot encoded columns.
        """
        self.logger.info(f"One-hot encoding columns: {self.one_hot_columns}")
        return pd.get_dummies(df, columns=self.one_hot_columns, dtype=int, errors='ignore')

    def _ordinal_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies ordinal encoding to specified ordered categorical columns.

        This private method maps categories in ordinal columns to numerical values based on
        the mappings defined in `self.ordinal_mappings`.

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
                df[col] = df[col].map(mapping).astype(int)
            else:
                self.logger.warning(f"Column '{col}' not found in DataFrame, skipping ordinal encoding for this column.")
        return df

    def _treatment_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes treatment columns based on the provided treatment mapping.

        This private method encodes columns related to medical treatments using the mapping
        defined in `self.treatment_mapping`. It fills missing mappings with 0, assuming 'No treatment'.

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
                df[col] = df[col].map(self.treatment_mapping).fillna(0).astype(int)
            else:
                self.logger.warning(f"Treatment column '{col}' not found in DataFrame, skipping treatment encoding for this column.")
        return df

    def _drop_low_variance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops columns with near-zero variance (nunique < 2).

        This private method identifies and removes columns that have very little variance,
        specifically columns where the number of unique values is less than 2.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with low-variance columns dropped.
        """
        dropped_cols = df.columns[df.nunique() < 2]
        if not dropped_cols.empty:
            self.logger.info(f"Dropping low-variance columns: {dropped_cols.tolist()}")
            return df.drop(dropped_cols, axis=1)
        else:
            self.logger.info("No low-variance columns found to drop.")
            return df

    def _impute_missing_race_ffill_bfill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing 'race' values using forward-fill and then backward-fill.

        This private method imputes missing values in 'race' columns by first using forward-fill
        within each patient group (`patient_nbr`) and then applying backward-fill to catch any
        remaining missing values at the beginning of patient records.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'race' imputed using ffill and bfill.
        """
        race_cols = [col for col in df.columns if col.startswith('race_')]
        missing_race_count = (df[race_cols].sum(axis=1) == 0).sum()

        if missing_race_count > 0:
            self.logger.info(f"Imputing {missing_race_count} missing race rows using ffill/bfill.")
            df.loc[df[race_cols].sum(axis=1) == 0, race_cols] = pd.NA
            df = df.sort_values('encounter_id')
            df[race_cols] = df.groupby('patient_nbr')[race_cols].ffill()
            df[race_cols] = df.groupby('patient_nbr')[race_cols].bfill()
        else:
            self.logger.info("No missing race values to impute with ffill/bfill.")
        return df

    def _impute_remaining_missing_race(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes remaining missing 'race' values with 'Unknown' and one-hot encodes it.

        This private method handles any 'race' values that are still missing after ffill and bfill.
        It imputes these remaining missing values by assigning them to the 'Unknown' category
        and ensures that the one-hot encoded 'race_Unknown' column is correctly updated.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with remaining missing 'race' values imputed as 'Unknown' and one-hot encoded.
        """
        race_cols = [col for col in df.columns if col.startswith('race_')]
        missing_race_count_remaining = (df[race_cols].isnull().all(axis=1)).sum()

        if missing_race_count_remaining > 0:
            self.logger.info(f"Imputing remaining {missing_race_count_remaining} missing race values with 'Unknown' and one-hot encoding.")
            unknown_race_mask = df[race_cols].isnull().all(axis=1)
            for col in race_cols:
                df.loc[unknown_race_mask, col] = 0
            if 'race_Unknown' in df.columns:
                df.loc[unknown_race_mask, 'race_Unknown'] = 1
            else:
                df['race_Unknown'] = 0
                df.loc[unknown_race_mask, 'race_Unknown'] = 1

        else:
             self.logger.info("No remaining missing race values to impute with 'Unknown'.")
        return df


if __name__ == "__main__":
    """
    Main execution block to demonstrate the FirstPhasePreprocessor class.

    This section is for testing and demonstration purposes only. When the script is run directly,
    it initializes the preprocessor, loads data, applies the preprocessing steps, and logs the progress.
    In a typical pipeline setup, this class would be imported and used in a separate main script
    where data loading and preprocessing are orchestrated as part of a larger workflow.
    """
    from config import ( # Import configuration parameters from config.py for demonstration
        RAW_DATA_PATH,
        MID_PROCESSING_PATH,
        NO_MISSINGS_PATH,
        MISSING_VALUES,
        DROP_COLUMNS,
        ONE_HOT_COLUMNS,
        ORDINAL_MAPPINGS,
        TREATMENT_COLUMNS,
        TREATMENT_MAPPING
    )

    # ======================== MAIN SCRIPT LOGGER CONFIGURATION (for demonstration) ========================
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/preprocessing_class_main_demo.log") # Log file for demonstration
        ]
    )
    main_script_logger = logging.getLogger(__name__)

    main_script_logger.info("=== DEMO SCRIPT - FIRST PHASE PREPROCESSING WITH CLASS STARTED ===")

    # Load data using the static method from the preprocessor class
    df_raw = FirstPhasePreprocessor.load_data(RAW_DATA_PATH, MISSING_VALUES)
    main_script_logger.info(f"Original data shape loaded: {df_raw.shape}")

    # Initialize the preprocessor class with configurations and (optionally) the logger
    preprocessor = FirstPhasePreprocessor(
        drop_columns=DROP_COLUMNS,
        one_hot_columns=ONE_HOT_COLUMNS,
        ordinal_mappings=ORDINAL_MAPPINGS,
        treatment_columns=TREATMENT_COLUMNS,
        treatment_mapping=TREATMENT_MAPPING,
        missing_values_encoding=MISSING_VALUES, # Pass missing value encoding to constructor
        # logger=main_script_logger # Optional: Pass the main script logger for unified logging
    ) # If logger is not passed, it defaults to the module-level logger

    # Apply the preprocessing transformations using the transform method
    df_processed = preprocessor.transform(df_raw.copy()) # Use .copy() to avoid modifying original DataFrame

    main_script_logger.info(f"Processed data shape after first phase: {df_processed.shape}")

    # Optional: Save the processed data to a CSV file
    # df_processed.to_csv(MID_PROCESSING_PATH, index=False)
    # main_script_logger.info(f"Processed data saved to: {MID_PROCESSING_PATH}")

    main_script_logger.info("=== DEMO SCRIPT - FIRST PHASE PREPROCESSING WITH CLASS COMPLETED ===")