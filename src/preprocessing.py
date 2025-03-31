import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional
from config import (
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
import logging

# ======================== LOGGER CONFIG ========================

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbosity
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("logs/preprocessing.log")  # Save to log file
    ]
)

logger = logging.getLogger(__name__)

# ======================== FUNCTIONS ========================

def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """

    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def feature_engineering(
    df: pd.DataFrame,
    encode: Dict[str, str],
    discard: List[str],
    one_hot: List[str],
    ordinal: Dict[str, Dict[str, int]],
    treatment_cols: List[str],
    treatment_map: Dict[str, int],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform feature engineering steps including:
    - Value replacement (e.g., missing values)
    - Dropping irrelevant columns
    - One-hot encoding for nominal variables
    - Ordinal encoding for ordered categorical variables
    - Treatment dose encoding for medications
    - Low-variance feature removal
    - Forward/backward filling of missing race values based on patient history
    - Optionally save the resulting DataFrame to CSV

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    encode : Dict[str, str]
        Mapping of values to replace (e.g., {"?": pd.NA}).
    discard : List[str]
        List of columns to drop from the dataset.
    one_hot : List[str]
        Columns to apply one-hot encoding to (nominal features).
    ordinal : Dict[str, Dict[str, int]]
        Ordinal encoding mappings for ordered categorical variables.
    treatment_cols : List[str]
        Columns representing treatment types to encode as ordered.
    treatment_map : Dict[str, int]
        Mapping of treatment intensity levels (No < Down < Steady < Up).
    save_path : Optional[str], default=None
        If provided, the processed DataFrame will be saved as a CSV file at this path.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with features transformed.
    """
    logger.info("Starting feature engineering...")

    # Replace missing value placeholders
    df = df.replace(encode)
    logger.info("Missing values replaced.")

    # Drop irrelevant or noisy columns
    df = df.drop(discard, axis=1)
    logger.info(f"Dropped {len(discard)} columns.")

    # One-hot encode nominal categorical features
    df = pd.get_dummies(df, columns=one_hot, dtype=int)
    logger.info(f"Applied one-hot encoding to {len(one_hot)} columns.")

    # Apply ordinal encoding for ordered categorical features
    for col, mapping in ordinal.items():
        df[col] = df[col].map(mapping)
    logger.info(f"Mapped {len(ordinal)} ordinal columns.")

    # Encode treatment dose progression (ordinal)
    for col in treatment_cols:
        df[col] = df[col].map(treatment_map)
    logger.info(f"Encoded treatment columns: {len(treatment_cols)} mapped using treatment scale.")

    # Drop constant (zero-variance) columns
    dropped_cols = df.columns[df.nunique() < 2]
    df = df.drop(dropped_cols, axis=1)
    logger.info(f"Dropped {len(dropped_cols)} low-variance columns.")

    # Handle missing race (one-hot) using patient history
    race_cols = [col for col in df.columns if col.startswith('race')]
    missing_race_count = (df[race_cols].sum(axis=1) == 0).sum()

    df.loc[df[race_cols].sum(axis=1) == 0, race_cols] = pd.NA
    df = df.sort_values('encounter_id')
    df[race_cols] = df.groupby('patient_nbr')[race_cols].ffill()
    df[race_cols] = df.groupby('patient_nbr')[race_cols].bfill()

    logger.info(f"Filled {missing_race_count} missing race rows using ffill/bfill.")

    # Optionally save to CSV
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Feature-engineered DataFrame saved to {save_path}")

    return df


def impute_missing_race(
    df_prep: pd.DataFrame
) -> pd.DataFrame:
    """
    Impute missing race values by replacing them with 'Unknown'
    and then apply one-hot encoding.

    Parameters
    ----------
    df_prep : pd.DataFrame
        Preprocessed DataFrame including the 'race' column with pd.NA values.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'race' replaced and one-hot encoded.
    """
    logger.info("Starting race imputation using 'Unknown' strategy.")

    # Replace missing values with 'Unknown'
    missing_count = df_prep['race'].isna().sum()
    df_prep['race'] = df_prep['race'].fillna('Unknown')
    logger.info(f"Replaced {missing_count} missing values in 'race' with 'Unknown'.")

    # One-hot encode race
    race_onehot = pd.get_dummies(df_prep['race'], prefix='race', dtype=int)

    # Drop the original race column and add one-hot encoded version
    df_prep = pd.concat([df_prep.drop(columns=['race']), race_onehot], axis=1)
    logger.info("One-hot encoded 'race' and merged into dataframe.")

    return df_prep

# ======================== MAIN ========================

if __name__ == "__main__":
    """
    Main preprocessing pipeline:
    1. Load data
    2. Apply feature engineering
    3. Impute missing race values using MICE
    4. Save clean dataset
    """

    logger.info("=== PREPROCESSING STARTED ===")

    with tqdm(total=5, desc="Preprocessing", unit="step") as pbar:
        # Step 1: Load data
        df = load_data(path=RAW_DATA_PATH)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        pbar.update(1)

        # Step 2: Feature engineering
        df_encoded = feature_engineering(df, MISSING_VALUES, DROP_COLUMNS, ONE_HOT_COLUMNS, ORDINAL_MAPPINGS, TREATMENT_COLUMNS, TREATMENT_MAPPING, MID_PROCESSING_PATH)
        logger.info(f"After feature engineering: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns.")
        pbar.update(1)

        # Step 3: Impute missing race
        df_clean = impute_missing_race(df_encoded)
        logger.info(f"After imputation: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")
        pbar.update(1)

        # Step 4: Save to CSV
        df_clean.to_csv(NO_MISSINGS_PATH, index=False)
        logger.info(f"Data saved to {NO_MISSINGS_PATH}")
        pbar.update(1)

        # Step 5: Final validation, ensure no missing values
        missing_cols = df_clean.columns[df_clean.isna().any()].tolist()
        if missing_cols:
            missing_info = df_clean[missing_cols].isna().sum()
            logger.warning("Missing values remain in the final dataset:")
            for col, count in missing_info.items():
                logger.warning(f" - {col}: {count} missing")
        else:
            logger.info("No missing values detected in the final dataset.")
        

    logger.info("=== PREPROCESSING COMPLETE ===")
