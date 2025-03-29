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
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

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

# ======================== CUSTOM ITERATIVE IMPUTER ========================
class ProgressIterativeImputer(IterativeImputer):
    """ Extends the IterativeImputer to add a progress bar using tqdm. *** NOT WORKING *** """
    def _fit(self, X_filled, mask_missing_values, complete_mask):
        for iteration in tqdm(range(self.max_iter), desc="MICE Imputation Progress", unit="iter"):
            super()._fit(X_filled, mask_missing_values, complete_mask)
        return self

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
    df_prep: pd.DataFrame,
    exclude_vars: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Impute missing race values using IterativeImputer (MICE)
    with RandomForestClassifier as the estimator.

    Parameters
    ----------
    df_prep : pd.DataFrame
        Preprocessed DataFrame with missing values in 'race' (not one-hot encoded).
    exclude_vars : Optional[List[str]]
        List of variable prefixes to exclude from MICE input (e.g., high-cardinality).

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed race values (one-hot encoded).
    """
    logger.info("Starting race imputation using MICE + RandomForestClassifier")

    # 1. Prepare race for encoding
    race_series = df_prep['race']
    known_mask = race_series.notna()

    le = LabelEncoder()
    race_encoded = le.fit_transform(race_series[known_mask])
    logger.info(f"LabelEncoder classes: {list(le.classes_)}")

    race_full = pd.Series(pd.NA, index=race_series.index, dtype="Int64")
    race_full[known_mask] = race_encoded

    # 2. Drop IDs and optionally one-hot variables
    X_full = df_prep.drop(columns=['race', 'encounter_id', 'patient_nbr'], errors='ignore').copy()

    if exclude_vars:
        for var in exclude_vars:
            to_drop = [col for col in X_full.columns if col.startswith(f"{var}_")]
            logger.info(f"Dropping {len(to_drop)} columns from '{var}' for imputation.")
            X_full.drop(columns=to_drop, inplace=True)

    X_full = X_full.select_dtypes(include=[np.number])
    X_full['race_code'] = race_full

    # 3. Simulate missing values for evaluation
    logger.info("Simulating missing values for evaluation...")
    eval_indices = X_full[known_mask].sample(frac=0.3, random_state=42).index
    true_values = X_full.loc[eval_indices, 'race_code'].astype(int)
    X_full.loc[eval_indices, 'race_code'] = pd.NA

    # 4. Imputation with progress
    imputer = ProgressIterativeImputer(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',  # 💡 This will balance automatically the minority classes
            random_state=42
        ),
        max_iter=10,
        random_state=42
    )
    X_imputed = imputer.fit_transform(X_full)

    # 5. Decode results
    imputed_codes = X_imputed[:, -1].round().astype(int)
    imputed_labels = le.inverse_transform(imputed_codes)
    X_full['race_imputed'] = imputed_labels

    predicted_eval = X_full.loc[eval_indices, 'race_imputed']
    true_eval = le.inverse_transform(true_values)
    logger.info("Race imputation evaluation (simulated missing data):")
    logger.info("\n" + classification_report(true_eval, predicted_eval))

    # 6. Merge encoded race
    race_onehot = pd.get_dummies(X_full['race_imputed'], prefix='race', dtype=int)
    df_final = pd.concat([df_prep.drop(columns=['race']), race_onehot], axis=1)

    logger.info("Race imputation complete. One-hot encoded race added to dataset.")
    return df_final

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

    with tqdm(total=4, desc="Preprocessing", unit="step") as pbar:
        # Step 1: Load data
        df = load_data(path=RAW_DATA_PATH)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        pbar.update(1)

        # Step 2: Feature engineering
        df_encoded = feature_engineering(df, MISSING_VALUES, DROP_COLUMNS, ONE_HOT_COLUMNS, ORDINAL_MAPPINGS, TREATMENT_COLUMNS, TREATMENT_MAPPING, MID_PROCESSING_PATH)
        logger.info(f"After feature engineering: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns.")
        pbar.update(1)

        # Step 3: Impute missing race
        df_clean = impute_missing_race(df_encoded, exclude_vars=['discharge_disposition_id', 'admission_source_id'])
        logger.info(f"After imputation: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")
        pbar.update(1)

        # Step 4: Save to CSV
        df_clean.to_csv(NO_MISSINGS_PATH, index=False)
        logger.info(f"Data saved to {NO_MISSINGS_PATH}")
        pbar.update(1)

    logger.info("=== PREPROCESSING COMPLETE ===")
