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

class ProgressIterativeImputer(IterativeImputer):
    """ Extends the IterativeImputer to add a progress bar using tqdm. *** NOT WORKING *** """
    def _fit(self, X_filled, mask_missing_values, complete_mask):
        for iteration in tqdm(range(self.max_iter), desc="MICE Imputation Progress", unit="iter"):
            super()._fit(X_filled, mask_missing_values, complete_mask)
        return self

def impute_missing_race(
    logger: logging.Logger,
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

    Example
    --------
    # Step 3: Impute missing race
    >>>> df_clean = impute_missing_race(df_encoded, exclude_vars=['discharge_disposition_id', 'admission_source_id'])
    >>>>    logger.info(f"After imputation: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")
    >>>>    pbar.update(1)
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