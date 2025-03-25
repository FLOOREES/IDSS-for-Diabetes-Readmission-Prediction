import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List
from config import (
    RAW_DATA_PATH,
    NO_MISSINGS_PATH,
    MISSING_VALUES,
    DROP_COLUMNS,
    ONE_HOT_COLUMNS,
    ORDINAL_MAPPINGS
)
import logging
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ======================== LOGGER CONFIG ========================

logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG para más detalle
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Muestra en consola
        logging.FileHandler("preprocessing.log")  # Guarda en archivo
    ]
)

logger = logging.getLogger(__name__)

# ======================== FUNCTIONS ========================

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def feature_engineering(
    df: pd.DataFrame,
    encode: Dict[str, str],
    discard: List[str],
    one_hot: List[str],
    ordinal: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    logger.info("Starting feature engineering...")

    df = df.replace(encode)
    logger.info("Missing values replaced.")

    df = df.drop(discard, axis=1)
    logger.info(f"Dropped {len(discard)} columns.")

    df = pd.get_dummies(df, columns=one_hot, dtype=int)
    logger.info(f"Applied one-hot encoding to {len(one_hot)} columns.")

    for col, mapping in ordinal.items():
        df[col] = df[col].map(mapping)
    logger.info(f"Mapped {len(ordinal)} ordinal columns.")

    dropped_cols = df.columns[df.nunique() < 2]
    df = df.drop(dropped_cols, axis=1)
    logger.info(f"Dropped {len(dropped_cols)} low-variance columns.")

    # Handle missing race (one-hot) via patient visits
    race_cols = [col for col in df.columns if col.startswith('race')]
    missing_race_count = (df[race_cols].sum(axis=1) == 0).sum()
    df.loc[df[race_cols].sum(axis=1) == 0, race_cols] = pd.NA

    df = df.sort_values('encounter_id')
    df[race_cols] = df.groupby('patient_nbr')[race_cols].ffill()
    df[race_cols] = df.groupby('patient_nbr')[race_cols].bfill()
    logger.info(f"Filled {missing_race_count} missing race rows using ffill/bfill.")

    return df


def impute_missing_race(df_raw: pd.DataFrame, df_prep: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting race imputation using MICE + RandomForestClassifier")

    # Identificar columnas one-hot de 'race'
    race_cols = [col for col in df_prep.columns if col.startswith('race')]

    # Crear versión con codificación numérica de la raza
    race_series = df_raw['race'].replace('?', np.nan)
    known_mask = race_series.notna()

    # Codificar la raza para el imputador (LabelEncoder)
    le = LabelEncoder()
    race_encoded = le.fit_transform(race_series.dropna())
    race_full = pd.Series(data=np.nan, index=race_series.index)
    race_full[known_mask] = race_encoded

    # Combinar con features para imputación
    X_full = df_prep.drop(columns=race_cols).copy()
    X_full['race_code'] = race_full

    # Simulación para evaluación: ocultamos parte de los datos conocidos
    logger.info("Simulating missing values for evaluation...")
    evaluation_mask = (df_raw['race'] != '?')
    eval_indices = X_full[evaluation_mask].sample(frac=0.3, random_state=42).index
    true_values = X_full.loc[eval_indices, 'race_code'].astype(int)
    X_full.loc[eval_indices, 'race_code'] = np.nan

    # Imputar usando MICE + Random Forest
    imputer = IterativeImputer(estimator=RandomForestClassifier(), max_iter=10, random_state=0)
    X_imputed = imputer.fit_transform(X_full)

    # Recuperar columna imputada
    imputed_race_codes = X_imputed[:, -1].round().astype(int)
    imputed_race_names = le.inverse_transform(imputed_race_codes)
    X_full['race_imputed'] = imputed_race_names

    # Evaluación
    if len(eval_indices) > 0:
        predicted_eval = X_full.loc[eval_indices, 'race_imputed']
        true_eval = le.inverse_transform(true_values)
        logger.info("Race imputation evaluation (simulated missing data):")
        logger.info("\n" + classification_report(true_eval, predicted_eval))

    # Reconvertir a one-hot
    imputed_one_hot = pd.get_dummies(X_full['race_imputed'], prefix='race', dtype=int)

    # Asegurar que todas las columnas estén
    for col in race_cols:
        if col not in imputed_one_hot.columns:
            imputed_one_hot[col] = 0
    imputed_one_hot = imputed_one_hot[race_cols]

    # Insertar en el dataset original
    df_prep.loc[:, race_cols] = imputed_one_hot.values
    logger.info("Race imputation (MICE + RF) complete.")

    return df_prep

# ======================== MAIN ========================

if __name__ == "__main__":
    logger.info("=== PREPROCESSING STARTED ===")

    with tqdm(total=4, desc="Preprocessing", unit="step") as pbar:
        # Step 1: Load data
        df = load_data(path=RAW_DATA_PATH)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        pbar.update(1)

        # Step 2: Feature engineering
        df_encoded = feature_engineering(df, MISSING_VALUES, DROP_COLUMNS, ONE_HOT_COLUMNS, ORDINAL_MAPPINGS)
        logger.info(f"After feature engineering: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns.")
        pbar.update(1)

        # Step 3: Impute missing race
        df_clean = impute_missing_race(df, df_encoded)
        logger.info(f"After imputation: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")
        pbar.update(1)

        # Step 4: Save to CSV
        df_clean.to_csv(NO_MISSINGS_PATH, index=False)
        logger.info(f"Data saved to {NO_MISSINGS_PATH}")
        pbar.update(1)

    logger.info("=== PREPROCESSING COMPLETE ===")
