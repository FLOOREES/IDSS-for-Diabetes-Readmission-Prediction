import pandas as pd

# DATA LOADING
df = pd.read_csv('data/diabetic_data.csv')

# FEATURE ENGINEERING
def feature_engineering(df, encode, discard, one_hot, ordinal):
    df = df.replace(encode) # Assignar missings

    df = df.drop(discard, axis=1) # Descar variables

    df = pd.get_dummies(df, columns=one_hot, dtype=int) # One-hot

    for col, mapping in ordinal.items(): # Mapejar (ordinals)
        df[col] = df[col].map(mapping)
    
    df.drop(df.columns[df.nunique() < 2], axis=1) # Descartar nominals amb 1 sol valor

    # Generalitzar raça
    # Com per un usuari podia haver-hi diferents ho he fet per l'última visita i
    # si no la següent
    # Tot i així encada queden molts missings de raça que podem mirar d'imputar
    # Això no se si hauria d'anar a imputació i a més és molt específic de race
    # i no sabria com posar-ho més abstracte
    race_cols = [col for col in df.columns if col.startswith('race')]
    df.loc[df[race_cols].sum(1) == 0, race_cols] = pd.NA

    df = df.sort_values('encounter_id')
    df[df.isna()] = df.groupby('patient_nbr')[race_cols].ffill()
    df[df.isna()] = df.groupby('patient_nbr')[race_cols].bfill()

    return df

encode = {
    '?': pd.NA,
}

petar_vars = [
    'diag_1',
    'diag_2',
    'diag_3',
    'weight',
    'payer_code',
    'medical_specialty',
    'max_glu_serum',
    'A1Cresult',
    'change',
    'diabetesMed'
]

one_hot_vars = [
    'race',
    'gender',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'metformin',
    'repaglinide',
    'nateglinide',
    'chlorpropamide',
    'glimepiride',
    'acetohexamide',
    'glipizide',
    'glyburide',
    'tolbutamide',
    'pioglitazone',
    'rosiglitazone',
    'acarbose',
    'miglitol',
    'troglitazone',
    'tolazamide',
    'examide',
    'citoglipton',
    'insulin',
    'glyburide-metformin',
    'glipizide-metformin',
    'glimepiride-pioglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

ordinal_vars = {
    'age': {
        '[0-10)': 1,
        '[10-20)': 2,
        '[20-30)': 3,
        '[30-40)': 4,
        '[40-50)': 5,
        '[50-60)': 6,
        '[60-70)': 7,
        '[70-80)': 8,
        '[80-90)': 9,
        '[90-100)': 10
    },
    'readmitted' : {
        'NO': 0,
        '>30': 1,
        '<30': 2
    }
}

df = feature_engineering(df, encode, petar_vars, one_hot_vars, ordinal_vars)

# MISSING IMPUTATION
def missing_imputation(): # Específica per race, potser podem fer abstracció
    pass

# OUTLIER DETECTION

# NORMALIZATION

# SAVING
df.to_csv('data/diabetic_data_preprocessed.csv', index=False)