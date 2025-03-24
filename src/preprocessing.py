import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
df_one_hot = feature_engineering(df, encode, petar_vars, one_hot_vars, ordinal_vars)


# MISSING IMPUTATION
def missing_imputation(df_raw, df_prep): # Específica per race, potser podem fer abstracció
    '''
    Funció per imputar els missings de la única columna que té missings (race)
    '''
    race_cols = [col for col in df_prep.columns if col.startswith('race')]
    norace = df_prep.drop(race_cols, axis=1)
    train = norace.loc[df_raw.race != '?']
    test = norace.loc[df_raw.race == '?']
    y_train = df_raw.loc[df_raw.race != '?', 'race']

    model = RandomForestClassifier()
    model.fit(train, y_train)  # Usar variables relevantes

    # Predecir missings
    missings = model.predict(test)    
    imputed_one_hot = pd.get_dummies(missings, prefix='race', dtype=int)

    # Añadir las columas sin representacion en el test
    for col in race_cols:
        if col not in imputed_one_hot.columns:
            imputed_one_hot[col] = 0

    # Reordenar las columnas
    imputed_one_hot = imputed_one_hot[race_cols]

    # Assignar valores al dataset original
    df_prep.loc[df_raw.race == '?', race_cols] = imputed_one_hot.values

    return df_prep

df_no_na = missing_imputation(df, df_one_hot)
# OUTLIER DETECTION

# NORMALIZATION

# SAVING (uncomment to save)

# df.to_csv('data/diabetic_data_preprocessed.csv', index=False)
df_no_na.to_csv('data/diabetic_data_no_na.csv', index=False)
