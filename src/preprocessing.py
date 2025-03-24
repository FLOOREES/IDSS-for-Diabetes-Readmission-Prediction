import pandas as pd

# DATA LOADING
df = pd.read_csv('data/diabetic_data.csv')

# FEATURE ENGINEERING
def feature_engineering(df, encode, discard, one_hot, ordinal):
    df = df.replace(encode)
    df = df.drop(discard, axis=1)
    df = pd.get_dummies(df, columns=one_hot, dtype=int)
    for col, mapping in ordinal.items():
        df[col] = df[col].map(mapping)
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

# OUTLIER DETECTION

# NORMALIZATION

# SAVING
df.to_csv('data/diabetic_data_preprocessed.csv', index=False)