import pandas as pd

def preprocess(df, discard, one_hot, ordinal):
    df = df.drop(discard, axis=1)
    df = pd.get_dummies(df, columns=one_hot)
    

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
    },
    'readmitted' # target
}

