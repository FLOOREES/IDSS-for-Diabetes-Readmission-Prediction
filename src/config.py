import pandas as pd

RAW_DATA_PATH = "data/diabetic_data.csv"
MID_PROCESSING_PATH = "data/diabetic_data_mid.csv"
NO_MISSINGS_PATH="data/diabetic_data_no_na_diag.csv"

MISSING_VALUES = {'?': pd.NA}

DROP_COLUMNS = [
    'weight',
    'payer_code', 'medical_specialty',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
]

ONE_HOT_COLUMNS = [
    'gender', 'admission_type_id', 
]

ORDINAL_MAPPINGS = {
    'age': {
        '[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4,
        '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8,
        '[80-90)': 9, '[90-100)': 10
    },
    'readmitted': {
        'NO': 0, '>30': 1, '<30': 2
    }
}

TREATMENT_MAPPING = {
    'No': 0,
    'Down': 1,
    'Steady': 2,
    'Up': 3
}

TREATMENT_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
    'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
    'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]