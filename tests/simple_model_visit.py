import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

database = pd.read_csv('./data/diabetic_data_no_na_diag_cod.csv')
#Keep only one row per patient
df = database.drop_duplicates(subset=['patient_nbr'])

# Map the diag variables from the diag_embredding.npy to the diag variables in the database
embeddings = np.load('./data/diag_embeddings.npy')

# Create a DataFrame for the embeddings with an index corresponding to the row numbers
def get_embedding(idx):
    try:
        idx = int(idx)
        if pd.isna(idx) or idx < 0 or idx >= len(embeddings):
            return np.zeros(8)  # Rellenar con ceros si es inválido
        else:
            return embeddings[idx]
    except:
        return np.zeros(8)
# print(df.shape, df.columns)
# Para cada columna diag, crear 8 columnas de embedding
for col in ["diag_1", "diag_2", "diag_3"]:
    # Obtener embeddings para esta columna
    emb_data = df[col].apply(get_embedding).tolist()
    
    # Crear DataFrame temporal con las 8 columnas
    emb_df = pd.DataFrame(
        emb_data,
        columns=[f"{col}_emb_{i}" for i in range(8)]
    )
    
    # Concatenar al DataFrame original
    emb_df.index = df.index

    df = pd.concat([df, emb_df], axis=1)
# print(df.shape, df.columns)
# Eliminar columnas originales diag_1, diag_2, diag_3
train = df.drop(columns=["diag_1", "diag_2", "diag_3","patient_nbr","encounter_id","readmitted"])

X_train, X_test, y_train, y_test = train_test_split(train, df['readmitted'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

###################### SIMPLE MODEL ######################
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = lgb.LGBMClassifier()

import optuna
from sklearn.metrics import log_loss
from lightgbm import early_stopping
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 3,  # Número de clases
        'metric': 'multi_logloss',  # Coherente con la pérdida que optimiza LightGBM
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': 1000,  # Fijar alto y usar early stopping
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'is_unbalance': True  # Si hay desequilibrio de clases
    }

    model = lgb.LGBMClassifier(**params)
    
    # Usar early stopping para evitar overfitting
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50)],
    )
    
    # Predecir probabilidades (mejor que accuracy para problemas desbalanceados)
    y_pred_proba = model.predict_proba(X_val)
    loss = log_loss(y_val, y_pred_proba)
    
    return loss  # Minimizar log_loss (mejor que accuracy para métricas probabilísticas)

# Crear estudio de Optuna
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner= MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,))
    
# study.optimize(objective, n_trials=50)

# best_model = lgb.LGBMClassifier(**study.best_params)
params = {'boosting_type': 'gbdt', 'num_leaves': 75, 'learning_rate': 0.010154597718993866, 'max_depth': 10, 'min_child_samples': 33, 'reg_alpha': 7.744816012023702e-09, 'reg_lambda': 0.1767992298499557, 'subsample': 0.7764938427583004, 'colsample_bytree': 0.6675982909322207}
best_model = lgb.LGBMClassifier(**params)
best_model.fit(X_train, y_train)
lgb.plot_importance(best_model, figsize=(12, 8))

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
# Predecir
y_pred_rf = rf.predict(X_val)

# Calcular la precisión
accuracy_rf = accuracy_score(y_val, y_pred_rf)
print("Accuracy Random Forest:", accuracy_rf)

# Evaluar
print("Accuracy:", best_model.score(X_val, y_val))