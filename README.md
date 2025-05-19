# IDSS-for-Diabetes-Readmission-Prediction

## Data Preprocessing

The preprocessing script prepares the raw dataset for modeling by applying a series of cleaning, transformation, and imputation steps. The script is located at `src/preprocessing.py` and is designed to be modular, maintainable, and production-ready.

### 🛠️ Main Steps

1. **Data Loading**  
   Loads the raw CSV dataset from the path specified in `config.py` (`RAW_DATA_PATH`).

2. **Feature Engineering**  
   - Replaces custom missing values (e.g. `"?"`) with proper nulls.
   - Drops irrelevant or high-missing-value columns.
   - Applies one-hot encoding to categorical variables.
   - Encodes ordinal features using defined mappings.
   - Removes low-variance features.
   - Attempts to fill missing race information by propagating values forward and backward within the same patient history.

3. **Missing Value Imputation (MICE + Random Forest)**  
   - Uses `IterativeImputer` from scikit-learn with a `RandomForestClassifier` to impute missing values in the `race` field.
   - A portion of known values is masked to simulate missingness and evaluate imputation quality.
   - Performance is measured using a classification report (accuracy, precision, recall, F1).
   - The imputed `race` column is restored in one-hot encoded format.

4. **Exporting Results**  
   The final cleaned dataset is saved to `NO_MISSINGS_PATH` as defined in `config.py`.

### 🚀 How to Run

Make sure your environment is activated and all required dependencies are installed.

```bash
python -m src.run
```

### 📋 Output
A progress bar (tqdm) indicates execution progress.

Logs are saved to preprocessing.log and printed to the console.

Final dataset is exported as a .csv file.

### 📁 Configuration
All configuration values (paths, feature lists, mappings) are defined in config.py for full control and reusability.
