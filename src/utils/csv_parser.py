# Continuing in csv_validator.py
import pandas as pd
import os
# E.g., in a new file named csv_validator.py

class CSVValidationError(Exception):
    """Custom exception for CSV validation errors."""
    def __init__(self, errors, message="CSV validation failed"):
        self.errors = errors # List of error messages
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        error_details = "\n- ".join(self.errors)
        return f"{self.message}:\n- {error_details}"

# Define Your Expected CSV Schema (This could also be passed as an argument for more flexibility)
EXPECTED_COLUMNS_INFO = {
    # Column Name: { 'dtype': expected_pandas_dtype, 'required': True/False, 'allowed_values': [list] or None, 'range': (min, max) or None }
    'encounter_id': {'dtype': 'int64', 'required': True, 'range': (0, None)},
    'patient_nbr': {'dtype': 'int64', 'required': True, 'range': (0, None)},
    'race': {'dtype': 'object', 'required': True, 'allowed_values': ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', '?']},
    'gender': {'dtype': 'object', 'required': True, 'allowed_values': ['Male', 'Female', 'Unknown/Invalid']},
    'age': {'dtype': 'object', 'required': True, 'allowed_values': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']},
    'time_in_hospital': {'dtype': 'int64', 'required': True, 'range': (1, 1000)},
    'num_lab_procedures': {'dtype': 'int64', 'required': True, 'range': (0, 100)},
    'num_procedures': {'dtype': 'int64', 'required': True, 'range': (0, 100)},
    'num_medications': {'dtype': 'int64', 'required': True, 'range': (0, 100)},
    'number_outpatient': {'dtype': 'int64', 'required': True, 'range': (0, 100)},
    'number_emergency': {'dtype': 'int64', 'required': True, 'range': (0, 100)},
    'number_inpatient': {'dtype': 'int64', 'required': True, 'range': (0, 100)},
    'diag_1': {'dtype': 'object', 'required': False, 'allow_empty_string': True}, # Allow empty strings explicitly
    'diag_2': {'dtype': 'object', 'required': False, 'allow_empty_string': True},
    'diag_3': {'dtype': 'object', 'required': False, 'allow_empty_string': True},
    'number_diagnoses': {'dtype': 'int64', 'required': True, 'range': (0, None)},
    'metformin': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'insulin': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'repaglinide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'nateglinide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'chlorpropamide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'glimepiride': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'acetohexamide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'glipizide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'glyburide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'tolbutamide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'pioglitazone': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'rosiglitazone': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'acarbose': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'miglitol': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'troglitazone': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'tolazamide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'examide': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'citoglipton': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'glyburide-metformin': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'glipizide-metformin': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'glimepiride-pioglitazone': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'metformin-rosiglitazone': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'metformin-pioglitazone': {'dtype': 'object', 'required': True, 'allowed_values': ['No', 'Steady', 'Up', 'Down']},
    'readmitted': {'dtype': 'object', 'required': True, 'allow_empty_string': True,'allowed_values': ['NO', '>30', '<30', '']},
    'change': {'dtype': 'object', 'required': True, 'allow_empty_string': True, 'allowed_values': ['No', 'Ch', 'Up', 'Down','']},
    'diabetesMed': {'dtype': 'object', 'required': True, 'allow_empty_string': True, 'allowed_values': ['No', 'Yes','']},
    # Optional column
    'weight': {'dtype': 'object', 'required': False, 'allow_empty_string': True},
       'payer_code': {'dtype': 'object', 'required': False, 'allow_empty_string': True},
       'medical_specialty': {'dtype': 'object', 'required': False, 'allow_empty_string': True},
       'max_glu_serum': {'dtype': 'object', 'required': False, 'allow_empty_string': True},
       'A1Cresult': {'dtype': 'object', 'required': False, 'allow_empty_string': True},
       'admission_type_id': {
          'dtype': 'int64',
          'required': True,
          'range': (1, 9),
          'allowed_values': [1, 2, 3, 4, 5, 6, 7, 8, 9]
       },
       'discharge_disposition_id': {
          'dtype': 'int64',
          'required': True,
          'allowed_values': [
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             22, 23, 24, 25, 27, 28
          ]
       },
       'admission_source_id': {
          'dtype': 'int64',
          'required': True,
          'allowed_values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 20, 22, 25]
       },

    # change,diabetesMed,readmitted
    # ... (add ALL your other columns and their rules here) ...
    # Example for a column that could be truly optional (not even a '?' or empty string needed)
    # 'optional_notes': {'dtype': 'object', 'required': False},
}

def validate_csv_data(csv_filepath, schema=EXPECTED_COLUMNS_INFO):
    """
    Validates the structure and content of a CSV file against a defined schema.

    Args:
        csv_filepath (str): The path to the CSV file.
        schema (dict): A dictionary defining the expected column properties.
                       Defaults to EXPECTED_COLUMNS_INFO.

    Returns:
        pandas.DataFrame: The validated DataFrame if all checks pass.

    Raises:
        FileNotFoundError: If the csv_filepath does not exist.
        CSVValidationError: If any validation checks fail, containing a list of errors.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV cannot be parsed.
        Exception: For other unexpected errors during file reading.
    """
    validation_errors = []

    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSV file not found at path: {csv_filepath}")

    try:
        df = pd.read_csv(csv_filepath)
    except pd.errors.EmptyDataError:
        raise # Re-raise the specific pandas error
    except pd.errors.ParserError:
        raise # Re-raise
    except Exception as e:
        # Catch other potential read errors and wrap them
        raise Exception(f"An unexpected error occurred while reading the CSV: {str(e)}")


    # --- 1. Validate Column Presence and Extra Columns ---
    actual_columns = df.columns.tolist()
    missing_required_columns = []
    expected_col_names = set(schema.keys())

    for col_name, col_info in schema.items():
        if col_info.get('required', False) and col_name not in actual_columns:
            missing_required_columns.append(col_name)
    
    if missing_required_columns:
        validation_errors.append(f"Columnas requeridas faltantes: {', '.join(missing_required_columns)}")

    extra_columns = [col for col in actual_columns if col not in expected_col_names]
    if extra_columns:
        validation_errors.append(f"Columnas extra no esperadas encontradas: {', '.join(extra_columns)}. Por favor, remuévelas o verifica el esquema.")

    # If fundamental column structure is wrong, it might not make sense to proceed with content validation
    if validation_errors:
        raise CSVValidationError(validation_errors, "Error de estructura de columnas del CSV")

    # --- 2. Validate Data Types, Values, and Constraints ---
    for col_name, col_info in schema.items():
        if col_name not in df.columns:
            # This case is for optional columns not present in the CSV.
            # If it was required and missing, it would have been caught above.
            continue

        expected_dtype_str = col_info.get('dtype')
        series_to_validate = df[col_name].copy() # Work on a copy for modifications

        # --- 2a. Handle Missing Values Based on Requirement ---
        # If a column is required, it should not have NaNs after initial read.
        # Some '?' or empty strings might be acceptable as per 'allowed_values' or 'allow_empty_string'.
        if col_info.get('required', False) and series_to_validate.isnull().any():
            # If it's required AND has nulls, and nulls aren't explicitly allowed (e.g. via '?' in allowed_values)
            # This check is a bit broad, refined by type/value checks later
            is_nullable_by_value = '?' in col_info.get('allowed_values', []) or "" in col_info.get('allowed_values', [])
            if not is_nullable_by_value:
                 first_na_row_idx = series_to_validate[series_to_validate.isnull()].index.min()
                 validation_errors.append(f"Columna requerida '{col_name}' tiene valores nulos/faltantes (ej. en fila ~{first_na_row_idx+2}).")


        # --- 2b. Data Type Conversion and Validation ---
        if expected_dtype_str:
            original_series_for_error_reporting = df[col_name].copy() # Before type coercion
            try:
                if expected_dtype_str == 'int64':
                    # Coerce non-numeric to NaN, then check if any original non-NaN became NaN
                    converted_series = pd.to_numeric(series_to_validate, errors='coerce')
                    problematic = converted_series.isnull() & series_to_validate.notnull()
                    if problematic.any():
                        first_bad_idx = problematic.idxmax()
                        bad_val = original_series_for_error_reporting.loc[first_bad_idx]
                        validation_errors.append(f"Columna '{col_name}' (valor '{bad_val}' en fila ~{first_bad_idx+2}) no pudo ser convertida a entero.")
                    series_to_validate = converted_series
                elif expected_dtype_str == 'float64':
                    converted_series = pd.to_numeric(series_to_validate, errors='coerce')
                    problematic = converted_series.isnull() & series_to_validate.notnull()
                    if problematic.any():
                        first_bad_idx = problematic.idxmax()
                        bad_val = original_series_for_error_reporting.loc[first_bad_idx]
                        validation_errors.append(f"Columna '{col_name}' (valor '{bad_val}' en fila ~{first_bad_idx+2}) no pudo ser convertida a numérico decimal.")
                    series_to_validate = converted_series
                elif expected_dtype_str == 'object':
                    # If allow_empty_string is False (default), then empty strings become NaN if not in allowed_values.
                    # If it's an object type, ensure strings. Handle explicit empty strings if needed.
                    series_to_validate = series_to_validate.astype(str).replace('nan', pd.NA) # Treat 'nan' strings as NA
                    if not col_info.get('allow_empty_string', False):
                        empty_strings = (series_to_validate == "")
                        if empty_strings.any():
                            # Only an error if "" is not in allowed_values and it's required
                            is_empty_allowed = "" in col_info.get('allowed_values', [])
                            if col_info.get('required', False) and not is_empty_allowed:
                                first_empty_idx = empty_strings.idxmax()
                                validation_errors.append(f"Columna '{col_name}' (fila ~{first_empty_idx+2}) es una cadena vacía pero se requiere un valor o no está permitido.")
                elif expected_dtype_str == 'bool':
                    # Example: map specific strings to bool, others are errors
                    bool_map = {'True': True, 'False': False, '1': True, '0': False, 'yes': True, 'no': False}
                    # This is a simple example, robust bool conversion can be tricky
                    if series_to_validate.notnull().any():
                        try:
                            series_to_validate = series_to_validate.dropna().map(bool_map) # DropNA before map
                        except Exception: # Or more specific error if map fails
                             validation_errors.append(f"Columna '{col_name}' no pudo ser convertida a booleano usando el mapeo definido.")
                # Add other specific dtype conversions (e.g., datetime with pd.to_datetime)
            except Exception as e:
                validation_errors.append(f"Error inesperado al procesar tipo para columna '{col_name}': {str(e)}")
                continue # Skip further value checks if type processing failed

        # Update DataFrame column with potentially coerced/cleaned series for subsequent checks
        df[col_name] = series_to_validate

        # --- 2c. Allowed Values Check (after type coercion and NaN handling) ---
        allowed_vals_list = col_info.get('allowed_values')
        if allowed_vals_list and series_to_validate.notna().any():
            # Ensure comparison is done with consistent types
            # If expected dtype is object (string), convert allowed_values to string for comparison
            if expected_dtype_str == 'object':
                allowed_set = set(map(str, allowed_vals_list))
                series_for_check = series_to_validate.dropna().astype(str)
            else: # For numeric, bool, etc.
                allowed_set = set(allowed_vals_list)
                series_for_check = series_to_validate.dropna()

            invalid_entries = series_for_check[~series_for_check.isin(allowed_set)]
            if not invalid_entries.empty:
                first_invalid_val = invalid_entries.iloc[0]
                first_invalid_idx = invalid_entries.index[0]
                validation_errors.append(f"Columna '{col_name}' (valor '{first_invalid_val}' en fila ~{first_invalid_idx+2}) no está en la lista de valores permitidos: {allowed_vals_list}")

        # --- 2d. Range Check (for numeric columns, after type coercion) ---
        value_range = col_info.get('range')
        if value_range and series_to_validate.notna().any() and pd.api.types.is_numeric_dtype(series_to_validate.dtype):
            min_val, max_val = value_range
            out_of_range_condition = pd.Series(False, index=series_to_validate.index) # Start with all false

            if min_val is not None:
                out_of_range_condition |= (series_to_validate < min_val)
            if max_val is not None:
                out_of_range_condition |= (series_to_validate > max_val)
            
            # Apply only to non-NA values
            final_out_of_range = series_to_validate[out_of_range_condition & series_to_validate.notna()]

            if not final_out_of_range.empty:
                first_oor_val = final_out_of_range.iloc[0]
                first_oor_idx = final_out_of_range.index[0]
                range_str = f"(esperado: {min_val if min_val is not None else '-inf'} a {max_val if max_val is not None else 'inf'})"
                validation_errors.append(f"Columna '{col_name}' (valor '{first_oor_val}' en fila ~{first_oor_idx+2}) está fuera del rango permitido {range_str}.")

    if validation_errors:
        raise CSVValidationError(validation_errors, "Errores de validación de contenido del CSV")

    return df # Return the validated (and potentially type-coerced) DataFrame