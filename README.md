# AI-Powered Clinical Decision Support for Diabetes Readmission

## Project Overview

This project is an end-to-end clinical decision support system designed to predict the risk of hospital readmission for diabetes patients. It combines a sequential deep learning model with advanced AI techniques to not only provide an accurate prediction but also a detailed, evidence-based explanation in natural language, making the model's reasoning transparent and useful for clinical professionals. The system is deployed as a Flask web application with a user-friendly interface.

## Core Features

* **Sequential Prediction Model:** A Recurrent Neural Network (RNN) built with PyTorch (configurable as LSTM/GRU) that processes patient visit history as sequences to predict readmission risk across three categories: No readmission, readmission in over 30 days, and readmission in under 30 days.
* **Explainable AI (XAI):** Utilizes SHAP (`DeepExplainer`) to interpret the PyTorch model's predictions, identifying the most influential clinical factors (e.g., number of inpatient visits, specific diagnoses, medication changes) driving the risk assessment for a given patient.
* **Retrieval-Augmented Generation (RAG):** Implements a sophisticated agent that enriches the model's explanation. It uses a vector store (ChromaDB) of medical documents, LangChain, and the Google Gemini LLM to retrieve relevant clinical context and synthesize it with the SHAP findings.
* **End-to-End MLOps Pipeline:** A fully modular and configurable pipeline manages the entire workflow from data preprocessing and feature engineering to model training, evaluation, and artifact management (e.g., scalers, encoders, trained models).
* **Interactive Web Application:** A Flask-based web application provides two intuitive modes for data input: a dynamic questionnaire for manual entry and a CSV upload feature for batch processing. Results, including the LLM-generated report and SHAP plots, are rendered directly in the browser.

## System Workflow

1.  **Input:** A healthcare professional inputs a patient's clinical history through the Flask UI, either by filling out a detailed form or by uploading a CSV file with one or more patient encounters.
2.  **Prediction Engine:** The `SinglePatientPredictorEngine` is triggered. It loads all necessary preprocessing artifacts and orchestrates the inference process.
    * **Data Preprocessing:** The raw patient data undergoes a multi-phase, stateful preprocessing pipeline to handle missing values, perform ordinal/one-hot/label encoding, and scale numerical features. Diagnosis codes (ICD-9) are mapped to pre-trained embeddings.
    * **Sequence Modeling:** The processed data is transformed into sequences of patient visits, which are fed into the trained PyTorch RNN model to generate a readmission risk prediction.
3.  **Explanation Generation (`DiabetesAgent`):**
    * **XAI Analysis:** SHAP is used to calculate the feature importance values for the model's specific prediction, highlighting the key drivers.
    * **Context Retrieval (RAG):** A query is dynamically constructed from the patient's data, the model's prediction, and the top SHAP factors. This query is used to retrieve the most relevant text snippets from a knowledge base of medical documents vectorized with HuggingFace sentence transformers.
    * **LLM Synthesis:** The Google Gemini LLM receives a comprehensive prompt containing the patient summary, prediction probabilities, SHAP analysis, and the retrieved medical context. It synthesizes this information into a structured, evidence-based "Model Prediction Explainer Report."
4.  **Output:** The Flask application presents the final report and the SHAP feature importance visualization to the user, providing a clear, contextualized, and explainable prediction.

## Key Technologies & Libraries

* **Backend & Deployment:** Python, Flask
* **Machine Learning & DL:** PyTorch, Scikit-learn, Pandas, NumPy, SHAP
* **LLM & RAG Framework:** Google Gemini, LangChain, ChromaDB (Vector Store), HuggingFace Transformers
* **MLOps & Tooling:**
    * Stateful preprocessing pipeline with saved artifacts (scalers, encoders).
    * Modular project structure (`preprocessing`, `modeling`, `training`, `inference`).
    * Command-line interface (`run.py`) for executing and configuring the training pipeline.
* **Web Frontend:** HTML, JavaScript (for dynamic elements like ICD-9 search).

## Requirements & Dependencies

The project relies on a set of Python libraries and external services.

* **Python Libraries:** All required Python packages are listed in the `requirements.txt` file and can be installed using pip:
    ```bash
    pip install -r requirements.txt
    ```
* **Language Models:** The project uses a specific SpaCy model for scientific text processing to generate diagnosis embeddings. It must be downloaded separately:
    ```bash
    python -m spacy download en_core_sci_md
    ```
* **API Keys:** The Retrieval-Augmented Generation (RAG) feature requires access to the Google Gemini API. You must provide a valid API key in a `.env` file at the project's root directory:
    ```
    # .env
    GOOGLE_API_KEY="your_api_key_here"
    ```

## Execution

The project is modular, allowing for different components to be run independently.

### 1. Running the Web Application

The primary interface is the Flask web application. It can be started by running the `app.py` file from the project's root directory.

```bash
# From the root of the project
python app.py
```
The application will be accessible at `http://127.0.0.1:5000` by default.

### 2. Executing the MLOps Pipeline

The entire machine learning pipeline (data preprocessing, model training, and evaluation) is managed by `src/run.py`. This script acts as a command-line interface to orchestrate the workflow defined in `src/pipeline.py`.

* **Run with default configuration from `src/config.py`:**
    ```bash
    python -m src.run
    ```
* **Force retraining of the autoencoder and predictor models:**
    ```bash
    python -m src.run --train-ae --train-predictor
    ```
* **Override hyperparameters from the command line:**
    ```bash
    python -m src.run --predictor-epochs 50 --predictor-learning-rate 0.0001 --hidden-dim 256
    ```

### 3. Standalone Analysis & Utility Scripts

Specific tasks can be executed by running individual scripts. This is useful for debugging, testing, or generating specific artifacts.

* **Generate SHAP explanations for a single patient:**
    ```bash
    python -m src.shap.test_shap path/to/patient_data.csv
    ```
* **Create sample data files for testing or analysis:**
    ```bash
    # Create background data for SHAP from the training set
    python -m src.utils.create_train_samples

    # Create individual raw CSV files for verified test patients
    python -m src.utils.create_test_samples
    ```