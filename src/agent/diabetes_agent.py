import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import logging
import json # For loading ohe_feature_names if needed directly by engine parts
import sys
import logging
from typing import Dict, Any, Optional
import argparse

logger = logging.getLogger(__name__)

# --- Langchain & LLM Imports ---
LANGCHAIN_AVAILABLE = False
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain components imported successfully.")
except ImportError:
    logging.warning("LangChain or langchain_google_genai not found. RAG and LLM features will be limited.")
    # Define dummy classes if needed to prevent downstream attribute errors if an LLM object is expected
    class ChatGoogleGenerativeAI: pass
    class RetrievalQA: pass


# --- Project Imports ---
from src import config as AppConfig
from src.inference.predictor_engine import SinglePatientPredictorEngine
from src.shap.shap_explainer import generate_shap_explanation
from src.shap.shap_utils import prepare_shap_background_data
from src.preprocessing.first_phase import FirstPhasePreprocessor

try:
    from src.agent.embedding_maker import vectorstore_loader, vectorstore_maker
    EMBEDDING_MAKER_AVAILABLE = True
except ImportError:
    logging.warning("embedding_maker.py or its functions not found. RAG will be disabled.")
    EMBEDDING_MAKER_AVAILABLE = False
    if LANGCHAIN_AVAILABLE: # Define dummies only if LangChain itself loaded
        def vectorstore_loader(db_name_path: str): logging.error("vectorstore_loader (dummy) called."); return None
        def vectorstore_maker(db_name_path: str, doc_folder_path: str): logging.error("vectorstore_maker (dummy) called."); pass

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- RAG Configuration ---
DOC_FOLDER_PATH = getattr(AppConfig, 'DOC_FOLDER', "./src/agent/Diabetes_docs/")
DB_NAME_PATH = getattr(AppConfig, 'DB_NAME', "./src/agent/db_place")

# --- Environment Setup ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if LANGCHAIN_AVAILABLE and not google_api_key:
    logger.warning("GOOGLE_API_KEY not found in env. LLM/RAG features will be disabled.")
elif LANGCHAIN_AVAILABLE:
    os.environ["GOOGLE_API_KEY"] = google_api_key


class DiabetesAgent:
    def __init__(self, cfg=AppConfig):
        self.cfg = cfg
        self.engine: Optional[SinglePatientPredictorEngine] = None # Type hint for clarity
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.vectorstore: Optional[Any] = None # MODIFIED: Initialize instance attribute

        logger.info("Initializing DiabetesAgent...")
        try:
            self.engine = SinglePatientPredictorEngine(cfg=self.cfg)
            logger.info("SinglePatientPredictorEngine initialized successfully.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize SinglePatientPredictorEngine: {e}", exc_info=True)
            raise RuntimeError(f"DiabetesAgent failed to initialize predictor engine: {e}") from e

        if LANGCHAIN_AVAILABLE and google_api_key and EMBEDDING_MAKER_AVAILABLE:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=getattr(self.cfg, "LLM_MODEL_NAME", "gemini-1.0-pro"),
                    temperature=getattr(self.cfg, "LLM_TEMPERATURE", 0.2),
                    convert_system_message_to_human=True
                )
                logger.info(f"LLM initialized with model: {self.llm.model}")
                
                # Load or create the vectorstore
                loaded_vectorstore = self._vectorstore_import(DB_NAME_PATH, DOC_FOLDER_PATH) 
                
                if loaded_vectorstore:
                    self.vectorstore = loaded_vectorstore # <<< --- FIX: Assign to self.vectorstore ---
                    self.qa_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=self.vectorstore.as_retriever( # Now uses self.vectorstore
                            search_kwargs={"k": getattr(self.cfg, "RAG_NUM_DOCS_TO_RETRIEVE", 2)}
                        ),
                        return_source_documents=True 
                    )
                    logger.info("RAG QA chain initialized.")
                else:
                    logger.warning("Failed to initialize vectorstore. RAG QA chain will not be available.")
                    self.vectorstore = None # Ensure it's None if loading failed
            except Exception as e:
                logger.error(f"Error initializing LLM or RAG chain: {e}", exc_info=True)
                self.llm = None 
                self.qa_chain = None
                self.vectorstore = None # Ensure it's None on error
        else:
            logger.warning("LLM/RAG chain setup skipped due to missing API key, LangChain, or embedding_maker.")
            self.vectorstore = None # Ensure it's None if setup is skipped

    def _vectorstore_import(self, db_name_str: str, doc_folder_str: str):
        db_path = Path(db_name_str)
        doc_path = Path(doc_folder_str)
        
        if not db_path.exists():
            logger.info(f"Vectorstore not found at {db_path}. Attempting to create a new one.")
            if not doc_path.exists() or not doc_path.is_dir() or not any(doc_path.iterdir()):
                logger.error(f"Document folder '{doc_path}' for RAG does not exist, is not a directory, or is empty. Cannot create new vectorstore.")
                return None # Critical failure for RAG
            logger.info("Creating new vectorstore...")
            vectorstore_maker(str(db_path), str(doc_path)) 
            logger.info(f"New vectorstore created from documents in '{doc_path}'.")
        else:
            logger.info(f"Vectorstore found at {db_path}. Loading...")
        
        vs = vectorstore_loader(str(db_path)) # Let errors propagate if loading fails
        if vs: logger.info("Vectorstore loaded successfully.")
        else: logger.warning(f"vectorstore_loader returned None for path: {db_path}")
        return vs

    def _get_patient_history_summary_for_prompt(self, raw_patient_df: pd.DataFrame) -> str:
        """Creates a more detailed string summary of patient history for the LLM prompt."""
        if raw_patient_df.empty:
            return "No patient history provided for summarization."
        
        patient_id = raw_patient_df[self.cfg.PATIENT_ID_COL].iloc[0]
        num_visits_in_csv = len(raw_patient_df)
        
        summary_parts = [
            f"Patient ID: {patient_id}",
            f"Number of visits in provided history: {num_visits_in_csv}"
        ]

        # Helper to get unique, clean string values from a column
        def get_unique_col_values(df, col_name, max_items=5):
            if col_name in df.columns:
                valid_series = df[col_name].dropna()
                if not valid_series.empty:
                    unique_vals = valid_series.astype(str).unique().tolist()
                    if unique_vals:
                         # Filter out common non-informative placeholders if any remain after load_data
                        unique_vals = [val for val in unique_vals if val.lower() not in ['?', '', 'none', 'nan']]
                        if unique_vals:
                            return f"{col_name}: {', '.join(unique_vals[:max_items])}{'...' if len(unique_vals) > max_items else ''}"
            return None

        # Key variables from your list (add more as needed)
        key_demographics = ['race', 'gender', 'age']
        for col in key_demographics:
            val_str = get_unique_col_values(raw_patient_df, col)
            if val_str: summary_parts.append(f"- {val_str}")

        key_encounter_details = [
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'
        ]
        for col in key_encounter_details:
            # For numerical, maybe show range or list if few visits
            if col in raw_patient_df.columns and pd.api.types.is_numeric_dtype(raw_patient_df[col]):
                if num_visits_in_csv == 1:
                    summary_parts.append(f"- {col}: {raw_patient_df[col].iloc[0]}")
                else: # Multiple visits, show list for now
                    summary_parts.append(f"- {col} (all visits): {raw_patient_df[col].tolist()}")
            else: # Categorical or string
                val_str = get_unique_col_values(raw_patient_df, col)
                if val_str: summary_parts.append(f"- {val_str}")
        
        # Diagnoses (more careful handling)
        diag_cols = getattr(self.cfg, 'DIAG_COLS', ['diag_1', 'diag_2', 'diag_3'])
        all_diags_texts = []
        for col in diag_cols:
            if col in raw_patient_df.columns:
                # Get unique non-null/non-'?' diagnoses, ensuring string conversion
                # FirstPhasePreprocessor.load_data converts '?' to np.nan.
                col_series_str = raw_patient_df[col].dropna().astype(str)
                valid_diags = [d for d in col_series_str.unique() if d.lower() not in ['?', '', 'none', 'nan', '0']]
                all_diags_texts.extend(valid_diags)
        
        unique_diags_sample = sorted(list(set(all_diags_texts)))
        if unique_diags_sample:
            summary_parts.append(f"- All unique diagnosis codes mentioned: {', '.join(unique_diags_sample)}")
        else:
            summary_parts.append("- No specific primary/secondary diagnosis codes listed (or were placeholders).")

        # Key medications (example - just list if they are not 'No')
        # This assumes treatment columns in your config are the medication names
        med_cols = self.cfg.TREATMENT_COLUMNS 
        active_meds = []
        for med_col in med_cols:
            if med_col in raw_patient_df.columns:
                # Check if any visit has this medication not as 'No' or '0' (after potential mapping)
                # This depends on raw values. If raw is 'Steady', 'Up', 'Down' vs 'No'
                if raw_patient_df[med_col].astype(str).str.lower().isin(['steady', 'up', 'down']).any():
                    active_meds.append(med_col)
        if active_meds:
            summary_parts.append(f"- Key medications noted as active (Steady/Up/Down): {', '.join(active_meds)}")

        return "\n".join(summary_parts)

    def _format_shap_explanation_for_prompt(self, 
                                            shap_results: Dict[str, Any], 
                                            predicted_class_index: Optional[int], # NEW: Pass the predicted class index
                                            num_top_features_per_class: int = 5,
                                            num_contrast_features: int = 2) -> str:
        if not shap_results or "all_class_shap_values_sequence" not in shap_results:
            return "SHAP explanation data is not available or incomplete."

        expl_parts = ["Key factors identified by SHAP analysis (average impact across patient's visits):"]
        all_shaps_per_class = shap_results['all_class_shap_values_sequence']
        feature_names = shap_results['encoder_input_feature_names']
        
        class_names_cfg = getattr(self.cfg, 'TARGET_CLASS_NAMES', 
                                  [f"Class {i}" for i in range(len(all_shaps_per_class))])
        if len(class_names_cfg) != len(all_shaps_per_class):
            class_names_cfg = [f"Class {i}" for i in range(len(all_shaps_per_class))]

        # First, detail the predicted class
        if predicted_class_index is not None and 0 <= predicted_class_index < len(all_shaps_per_class):
            class_shaps_np = all_shaps_per_class[predicted_class_index]
            if class_shaps_np is not None and isinstance(class_shaps_np, np.ndarray) and class_shaps_np.ndim == 2:
                avg_abs_shaps = pd.Series(np.abs(class_shaps_np).mean(axis=0), index=feature_names).sort_values(ascending=False)
                class_name_str = class_names_cfg[predicted_class_index]
                expl_parts.append(f"\nFor the predicted outcome '{class_name_str}':")
                for feat_name, shap_val in avg_abs_shaps.head(num_top_features_per_class).items():
                    expl_parts.append(f"  - '{feat_name}' (Avg. SHAP Magnitude: {shap_val:.3f})")
            else:
                expl_parts.append(f"\nSHAP values for predicted outcome '{class_names_cfg[predicted_class_index]}' are unavailable or malformed.")
        else:
            expl_parts.append("\nPredicted class index not available for detailed SHAP summary.")

        # Optionally, add top contrastive features for other classes
        expl_parts.append("\nFor context, other outcomes were influenced by:")
        for i, class_shaps_np in enumerate(all_shaps_per_class):
            if i == predicted_class_index or class_shaps_np is None or \
               not isinstance(class_shaps_np, np.ndarray) or class_shaps_np.ndim != 2:
                continue # Skip if it's the predicted class or data is bad
            
            avg_abs_shaps = pd.Series(np.abs(class_shaps_np).mean(axis=0), index=feature_names).sort_values(ascending=False)
            class_name_str = class_names_cfg[i]
            expl_parts.append(f"  Outcome '{class_name_str}': Top factors included " +
                              ", ".join([f"'{fn}' ({fv:.2f})" for fn, fv in avg_abs_shaps.head(num_contrast_features).items()]))
        
        return "\n".join(expl_parts)

    def _get_model_prediction_summary(self, prediction_result: Dict[str, Any]) -> str:
        # (Implementation from previous response)
        if not prediction_result or not prediction_result.get("visit_predictions"): return "Model prediction not available."
        last_visit_preds = prediction_result["visit_predictions"][-1]
        pred_idx = last_visit_preds.get("predicted_class")
        class_names = getattr(self.cfg, 'TARGET_CLASS_NAMES', [f"Class {i}" for i in range(self.cfg.NUM_CLASSES)])
        pred_name = class_names[pred_idx] if pred_idx is not None and 0 <= pred_idx < len(class_names) else f"Class {pred_idx}"
        probs = ", ".join([f"'{class_names[k]}': {last_visit_preds.get(f'prob_class_{k}',0):.1%}" for k in range(len(class_names))])
        return f"Prediction for latest encounter: '{pred_name}'. Probabilities: [{probs}]."

    def _create_final_llm_prompt(self, 
                                 patient_history_summary: str, 
                                 model_prediction_summary: str, 
                                 shap_explanation_summary: str, # Summary of SHAP for predicted class
                                 retrieved_docs_summary: str) -> str:
        
        prompt = f"""You are an Expert Clinical AI Analyst. Your primary role is to explain a machine learning model's prediction regarding patient readmission to a medical professional. This explanation must be clear, deeply analytical, evidence-based (using provided documents), and adhere strictly to an objective interpretation of the model's output.

**Core Task: Construct a "Model Prediction Explainer Report" with the following structured sections:**

**Section 1: Model Prediction Overview**
    * Start by clearly stating the model's predicted readmission category for this patient's latest encounter.
    * Include the probabilities assigned by the model to all possible readmission categories.

**Section 2: Key Predictive Factors & Detailed Justification (SHAP Insights + Patient Data + Retrieved Documents)**
    * Identify the top 3-5 most influential factors (from the SHAP summary provided below) that drove the model towards its **specific predicted outcome** for this patient.
    * For EACH of these top factors:
        1.  **Factor & Patient's Value:** Clearly state the factor (e.g., 'number_outpatient', 'age', 'emb_dim_28'). Then, reference the 'Patient History Summary' below and state the patient's actual value or relevant characteristic for this factor (e.g., "The patient had [X] outpatient visits," or "The patient's age group is [Y]").
        2.  **Model's Interpretation (Directional Influence):** Explain how this specific factor value, according to its SHAP importance, likely influenced the model's prediction (e.g., "This high number of outpatient visits appears to have significantly *decreased* the model's assessed risk of readmission, contributing to the 'Class 0' prediction.").
        3.  **Evidence from Retrieved Documents (CRITICAL):** You MUST consult the 'Retrieved Context from Medical Documents' section provided below.
            * If a retrieved document offers supporting evidence, counter-evidence, or relevant medical context for why this factor might influence readmission risk in diabetic patients, you **MUST explicitly cite that document using its title and page number if available** (e.g., "This observation is contextualized by the document titled 'Intensive Blood Glucose Control...' (Doc 1, Page 8), which indicates that...").
            * If multiple documents support a point, synthesize them. If no directly relevant snippet is found for a specific factor, clearly state that (e.g., "The provided medical documents do not offer specific context on this factor's direct link to readmission for this patient profile.").
        4.  **Interpreting Embedding Dimensions:** For factors named `emb_dim_X` or similar (e.g., `diag_1_emb_X`):
            * Acknowledge these represent complex, learned patterns from coded clinical data (like diagnoses, admission/discharge types, or specific medication groups).
            * State their general influence based on SHAP (e.g., "`emb_dim_28` strongly pushed the prediction towards 'no readmission'").
            * If possible, correlate this with information in the 'Patient History Summary' (e.g., "Given the patient's listed diagnoses of '574' and '599', this important embedding dimension might be capturing patterns related to these conditions."). Then, link to the 'Retrieved Context' if any document discusses these conditions in relation to risk.

**Section 3: Overall Synthesis, Prediction Grounding, and Clinical Contextualization**
    * **Overall Rationale for Prediction:** Provide a brief overall synthesis explaining why the model likely arrived at its prediction, based on the interaction of the key factors discussed above and the patient's specific data.
    * **Assessment of Prediction Grounding:** Review the 'Retrieved Context from Medical Documents'. Based on the alignment (or divergence) between the model's key predictive factors, the patient's specific history, and the insights from these documents, provide an assessment of how well-grounded the model's prediction appears.
        * If the model's prediction and its drivers align well with information from the retrieved documents and the patient's context, state this coherence (e.g., "The model's prediction of [Class X] appears consistent with general medical understanding, as [Document Title Y] suggests that factors like [Factor A] and [Factor B] (which were important for this patient) are indeed linked to such outcomes.").
        * If the retrieved medical knowledge introduces important nuances, alternative perspectives, or highlights other significant patient factors (from their history) that the model did *not* strongly emphasize, you MUST discuss this critically. For example: "While the model's prediction of [Class X] is strongly driven by [Factor A], the document '[Relevant Doc Title]' also highlights [Factor D from RAG/history] as a critical risk for patients with [Patient Characteristic C from history]. This suggests that while the model focused on [Factor A], broader clinical context indicates [Factor D] also warrants significant consideration in the overall clinical picture."
    * **Concluding Remarks:** Offer a final summary statement about the model's prediction in light of all available information. This is not about making a new prediction or giving new probabilities, but about framing the model's output within a richer, evidence-informed clinical context.

**Strict Constraints (Reiteration):**
    * Maintain a professional, objective, and highly informative tone.
    * The explanation must be about the MODEL'S PREDICTION and its drivers.
    * **Absolutely DO NOT provide direct medical advice, suggest treatments, or make new diagnoses.**
    * Attribute all medical knowledge claims to the "Retrieved Context from Medical Documents" by citing titles (and page numbers, if provided in the context).

---
**PROVIDED INFORMATION FOR SYNTHESIS:**

**1. Patient History Summary:**
{patient_history_summary}
---
**2. Model's Prediction for the Latest Encounter:**
{model_prediction_summary}
---
**3. Key Factors Influencing the Model (SHAP - average impact across patient's visits for various outcomes):**
{shap_explanation_summary} 
---
**4. Retrieved Context from Medical Documents (Use this to justify/contextualize the "Why", citing with Title and Page as available in this section):**
{retrieved_docs_summary} 
---

**EXPERT CLINICAL AI ANALYST REPORT (Follow the structured sections and all instructions meticulously):**
"""
        return prompt

    def generate_diagnostic_explanation(self, patient_csv_path: str) -> Dict[str, Any]:
        """
        Main method: Processes a single patient's CSV, gets predictions, SHAP explanations,
        and generates an LLM-based diagnostic explanation using an explicit RAG step.
        """
        logger.info(f"--- Starting Full Diagnostic Explanation for Patient CSV: {patient_csv_path} ---")

        if not self.engine:
            logger.error("Predictor engine not initialized in DiabetesAgent. Cannot proceed.")
            return {"error": "DiabetesAgent's predictor engine is not available."}
        
        # 1. Load raw patient data from CSV
        logger.info(f"Loading patient data from CSV: {patient_csv_path}")
        try:
            raw_patient_df = FirstPhasePreprocessor.load_data(patient_csv_path, self.cfg.MISSING_VALUES)
            if raw_patient_df.empty:
                logger.error(f"No data found or loaded from patient CSV: {patient_csv_path}")
                return {"error": f"No data in patient CSV: {patient_csv_path}"}
            raw_patient_df = raw_patient_df.sort_values(by=self.cfg.ENCOUNTER_ID_COL) # Ensure order
        except FileNotFoundError:
            logger.error(f"Patient CSV not found: {patient_csv_path}")
            return {"error": f"Patient CSV not found: {patient_csv_path}"}
        except Exception as e:
            logger.error(f"Failed to load patient CSV '{patient_csv_path}': {e}", exc_info=True)
            return {"error": f"Failed to load patient CSV: {str(e)}"}

        # 2. Get model prediction using the engine
        logger.info("Getting model prediction for the patient...")
        try:
            # This returns: {"patient_id": ..., "visit_predictions": ..., "processed_model_input": ...}
            prediction_result_dict = self.engine.predict_for_patient(raw_patient_df)
            model_pred_summary = self._get_model_prediction_summary(prediction_result_dict) # Formats for prompt
        except Exception as e:
            logger.error(f"Error during model prediction for patient: {e}", exc_info=True)
            return {"error": f"Model prediction step failed: {str(e)}"}

        # 3. Prepare SHAP background data (if SHAP is to be generated)
        predicted_class_idx = None
        if prediction_result_dict and prediction_result_dict.get("visit_predictions"):
            last_visit_preds = prediction_result_dict["visit_predictions"][-1]
            predicted_class_idx = last_visit_preds.get("predicted_class")
        shap_results_dict = None # Initialize
        shap_explanation_summary = "SHAP analysis was not performed or failed." # Default summary

        background_csv_default_path = str(Path(self.cfg.DATA_DIR) / "training_samples" / "sample_raw_patients_for_background.csv")
        background_csv_path = getattr(self.cfg, 'SHAP_BACKGROUND_CSV_PATH', background_csv_default_path)
        
        if not os.path.exists(background_csv_path):
            logger.warning(f"SHAP background CSV not found at {background_csv_path}. SHAP explanations will be skipped.")
        else:
            logger.info(f"Preparing SHAP background data using: {background_csv_path}")
            try:
                df_bg_raw = FirstPhasePreprocessor.load_data(background_csv_path, self.cfg.MISSING_VALUES)
                background_data_tensor = prepare_shap_background_data(
                    self.engine, df_bg_raw,
                    num_background_sequences_aim=getattr(self.cfg, 'NUM_SHAP_BACKGROUND_SAMPLES', 50)
                )
                
                # 4. Generate SHAP explanation
                logger.info("Generating SHAP explanations...")
                numerical_features = self.cfg.NUMERICAL_FEATURES
                ohe_actual_cols = self.engine.actual_ohe_columns_for_shap_naming # From engine's init

                shap_results_dict = generate_shap_explanation(
                    engine=self.engine,
                    raw_patient_df_to_explain=raw_patient_df,
                    background_data_tensor_for_shap=background_data_tensor,
                    config_numerical_features=numerical_features,
                    config_ohe_actual_cols=ohe_actual_cols
                )
                if shap_results_dict:
                    shap_explanation_summary = self._format_shap_explanation_for_prompt(shap_results_dict, predicted_class_index=predicted_class_idx, num_top_features_per_class=5)
                else:
                    logger.warning("SHAP explanation generation returned no results.")
                    shap_explanation_summary = "SHAP analysis completed but yielded no specific feature explanations."

            except Exception as e:
                logger.error(f"Error during SHAP background preparation or explanation generation: {e}", exc_info=True)
                shap_explanation_summary = f"SHAP analysis could not be completed due to an error: {str(e)}"
        
        # 5. Summarize patient history for LLM
        patient_hist_summary = self._get_patient_history_summary_for_prompt(raw_patient_df)
        
        # 6. Explicit RAG step: Retrieve documents based on focused query
        retrieved_docs_text_for_prompt = "No relevant medical context was retrieved or RAG system is unavailable."
        if self.llm and self.vectorstore : # Check if components for RAG are available
            # Construct a focused query using patient history, prediction, and top SHAP factors
            predicted_class_name_for_rag = model_pred_summary.split("'")[1] if "'" in model_pred_summary else "unknown class"
            top_shap_factors_sample = shap_explanation_summary.splitlines()[2][:100] if shap_explanation_summary.startswith("Key factors") and len(shap_explanation_summary.splitlines()) > 2 else "key model factors"
            history_snippet_for_rag = patient_hist_summary.split("visit(s).")[1][:150] if "visit(s)." in patient_hist_summary else patient_hist_summary[:150]

            focused_rag_query = (
                f"Provide medical context regarding diabetes readmission risk for a patient predicted as "
                f"'{predicted_class_name_for_rag}'. Key influencing factors identified: {top_shap_factors_sample}. "
                f"Patient history includes: {history_snippet_for_rag}..."
            )
            logger.info(f"Focused RAG query: {focused_rag_query}")
            
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": getattr(self.cfg, "RAG_NUM_DOCS_TO_RETRIEVE", getattr(self.cfg, "RAG_NUM_CHUNKS_TO_RETRIEVE", 5))}
            )
            try:
                source_documents = retriever.get_relevant_documents(focused_rag_query)
                if source_documents:
                    logger.info(f"RAG retrieved {len(source_documents)} documents based on focused query:")
                    temp_retrieved_texts = ["Retrieved context from medical documents that may be relevant:"]
                    for i, doc in enumerate(source_documents):
                        doc_title = doc.metadata.get('title', doc.metadata.get('source', 'N/A'))
                        doc_page = doc.metadata.get('page', 'N/A')
                        snippet = f"Doc {i+1} (Source: {doc_title}, Page: {doc_page}): \"{doc.page_content[:300].strip()}...\""
                        logger.info(f"  {snippet}") # Log for debugging
                        temp_retrieved_texts.append(snippet)
                    retrieved_docs_text_for_prompt = "\n".join(temp_retrieved_texts)
                else:
                    logger.info("Focused RAG query returned no specific source documents.")
                    retrieved_docs_text_for_prompt = "No specific documents were identified by the RAG system for this query's focus."
            except Exception as e:
                logger.error(f"Error during focused RAG document retrieval: {e}", exc_info=True)
                retrieved_docs_text_for_prompt = "An error occurred while attempting to retrieve supporting documents."
        else:
            logger.warning("LLM or vectorstore not available. RAG context retrieval will be skipped.")

        # 7. Create the final LLM prompt including the explicitly retrieved context
        final_llm_prompt = self._create_final_llm_prompt(
            patient_hist_summary,
            model_pred_summary,
            shap_explanation_summary,
            retrieved_docs_text_for_prompt # Pass the new, explicit summary of retrieved docs
        )
        logger.debug(f"\n--- Generated Final LLM Prompt ---\n{final_llm_prompt}\n--------------------------")

        # 8. Invoke LLM with this comprehensive prompt (NO RAG chain here, just the LLM)
        llm_explanation = "LLM is not available or failed during synthesis."
        if self.llm: # Check if LLM was initialized
            logger.info("Invoking LLM directly for final synthesis with explicit RAG context in prompt...")
            try:
                response_obj = self.llm.invoke(final_llm_prompt) # Direct call to LLM with the complete prompt
                llm_explanation = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            except Exception as e:
                logger.error(f"Error invoking LLM for final synthesis: {e}", exc_info=True)
                llm_explanation = f"LLM synthesis failed: {str(e)}"
        else:
            logger.warning("LLM not available. Final explanation will be based on rule-based summary without LLM synthesis.")
            # Fallback explanation if LLM is not available
            llm_explanation = (f"Model Prediction: {model_pred_summary}\n"
                               f"Key Factors (SHAP): {shap_explanation_summary}\n"
                               f"Retrieved Context: {retrieved_docs_text_for_prompt}\n"
                               "(LLM system not available for full synthesis.)")
            
        return {
            "patient_id": prediction_result_dict.get("patient_id"),
            "model_prediction_summary": model_pred_summary,
            "model_predictions_per_visit": prediction_result_dict.get("visit_predictions"),
            "shap_explanation_summary_for_prompt": shap_explanation_summary,
            "retrieved_documents_summary_for_llm": retrieved_docs_text_for_prompt,
            "raw_shap_results": shap_results_dict, # This contains the detailed SHAP arrays
            "llm_generated_explanation": llm_explanation,
            "full_llm_prompt_for_debug": final_llm_prompt
        }

# --- Example Main Execution Block (Simplified Error Handling) ---
if __name__ == "__main__":
    # For more detailed logs from all modules during this direct script run:
    logging.basicConfig(
        level=logging.DEBUG, # DEBUG level for development
        format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- Add Argument Parsing for patient_csv_path ---
    parser = argparse.ArgumentParser(description="Run DiabetesAgent for a single patient CSV to get diagnostic explanation.")
    parser.add_argument(
        "patient_csv_path",
        type=str,
        help="Path to the CSV file containing raw visits for a single patient."
    )
    # We don't need arguments for background_csv or num_bg_samples here,
    # as DiabetesAgent.generate_diagnostic_explanation uses AppConfig defaults for those.
    
    args = parser.parse_args()

    logger.info("--- Starting DiabetesAgent Demo with Command-Line Patient CSV ---")

    if not os.path.exists(args.patient_csv_path):
        logger.error(f"Provided patient CSV path does not exist: {args.patient_csv_path}")
        sys.exit(1)
    
    try:
        # Initialize the Agent
        agent = DiabetesAgent(cfg=AppConfig)
        
        if not agent.engine: # Check if critical component loaded
            logger.error("DiabetesAgent engine failed to initialize. Exiting demo.")
            sys.exit(1)
            
        logger.info(f"Processing patient data from: {args.patient_csv_path}")
        # Call the main method of the agent with the provided CSV path
        full_analysis_output = agent.generate_diagnostic_explanation(args.patient_csv_path)
        
        # --- Print the Results ---
        print("\n\n===================================================")
        print(f"      DIABETES READMISSION AGENT ANALYSIS      ")
        print("===================================================")
        if "error" in full_analysis_output:
            print(f"\n🛑 Agent Error: {full_analysis_output['error']}")
            # Optionally print other available info if error occurred mid-way
            if "prediction_summary" in full_analysis_output:
                 print(f"\n📈 Model Prediction Summary (despite error later):")
                 print(f"   {full_analysis_output['prediction_summary']}")
        else:
            print(f"\n👤 Patient ID: {full_analysis_output.get('patient_id', 'N/A')}") # Added N/A default
            
            print("\n📈 Model Prediction Summary:")
            print(f"   {full_analysis_output.get('model_prediction_summary', 'Not available.')}")

            print("\n📊 SHAP Explanation Summary (used for prompt):")
            print(full_analysis_output.get('shap_explanation_summary_for_prompt', 'SHAP summary not available.'))
            
            print("\n💡 LLM Generated Explanation:")
            print(full_analysis_output.get('llm_generated_explanation', 'LLM explanation not available.'))
            
            # For debugging, you might want to see the full prompt
            # if AppConfig.DEBUG_MODE or similar: # Assuming a DEBUG_MODE in config
            #     print("\n🔧 Full LLM Prompt (for debugging):")
            #     print(full_analysis_output.get('full_llm_prompt_for_debug', 'Not available.'))

        print("===================================================\n")

    except RuntimeError as rterr: # Catch critical init errors from engine or agent itself
        logger.error(f"Agent could not be initialized or run: {rterr}", exc_info=True)
    except FileNotFoundError as fnf_err: # Catch file not found for main patient CSV if path check fails
        logger.error(f"File error: {fnf_err}", exc_info=True)
    except Exception as e: # Catch any other unexpected errors during agent execution
        logger.error(f"An unexpected error occurred in the DiabetesAgent demo: {e}", exc_info=True)