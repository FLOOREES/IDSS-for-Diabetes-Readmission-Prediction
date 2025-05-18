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
        # (Implementation from previous response - keeping it simple, can be enhanced later)
        if raw_patient_df.empty: return "No patient history provided."
        summary_parts = [f"Patient ID {raw_patient_df[self.cfg.PATIENT_ID_COL].iloc[0]} has {len(raw_patient_df)} visit(s)."]
        if 'age' in raw_patient_df.columns: summary_parts.append(f"- Age group(s): {raw_patient_df['age'].unique().tolist()}")
        if 'time_in_hospital' in raw_patient_df.columns: summary_parts.append(f"- Time in hospital (days): {raw_patient_df['time_in_hospital'].tolist()}")
        diag_cols = getattr(self.cfg, 'DIAG_COLS', ['diag_1', 'diag_2', 'diag_3'])
        all_diags = [d for col in diag_cols if col in raw_patient_df for d_val in raw_patient_df[col].dropna().astype(str).unique() for d in [d_val] if d not in ['?', '0', 'nan', 'None', ''] and not d.isspace()]
        if all_diags: summary_parts.append(f"- Key diagnoses (sample): {', '.join(sorted(list(set(all_diags)))[:10])}")
        return "\n".join(summary_parts)

    def _format_shap_explanation_for_prompt(self, shap_results: Dict[str, Any], num_top_features: int = 5) -> str:
        # (Implementation from previous response)
        if not shap_results or "all_class_shap_values_sequence" not in shap_results: return "SHAP explanation not available."
        expl = ["Key factors (average SHAP magnitude across visits):"]
        class_names = getattr(self.cfg, 'TARGET_CLASS_NAMES', [f"Class {i}" for i in range(len(shap_results['all_class_shap_values_sequence']))])
        for i, shaps_np in enumerate(shap_results['all_class_shap_values_sequence']):
            if shaps_np is None: continue
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            avg_abs_shaps = pd.Series(np.abs(shaps_np).mean(axis=0), index=shap_results['encoder_input_feature_names']).sort_values(ascending=False)
            expl.append(f"\nFor outcome '{name}':")
            for feat, val in avg_abs_shaps.head(num_top_features).items(): expl.append(f"  - {feat} ({val:.3f})")
        return "\n".join(expl)

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
                                 shap_explanation_summary: str,
                                 retrieved_docs_summary: str) -> str: # Added 'retrieved_docs_summary'
        """Creates the comprehensive prompt for the LLM, including RAG instructions."""
        
        prompt = f"""You are an expert medical AI assistant explaining a machine learning model's prediction about patient readmission to a medical professional. Your explanation MUST be grounded in all provided information.

Instructions for your explanation:
1.  **State the Model's Prediction:** Clearly state the predicted readmission category and probabilities.
2.  **Identify Key Predictive Factors:** Based on SHAP, list the top 2-3 influential factors FOR THE PREDICTED CLASS.
3.  **Explain the "Why" - Synthesize and Reason:**
    * For each key factor, briefly explain its potential connection to readmission risk using the patient's specific history.
    * **Crucially, you MUST explicitly reference and integrate insights from the 'Retrieved Context from Medical Documents' section below to support your reasoning.** For example, if 'number_outpatient' is a key factor and the retrieved context mentions 'frequent outpatient visits reduce readmission risk...', you MUST connect this.
    * Your goal is to build a logical bridge: Patient Data -> SHAP Factors -> Medical Knowledge (from Retrieved Context) -> Prediction Explanation.
4.  **Professionalism and Constraints:** Objective, informative, explain model output only. DO NOT provide medical advice or new diagnoses.
5.  **Conciseness:** Be thorough in reasoning but concise.

---
PROVIDED INFORMATION:

Patient History Summary:
{patient_history_summary}
---
Model's Prediction for the Latest Encounter:
{model_prediction_summary}
---
Key Factors Influencing the Model (SHAP - average impact across patient's visits):
{shap_explanation_summary}
---
Retrieved Context from Medical Documents: 
{retrieved_docs_summary} 
---

DETAILED EXPLANATION FOR THE MEDICAL PROFESSIONAL (ensure you explicitly refer to how the 'Retrieved Context' supports the link between SHAP factors and the prediction):
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
                    shap_explanation_summary = self._format_shap_explanation_for_prompt(shap_results_dict)
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
                search_kwargs={"k": getattr(self.cfg, "RAG_NUM_DOCS_TO_RETRIEVE", 2)} # Retrieve 2-3 focused docs
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