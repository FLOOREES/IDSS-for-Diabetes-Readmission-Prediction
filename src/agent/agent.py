
import shap
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("gemini_key")





class DiabetesAgent:

    def __init__(self, model,data):
        self.model = model
        self.data = data

        self.question = None

        self.llm = genai.GenerativeModel("models/gemini-2.0-flash")
        self.shap_explainer = shap.DeepExplainer(self.model, self.data)

    def __obtain_shap_values(self, data):
        shap_values = self.shap_explainer.shap_values(data)
        return shap_values
    
    def __create_question(self, shap_values):
        shap_values_str = str(shap_values)

        question = "What do the following SHAP values mean?"
        question += f"\n{shap_values_str}"

        return question
    
    def generate_response(self):
        question = self.__create_question(self.__obtain_shap_values(self.data))
        response = self.llm.generate_content(question)
        return response.text

    

    

