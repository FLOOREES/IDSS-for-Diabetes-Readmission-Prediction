
import shap
import os
from dotenv import load_dotenv
from embedding_maker import vectorstore_loader, vectorstore_maker
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

DOC_FOLDER = "C:/Users/lukag/OneDrive/Desktop/Universidad/3ero/cuadrimestre2/PAID/github/IDSS-for-Diabetes-Readmission-Prediction/src/agent/Diabetes_docs/"
DB_NAME = "C:/Users/lukag/OneDrive/Desktop/Universidad/3ero/cuadrimestre2/PAID/github/IDSS-for-Diabetes-Readmission-Prediction/src/agent/db_place"

load_dotenv()
api_key = os.getenv("gemini_key")
os.environ["GOOGLE_API_KEY"] = api_key


# Model y data por defecto son NOne pero solo para pruebas, en la version final se tiene que quitar

class DiabetesAgent:

    def __init__(self, model=None,data=None):
        self.model = model
        self.data = data

        self.question = None

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        self.vectorstore = self._vectorstore_import(DB_NAME, DOC_FOLDER)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # o 'map_reduce', 'refine', según necesites
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}))


        #self.shap_explainer = shap.DeepExplainer(self.model, self.data)

    def _vectorstore_import(self, db_name, doc_folder):
        if not os.path.exists(db_name):
            print("El vectorstore no existe. Creando uno nuevo...")
            vectorstore_maker(db_name, doc_folder)
        else:
            print("El vectorstore ya existe. Cargando...")

        return vectorstore_loader(db_name)

    # def __obtain_shap_values(self, data):
    #     shap_values = self.shap_explainer.shap_values(data)
    #     return shap_values
    
    def __create_question(self, shap_values):
        shap_values_str = str(shap_values)

        # question = "What do the following SHAP values mean?"
        # question += f"\n{shap_values_str}"

        question = "¿de que son los documentos proporcionados?"

        return question
    
    def generate_response(self):
        #question = self.__create_question(self.__obtain_shap_values(self.data))
        question = self.__create_question('shap_values')
        output = self.qa_chain.invoke(question, return_only_outputs=True)
        return output["result"]

    

    
if __name__ == "__main__":

    diabetes_agent = DiabetesAgent()
    response = diabetes_agent.generate_response()
    print(response)