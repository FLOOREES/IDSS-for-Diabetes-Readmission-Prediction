
import shap
import os
from dotenv import load_dotenv
from embedding_maker import vectorstore_loader, vectorstore_maker
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()
api_key = os.getenv("gemini_key")
os.environ["GOOGLE_API_KEY"] = api_key



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

    

    
if __name__ == "__main__":

    doc_folder = "C:/Users/lukag/OneDrive/Desktop/Universidad/3ero/cuadrimestre2/PAID/github/IDSS-for-Diabetes-Readmission-Prediction/src/agent/Diabetes_docs/"
    db_name = "C:/Users/lukag/OneDrive/Desktop/Universidad/3ero/cuadrimestre2/PAID/github/IDSS-for-Diabetes-Readmission-Prediction/src/agent/db_place"

    # 1. Crea el vectorstore (si no existe)
    if not os.path.exists(db_name):
        vectorstore_maker(db_name, doc_folder)
    else:
        print("El vectorstore ya existe. Cargando...")


    vectorstore = vectorstore_loader(db_name)


    # 1. Configura el retriever (igual que antes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Instancia Gemini Pro (texto)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # 3. Construye la cadena RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",      # o 'map_reduce', 'refine', según necesites
        retriever=retriever)
    

    # 4. Función de consulta
    def answer_with_rag(prompt: str) -> str:
        return qa_chain.run(prompt)

    # Ejemplo
    respuesta = answer_with_rag("¿de que son los documentos proporcionados?")
    print(respuesta)