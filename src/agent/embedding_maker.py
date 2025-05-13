import os
import glob
import shutil
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader




def vectorstore_maker(db_name,doc_folder):

    # Ruta al directorio
    folder = glob.glob(doc_folder)[0]

    # Cargar todos los PDFs en ese directorio
    documents = []
    for pdf_path in glob.glob(os.path.join(folder, "*.pdf")):
        print(f"Cargando documento: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = "PDF"
            documents.append(doc)

    print(f"Se han cargado los documentos correctamente.")


    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Se han dividido los documentos en {len(chunks)} partes.")



    # Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Delete if already exists
    if os.path.exists(db_name):
        shutil.rmtree(db_name)

    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

    return None

def vectorstore_loader(db_name):

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

    return vectorstore
