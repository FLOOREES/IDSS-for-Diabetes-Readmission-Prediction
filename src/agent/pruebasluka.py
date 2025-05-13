import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()
api_key = os.getenv("llm_key")

print(api_key)

genai.configure(api_key=api_key)

# Carga un modelo de la capa gratuita
model = genai.GenerativeModel("models/gemini-2.0-flash")

# Genera texto
response = model.generate_content("buenos días gemini")
print(response.text)