import os
import google.generativeai as genai

with open("src/agent/api_key.txt", "r") as file:
    api_key = file.read().strip()

genai.configure(api_key=api_key)

# Carga un modelo de la capa gratuita
model = genai.GenerativeModel("models/gemini-2.0-flash")

# Genera texto
response = model.generate_content("buenos días gemini")
print(response.text)