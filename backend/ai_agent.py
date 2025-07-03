from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
llm_key = os.getenv("GOOGLE_API_KEY")

# Initialisation du LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    temperature=0.7
)

# Utilisation simple
response = llm.invoke("Bonjour toi, comment ça va ? je vais a paris, reponds en 100 mots")
print(response.content)