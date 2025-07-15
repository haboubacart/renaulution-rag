from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.langraph_agent.retriever import load_hybrid_retriever
from backend.langraph_agent.agent import build_langgraph_agent
import os
import sys
import warnings

# Supprimer les warnings inutiles
warnings.filterwarnings("ignore")

# S'assurer que les imports relatifs fonctionnent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")


print("Strating......")
# Chargement du retriever local + modèle HF
retriever = load_hybrid_retriever(
    index_path="./vectorstore/faiss_vectorestore",
    pickle_path="./vectorstore/documents.pkl",
    model_path="./models/bge-m3"
)

# LLM
"""llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1
)"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.1
)

# Agent LangGraph
agent = build_langgraph_agent(retriever, llm)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    result = agent.invoke({"question": user_message})

    final_response = result.get("final_response", "")
    image = result.get("graph_base64")  
    print(result.get("rag_result"))
    return jsonify({
        "response": final_response,
        "image": image  # null si pas d’image
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
