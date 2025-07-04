# app.py
from flask import Flask, render_template, request, jsonify
from langchain_core.documents import Document
from backend.retriever import load_hybrid_retriever
from backend.ai_agent import build_agent, ask_agent
import time

app = Flask(__name__,template_folder="frontend/templates", static_folder="frontend/static")



retriever = load_hybrid_retriever(
        index_path="vectorstore/faiss_vectorestore",
        pickle_path="vectorstore/documents.pkl",
        model_path="multilingual-e5-large"
    )

agent = build_agent(retriever)

# Dummy function simulating a retriever or LLM
# Replace this with your LangChain retriever/agent logic
def get_chat_response(user_message):
    return ask_agent(agent, user_message)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    response = get_chat_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
