# app.py
from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from backend.langraph_agent.retriever import load_hybrid_retriever
from backend.langraph_agent.agent import build_langgraph_agent
import os
import time

app = Flask(__name__,template_folder="frontend/templates", static_folder="frontend/static")

retriever = load_hybrid_retriever(
        index_path="./vectorstore/faiss_vectorestore_v2",
        pickle_path="./vectorstore/documents_v2.pkl",
        model_path="./bge-m3"
    )
llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.1)


agent = build_langgraph_agent(retriever, llm)


def get_chat_response(user_message):
    result = agent.invoke({"question": user_message})
    return result['final_response']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    response = get_chat_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
