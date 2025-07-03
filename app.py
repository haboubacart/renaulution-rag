# app.py
from flask import Flask, render_template, request, jsonify
from langchain_core.documents import Document
import time

app = Flask(__name__,template_folder="frontend/templates", static_folder="frontend/static")

# Dummy function simulating a retriever or LLM
# Replace this with your LangChain retriever/agent logic
def get_chat_response(user_message):
    return f"(Réponse automatique) Tu as dit : {user_message}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    response = get_chat_response(user_message)
    time.sleep(2)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
