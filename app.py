import logging
import os

import openai
from flask import Flask, jsonify, render_template, request, session

from chatbots import ApiBot, EchoBot, FineTuneBot, LocalLLMBot, RAGBot
from flask_session import Session

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_default_secret_key")
app.config["SESSION_TYPE"] = "filesystem"

Session(app)

echo_bot = EchoBot()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    from keys import OPENAI_API_KEY

    openai.api_key = OPENAI_API_KEY
api_bot = ApiBot("gpt-3.5-turbo", openai)
openai.client = openai.Client(api_key=openai.api_key)
fine_tune_bot = FineTuneBot(openai)
llm_bot = LocalLLMBot()
rag_bot = RAGBot(
    llm_repo_id="google/flan-t5-large",
    web_loader_url="http://jalammar.github.io/illustrated-transformer/",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
)


@app.route("/")
def home():
    if "conversation_history" in session:
        del session["conversation_history"]
    model_name = "gpt-3.5-turbo"
    port = 5000
    return render_template("index.html", model_name=model_name, port=port)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data or "bot_type" not in data:
        return jsonify({"error": "Missing 'message' or 'bot_type' in request."}), 400

    bot_type = data["bot_type"]
    message = data["message"]
    context = data.get("context", "")
    conversation_history = session.get("conversation_history", [])

    if context and len(conversation_history) == 0:
        conversation_history.append({"role": "user", "content": context})
    chatbot = {
        "api": api_bot,
        "echo": echo_bot,
        "fine_tune": fine_tune_bot,
        "local_llm": llm_bot,
        "rag": rag_bot,
    }[bot_type]

    reply = chatbot.get_response(message, conversation_history)
    session["conversation_history"] = conversation_history
    return jsonify({"response": reply})


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("conversation_history", None)
    return jsonify({"status": "reset complete"}), 200


if __name__ == "__main__":
    app.run(debug=True)
