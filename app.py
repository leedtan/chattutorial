import logging
import os

import openai
from flask import Flask, jsonify, render_template, request, session

from chatbots import ApiBot, EchoBot
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

    chatbot = api_bot if bot_type == "api" else echo_bot

    reply = chatbot.get_response(message, conversation_history)
    session["conversation_history"] = conversation_history
    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True)
