import logging
import os

import openai
from flask import Flask, jsonify, render_template, request, session

from chatbots import ApiBot, ChatBot, EchoBot
from flask_session import Session

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_default_secret_key")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.USE_API = os.environ.get("USE_API_BOT", "false").lower() == "true"


def create_chatbot():

    if app.USE_API:
        model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if openai.api_key is None:
            from keys import OPENAI_API_KEY

            openai.api_key = OPENAI_API_KEY
        return ApiBot(model_name, openai)
    else:
        return EchoBot()


# Move the chatbot initialization here
chatbot: ChatBot = create_chatbot()

# if USE_API:
#     model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     if openai.api_key is None:
#         from keys import OPENAI_API_KEY

#         openai.api_key = OPENAI_API_KEY
#     chatbot: ChatBot = ApiBot(model_name, openai)
# else:
#     chatbot: ChatBot = EchoBot()


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
    if data is None or "message" not in data:
        return jsonify({"error": "Missing 'message' in request."}), 400

    message = data["message"]
    conversation_history = session.get("conversation_history", [])

    reply = chatbot.get_response(message, conversation_history)

    session["conversation_history"] = conversation_history
    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True)
