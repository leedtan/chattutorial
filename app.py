import os

import openai
from flask import Flask, jsonify, render_template, request, session
from flask_session import Session

from chatbots import ChatBot, EchoBot

app = Flask(__name__)
chatbot: ChatBot = EchoBot()
openai.api_key = os.environ.get("PRIVATEKEY")
if openai.api_key is None:
    from keys import PRIVATEKEY

    openai.api_key = PRIVATEKEY
Session(app)

openai.api_key = os.getenv('OPENAI_API_KEY')
@app.route("/")
def home():
    model_name = "gpt-3.5-turbo"
    port = 5000
    return render_template("index.html", model_name=model_name, port=port)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if data is None or "message" not in data:
        return jsonify({"error": "Missing 'message' in request."}), 400
    message = data["message"]
    response = {"response": f"Echo: {message}"}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
