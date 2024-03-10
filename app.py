from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    # Example variables to pass to the template
    model_name = "gpt-3.5-turbo"
    port = 5000
    return render_template("index.html", model_name=model_name, port=port)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data["message"]
    # Placeholder response for demonstration
    response = {"response": f"Echo: {message}"}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
