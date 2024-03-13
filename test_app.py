import os

import pytest
from flask import json

from app import app as flask_app
from app import create_chatbot


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("USE_API_BOT", "false")
    os.environ["USE_API_BOT"] = "false"

    flask_app.USE_API = False
    flask_app.chatbot = create_chatbot()
    # breakpoint()
    yield flask_app


@pytest.fixture
def client(monkeypatch):
    flask_app.config["TESTING"] = True
    monkeypatch.setenv("USE_API_BOT", "false")
    flask_app.USE_API = False
    os.environ["USE_API_BOT"] = "false"
    flask_app.chatbot = create_chatbot()
    # breakpoint()
    with flask_app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"gpt-3.5-turbo" in response.data


def test_chat_post_success(client):
    response = client.post(
        "/chat", data=json.dumps({"message": "Hello"}), content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "Echo: Hello" in data["response"]


def test_chat_post_no_data(client):
    response = client.post(
        "/chat", data=json.dumps({}), content_type="application/json"
    )
    assert response.status_code == 400
