import pytest
from flask import json

from app import app as flask_app


@pytest.fixture
def app():
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


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
    assert (
        "Echo: Hello" in data["response"]
    )


def test_chat_post_no_data(client):
    response = client.post(
        "/chat", data=json.dumps({}), content_type="application/json"
    )
    assert response.status_code == 400

