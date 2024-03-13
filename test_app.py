import pytest
from flask import json

from app import app as flask_app

@pytest.fixture
def client():
    """Provide a test client for the Flask application."""
    with flask_app.test_client() as test_client:
        yield test_client

def test_home_page(client):
    """Test if the home page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"gpt-3.5-turbo" in response.data, "The home page does not display the expected model name."

def test_chat_post_success(client):
    """Test successful POST request to the chat endpoint."""
    bot_type = "echo"  # or "api", depending on what you want to test here
    response = client.post("/chat", data=json.dumps({"message": "Hello", "bot_type": bot_type}), content_type="application/json")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["response"] is not None, "The chat response should not be None."

def test_chat_post_no_data(client):
    """Test POST request to the chat endpoint without any data."""
    response = client.post("/chat", data=json.dumps({}), content_type="application/json")
    assert response.status_code == 400, "The server should return a 400 error for requests without data."
