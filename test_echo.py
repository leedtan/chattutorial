import pytest
from flask import json

from app import app as flask_app


@pytest.fixture
def client():
    """Provide a test client for the Flask application."""
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize(
    "bot_type,expected_in_response",
    [
        ("echo", "echo:"),
        ("api", "thank you"),
        ("local_llm", "thank you"),
        ("rag", "thank you"),
    ],
)
def test_chat_response(client, bot_type, expected_in_response):
    """Test POST request to the chat endpoint with different bot types."""
    response = client.post(
        "/chat",
        data=json.dumps(
            {
                "message": "You are a greeting agent. Please respond by thanking me",
                "bot_type": bot_type,
            }
        ),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert (
        expected_in_response in data["response"].lower()
    ), f"The chat response should include: {expected_in_response}"
