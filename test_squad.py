import pytest
from flask import json

from app import app as flask_app


@pytest.fixture
def client():
    """Provide a test client for the Flask application."""
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize(
    "query,expected_keywords",
    [
        (
            "What areas did Beyonce compete in when she was growing up?",
            ["singing", "dancing"],
        ),
    ],
)
def test_squad_queries(client, query, expected_keywords):
    """Test POST request to the chat endpoint with specific queries expected to improve after fine-tuning on SQuAD."""
    bot_type = "api"
    context = "You are a information research assistant. If you don't have the specific information, please say 'I don't have that specific information'."

    response = client.post(
        "/chat",
        data=json.dumps({"message": query, "bot_type": bot_type, "context": context}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert all(
        keyword in data["response"].lower() for keyword in expected_keywords
    ), f"""
        The chat response does not adequately address the specific medical query: {query}.
        Expected to find keywords: {expected_keywords}. 
        Instead, bot responded:
        {data["response"]}.
        Words included correctly: {{keyword for keyword in expected_keywords if keyword in data['response'].lower()}}.
        Words missing: {{keyword for keyword in expected_keywords if keyword not in data['response'].lower()}}
        """
