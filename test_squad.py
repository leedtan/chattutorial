import pytest
from flask import json
import json as pythonjson

from app import app as flask_app


@pytest.fixture
def client():
    """Provide a test client for the Flask application."""
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize(
    "query, expected_keywords, bot_type, expect_success",
    [
        (
            "In 1996, Destiny's Child picked their name based on a quote in which book of the Bible?",
            ["Isaiah"],
            "api",
            False,
        ),
        (
            "In 1996, Destiny's Child picked their name based on a quote in which book of the Bible?",
            ["Isaiah"],
            "fine_tune",
            True,
        ),
        (
            "What areas did Beyonce compete in when she was growing up?",
            ["singing", "dancing"],
            "api",
            False,
        ),
        (
            "What areas did Beyonce compete in when she was growing up?",
            ["singing", "dancing"],
            "fine_tune",
            True,
        ),
    ],
)
def test_squad_queries(client, query, expected_keywords, bot_type, expect_success):
    """Test POST request to the chat endpoint with specific queries expected to improve after fine-tuning on SQuAD."""
    context = "You are a information research assistant. If you don't have the specific information, please say 'I don't have that specific information'."

    response = client.post(
        "/chat",
        data=json.dumps({"message": query, "bot_type": bot_type, "context": context}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    if expect_success:
        assert all(
            keyword.lower() in data["response"].lower() for keyword in expected_keywords
        ), f"""
            The chat response does not adequately address the specific query: {query}.
            Expected to find keywords: {expected_keywords}.
            Instead, bot {bot_type} responded:
            {data["response"]}.
            Words included correctly: {{keyword for keyword in expected_keywords if keyword in {data['response'].lower()}}}.
            Words missing: {{keyword for keyword in expected_keywords if keyword not in {data['response'].lower()}}}
            """
    else:
        assert not all(
            [
                keyword.lower() in data["response"].lower()
                for keyword in expected_keywords
            ]
        ), data["response"].lower()


@pytest.mark.parametrize(
    "query, expected_answer, bot_type, expect_success",
    [
        (
            "when did Rhee evacuate from Seoul with some of the government?",
            ["June 27th"],
            "api",
            False,
        ),
        (
            "when did Rhee evacuate from Seoul with some of the government?",
            ["June 27th"],
            "fine_tune",
            False,
        ),
    ],
)
def test_squad_queries_aitest(client, query, expected_answer, bot_type, expect_success):
    """Test POST request to the chat endpoint with specific queries expected to improve after fine-tuning on SQuAD."""
    context = (
        "You are a information research assistant. If you don't have the specific information,"
        + " please say 'I don't have that specific information'."
    )

    response = client.post(
        "/chat",
        data=json.dumps({"message": query, "bot_type": bot_type, "context": context}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    answer = data["response"].strip()
    client.post("/reset")
    eval_context = """
You are an automated testing assistant. Given a question and its expected answer,
return a structured response with two parts: 'correctness', 
which should be 1 if the AI's answer is correct, -1 if incorrect, and 0 if uncertain;
and 'details' for any additional comments or explanations.
The response should be in JSON format. For example, if the AI's answer is correct,
the response should be: {"correctness": 1, "details": "The answer is correct because..."}
"""

    eval_query = (
        f"Question: {query}\nExpected Answer: {expected_answer}\nAI's answer: {answer}\n"
        "Format the response as instructed above."
    )

    eval_response = client.post(
        "/chat",
        data=json.dumps(
            {"message": eval_query, "bot_type": bot_type, "context": eval_context}
        ),
        content_type="application/json",
    )
    eval_data = json.loads(eval_response.data)

    bot_response_str = eval_data["response"].strip()

    try:
        bot_response = json.loads(bot_response_str)
        correctness = bot_response.get("correctness", 0)
        details = bot_response.get("details", "")

        assert correctness in [
            -1,
            0,
            1,
        ], f"Bot response was '{correctness}', expected -1, 0, or 1. Details: {details}"
        if expect_success:
            assert (
                correctness == 1
            ), f"Bot response was '{correctness}', expected 1. Answer: {answer}. Expected: {expected_answer}. Details: {details}"
        else:
            assert (
                correctness == -1
            ), f"Bot response was '{correctness}', expected -1. Details: {details}"

    except pythonjson.JSONDecodeError:
        assert (
            False
        ), f"Response format is incorrect, expected a JSON formatted string. received: {bot_response_str}"
