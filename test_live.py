# from unittest.mock import patch

# import pytest
# from flask import json

# from app import app as flask_app


# @pytest.fixture
# def client():
#     flask_app.config["TESTING"] = True
#     with flask_app.test_client() as client:
#         yield client


# @patch("chatbots.ApiBot.get_response")
# def test_chat_post_success_with_mock(mock_get_response, client):
#     mock_response = "This is a mocked response."
#     mock_get_response.return_value = mock_response
#     response = client.post(
#         "/chat", data=json.dumps({"message": "Hello"}), content_type="application/json"
#     )

#     assert response.status_code == 200
#     data = json.loads(response.data)
#     assert data["response"] == mock_response


# def test_chat_post_success_with_live_api(client):

#     response = client.post(
#         "/chat", data=json.dumps({"message": "Hello"}), content_type="application/json"
#     )
#     assert response.status_code == 200
#     data = json.loads(response.data)
#     assert data["response"] != ""
