import logging
import os

import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
# File handler for logging
file_handler = logging.FileHandler("fine_tuning.log")
logger.addHandler(file_handler)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    from keys import OPENAI_API_KEY

    openai.api_key = OPENAI_API_KEY

client = openai.Client(api_key=openai.api_key)


def fine_tune_model(training_file_id):
    # response = client.fine_tunes.create(
    #     training_file_id=training_file_id,
    #     model="gpt-3.5-turbo",
    #     n_epochs=3,
    # )

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id, model="gpt-3.5-turbo"
    )
    return response


def upload_file(file_path):
    with open(file_path, "rb") as file:
        response = client.files.create(file=file, purpose="fine-tune")
        return response.id


try:
    file_id = upload_file("squad_for_openai_chat.jsonl")
    logger.info(f"Uploaded file ID: {file_id}")

    fine_tuning_response = fine_tune_model(file_id)
    logger.info(f"Fine tuning started: {fine_tuning_response}")

except Exception as e:
    logger.error(f"Error during file upload or fine-tuning: {e}")
