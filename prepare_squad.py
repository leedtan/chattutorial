import json

import requests


def download_squad(version="2.0"):
    url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v{version}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def prepare_data_for_openai(data):
    prepared_data = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = (
                    qa["answers"][0]["text"] if qa["answers"] else "No answer found."
                )
                prepared_data.append(
                    {
                        "prompt": f"Context: {context}\n\nQuestion: {question}\nAnswer:",
                        "completion": f" {answer}",
                    }
                )
    return prepared_data


# Download and prepare the SQuAD data
squad_data = download_squad()
prepared_data = prepare_data_for_openai(squad_data)

# Save the prepared data to a JSONL file
with open("squad_for_openai.jsonl", "w") as outfile:
    for entry in prepared_data:
        json.dump(entry, outfile)
        outfile.write("\n")
