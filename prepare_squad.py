import json

import requests


def download_squad(version="2.0"):
    url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v{version}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def prepare_data_for_openai(data):
    prepared_data = []
    conversation = {"messages": []}
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = (
                    qa["answers"][0]["text"] if qa["answers"] else "No answer found."
                )

                # Add system, user, and assistant messages to the conversation
                conversation["messages"].append(
                    {
                        "role": "system",
                        "content": "Answer the question based on the context.",
                    }
                )
                conversation["messages"].append(
                    {
                        "role": "user",
                        "content": f"Context: {context}\nQuestion: {question}",
                    }
                )
                conversation["messages"].append(
                    {"role": "assistant", "content": answer}
                )

                # Add the full conversation to prepared_data and start a new conversation
                prepared_data.append(conversation)
                conversation = {"messages": []}

    return prepared_data


# Assuming you have a function download_squad to download the SQuAD data
squad_data = download_squad()
prepared_data = prepare_data_for_openai(squad_data)

# Save the prepared data to a JSONL file
with open("squad_for_openai_chat.jsonl", "w") as outfile:
    for entry in prepared_data:
        json.dump(entry, outfile)
        outfile.write("\n")
