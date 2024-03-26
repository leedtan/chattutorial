import json
import re
from unidecode import unidecode
import requests


def download_squad(version="2.0"):
    url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v{version}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def clean_text(text):
    cleaned_text = unidecode(text).strip()
    return cleaned_text


def prepare_data_for_openai(data):
    preparing_data = []
    conversation = {"messages": []}
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = clean_text(paragraph["context"])
            # if all(wd not in context.lower() for wd in ['beyonc', 'destiny', 'rhee']):
            #     continue
            for qa in paragraph["qas"]:
                question = clean_text(qa["question"])
                answer = (
                    clean_text(qa["answers"][0]["text"])
                    if qa["answers"]
                    else "No answer found."
                )

                conversation["messages"].append(
                    {
                        "role": "system",
                        "content": "Answer the question based on the context.",
                    }
                )
                conversation["messages"].append(
                    {
                        "role": "user",
                        "content": f"Context: {context[:2000]}\nQuestion: {question[:1000]}",
                    }
                )
                conversation["messages"].append(
                    {"role": "assistant", "content": answer[:1000]}
                )

                preparing_data.append(conversation)
                conversation = {"messages": []}

    return preparing_data


squad_data = download_squad()
prepared_data = prepare_data_for_openai(squad_data)

with open("squad_for_openai_clean.jsonl", "w") as outfile:
    for entry in prepared_data:
        json.dump(entry, outfile)
        outfile.write("\n")
