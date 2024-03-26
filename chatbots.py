import subprocess
import json

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory


class ChatBot:
    """
    Chatbot that builds message into conversation history
    """

    def __init__(self, *args, **kwargs):
        self.model_name = "Some chatbot"

    def get_response(self, message: str, conversation_history: list) -> str:
        """Update conversation history and get a response from the chatbot."""
        conversation_history.append({"role": "user", "content": message})
        response = self.generate_response(message, conversation_history)
        conversation_history.append({"role": "assistant", "content": response})
        return response

    def generate_response(self, message: str, conversation_history: list) -> str:
        """Generate a response based on the input message. To be implemented by subclasses."""
        raise NotImplementedError


class EchoBot(ChatBot):
    """
    Simple chatbot that echoes the input message.
    """

    def __init__(self):
        self.model_name = "Echo chatbot"

    def generate_response(self, message: str, conversation_history: list) -> str:
        return f"Echo: {message}. Conversation history was {conversation_history}"


class ApiBot(ChatBot):
    """
    Chatbot using OpenAI's API.
    """

    def __init__(self, model_name: str, openai):
        self.openai = openai
        self.model_name = model_name

    def generate_response(
        self, message: str, conversation_history: list, llm_settings={}
    ) -> str:
        response = self.openai.chat.completions.create(
            model=self.model_name, messages=conversation_history, **llm_settings
        )
        return response.choices[0].message.content


class FineTuneBot(ChatBot):
    """
    Chatbot using a fine-tuned model from OpenAI's API.
    """

    def __init__(self, openai):
        super().__init__()
        self.openai = openai
        self.model_name = self.get_latest_finetuned_model()

    def get_latest_finetuned_model(self):
        """
        Get the latest fine-tuned model from OpenAI's API.
        Messy for now due to hacks I'm putting in as we debug and measure and iterate on measuring data prep practices
        """
        fine_tunes = self.openai.client.fine_tuning.jobs.list()
        winner = None
        for job in fine_tunes.data:
            if job.status == "succeeded":
                file_id = job.training_file
                file_details = self.openai.files.retrieve(file_id)
                winner = job.fine_tuned_model
                if file_details.filename in [
                    "squad_for_openai_chat_clip.jsonl",
                ]:
                    return winner
        if winner is not None:
            return winner
        return "no fine tuned models found"

    def generate_response(self, message: str, conversation_history: list) -> str:
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=conversation_history,
            max_tokens=3000,
            stop=None,
            temperature=0.0,
        )
        return response.choices[0].message.content


class LocalLLMBot(ChatBot):
    """
    Local chatbot using Hugging Face's transformers library.
    """

    def __init__(
        self, model_name="microsoft/DialoGPT-medium", model_path="./models/dialogpt/"
    ):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self._setup_model()

    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def generate_response(self, message: str, conversation_history: list) -> str:
        input_text = ""
        for turn in conversation_history:
            input_text += f"{turn['role']}: {turn['content']}\n"
        input_text += "assistant:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=1000, num_return_sequences=1)
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "assistant:" in response_text:
            assistant_response = response_text.split("assistant:")[-1].strip()
        else:
            assistant_response = response_text.strip()
        return json.dumps({"response": assistant_response})


class RAGBot(ChatBot):
    def __init__(self, llm_repo_id, web_loader_url, embedding_model_name, hf_api_key):
        super().__init__()
        self.model_name = llm_repo_id
        self.loader = WebBaseLoader(web_loader_url)
        self.embedding_model_name = embedding_model_name
        self._hf_api_key = hf_api_key
        self._setup_chain()

    def _setup_chain(self):
        data = self.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        hf_embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        vectorstore = FAISS.from_documents(
            documents=all_splits, embedding=hf_embeddings
        )

        llm = HuggingFaceHub(
            repo_id=self.model_name, huggingfacehub_api_token=self._hf_api_key
        )
        memory = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", return_messages=True
        )

        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory, verbose=True
        )

    def generate_response(self, message: str, conversation_history: list) -> str:
        response = self.chat_chain(message)
        return json.dumps({"response": response})
