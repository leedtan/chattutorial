class ChatBot:
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
    def __init__(self):
        self.model_name = "Echo chatbot"

    def generate_response(self, message: str, conversation_history: list) -> str:
        return f"Echo: {message}. Conversation history was {conversation_history}"


class ApiBot(ChatBot):
    def __init__(self, model_name: str, openai):
        self.openai = openai
        self.model_name = model_name

    def generate_response(self, message: str, conversation_history: list) -> str:
        response = self.openai.chat.completions.create(
            model=self.model_name, messages=conversation_history
        )
        return response.choices[0].message.content
