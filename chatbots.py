class ChatBot:
    def __init__(self, *args, **kwargs):
        self.model_name = 'Some chatbot'

    def get_response(self, message: str, conversation_history:list) -> str:
        """Get a response from the chatbot based on the input message."""
        raise NotImplementedError


class EchoBot(ChatBot):
    def __init__(self, *args, **kwargs):
        self.model_name = 'Echo chatbot'
    def get_response(self, message: str, conversation_history:list) -> str:
        return f"Echo: {message}. conversation history was {conversation_history}"
    

class ApiBot(ChatBot):
    def __init__(self, model_name: str, openai):
        self.openai = openai
        self.model_name = model_name

    def get_response(self, message: str, conversation_history: list) -> str:

        conversation_history.append({"role": "user", "content": message})

        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=conversation_history
        )
        reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": reply})
        return reply # type: ignore
