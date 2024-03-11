class ChatBot:
    def get_response(self, message: str) -> str:
        """Get a response from the chatbot based on the input message."""
        raise NotImplementedError


class EchoBot(ChatBot):
    def get_response(self, message: str) -> str:
        return f"I am an echo chatbot. My prompt was: {message}"
