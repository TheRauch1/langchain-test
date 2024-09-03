import os

from langchain_community.llms.ollama import Ollama


class Model:
    def __init__(self) -> None:
        ollama_host_url = os.getenv("OLLAMA_HOST_URL")
        print(f"Ollama host url: {ollama_host_url}")
        self.model = Ollama(
            model="llama3.1",
            base_url=ollama_host_url,
        )

    def generate_response(self, message: str):
        return self.model.invoke(message)
