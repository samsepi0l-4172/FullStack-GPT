from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
OpenaiAPIKey: str | None = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(
    api_key=OpenaiAPIKey,
    temperature=0.5,
)
with get_openai_callback() as usage:
    chat.invoke("What is the recipe for soju")
    print(usage)
