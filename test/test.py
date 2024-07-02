import pydantic
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate

chat = ChatOpenAI(
    temperature=0.5,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)
