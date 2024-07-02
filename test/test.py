import pydantic
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache


set_llm_cache(InMemoryCache())

chat = ChatOpenAI(
    temperature=0.5,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

chat.predict("How do you make italian pasta")
