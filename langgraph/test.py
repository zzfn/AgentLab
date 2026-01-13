import os

from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage

# 创建带流式输出的 LLM
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 发送消息（输出会实时流式显示）
response = llm([HumanMessage(content="写一首关于春天的诗")])