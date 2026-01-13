from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class ChatState(TypedDict):
    messages: list[BaseMessage]


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment.")
    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE"),
        model="glm-4.7",
    )


def main() -> None:
    load_dotenv()
    llm = build_llm()

    def chat_node(state: ChatState) -> ChatState:
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)

    app = graph.compile()

    prompt = "Explain what LangGraph is in 2 sentences."
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
