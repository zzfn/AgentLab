from __future__ import annotations

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class RouterState(TypedDict):
    text: str
    answer: str


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in environment.")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )


def route(state: RouterState) -> str:
    if re.search(r"\d", state["text"]):
        return "math"
    return "chat"


def solve_math(state: RouterState) -> RouterState:
    match = re.search(r"[-+*/().0-9 ]+", state["text"])
    if not match:
        return {"text": state["text"], "answer": "I can only do simple arithmetic."}
    expression = match.group(0).strip()
    if not expression:
        return {"text": state["text"], "answer": "I can only do simple arithmetic."}
    if not re.fullmatch(r"[-+*/().0-9 ]+", expression):
        return {"text": state["text"], "answer": "I can only do simple arithmetic."}
    result = eval(expression, {"__builtins__": {}})
    return {"text": state["text"], "answer": f"{expression} = {result}"}


def main() -> None:
    load_dotenv()
    llm = build_llm()

    def chat(state: RouterState) -> RouterState:
        response = llm.invoke(state["text"])
        return {"text": state["text"], "answer": response.content}

    graph = StateGraph(RouterState)
    graph.add_node("router", lambda state: state)
    graph.add_node("math", solve_math)
    graph.add_node("chat", chat)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route, {"math": "math", "chat": "chat"})
    graph.add_edge("math", END)
    graph.add_edge("chat", END)

    app = graph.compile()

    print(app.invoke({"text": "What is 12 * 8 + 5?"})["answer"])
    print(app.invoke({"text": "Give me one sentence about LangGraph."})["answer"])


if __name__ == "__main__":
    main()
