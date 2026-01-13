from __future__ import annotations

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class ToolState(TypedDict):
    question: str
    a: int | None
    b: int | None
    result: int | None
    answer: str


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


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

    def extract_numbers(state: ToolState) -> ToolState:
        numbers = re.findall(r"-?\d+", state["question"])
        if len(numbers) >= 2:
            return {
                "question": state["question"],
                "a": int(numbers[0]),
                "b": int(numbers[1]),
                "result": None,
                "answer": "",
            }
        return {
            "question": state["question"],
            "a": None,
            "b": None,
            "result": None,
            "answer": "",
        }

    def run_tool(state: ToolState) -> ToolState:
        if state["a"] is None or state["b"] is None:
            return {**state, "result": None}
        return {**state, "result": add(state["a"], state["b"])}

    def respond(state: ToolState) -> ToolState:
        if state["result"] is None:
            return {
                **state,
                "answer": "I couldn't find two numbers to add.",
            }
        prompt = (
            "You are a helpful assistant. Use the tool result to answer.\n"
            f"Question: {state['question']}\n"
            f"Tool result: {state['result']}\n"
            "Answer in one short sentence."
        )
        response = llm.invoke(prompt)
        return {**state, "answer": response.content}

    graph = StateGraph(ToolState)
    graph.add_node("extract", extract_numbers)
    graph.add_node("tool", run_tool)
    graph.add_node("respond", respond)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "tool")
    graph.add_edge("tool", "respond")
    graph.add_edge("respond", END)

    app = graph.compile()

    prompt = "What is 19 + 7? Use the add tool."
    result = app.invoke({"question": prompt, "a": None, "b": None, "result": None, "answer": ""})
    print(result["answer"])


if __name__ == "__main__":
    main()
