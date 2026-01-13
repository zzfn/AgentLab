"""
LangGraph 演示 02: 流式输出

演示如何使用 LangGraph 的流式输出功能：
- 使用 app.stream() 获取流式事件
- 实时显示 LLM 生成的内容

运行: uv run python langgraph/02_streaming.py
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY")
    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE"),
        model="glm-4.7",
        temperature=0.7,
    )


def main() -> None:
    load_dotenv()
    llm = get_llm()

    def chat_node(state: ChatState) -> dict:
        """对话节点"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 构建图
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    app = graph.compile()

    # 使用流式输出
    prompt = "写一首关于编程的短诗。"
    print(f"问题: {prompt}")
    print("=" * 40)

    input_state = {"messages": [HumanMessage(content=prompt)]}

    # stream() 会逐步返回每个节点的输出
    for event in app.stream(input_state):
        for node_name, value in event.items():
            print(f"\n[节点: {node_name}]")
            print(value["messages"][-1].content)

    print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
