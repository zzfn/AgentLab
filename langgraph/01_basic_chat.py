"""
LangGraph 演示 01: 基础对话

演示 LangGraph 最基本的用法：
- 定义状态 (State)
- 创建节点 (Node)
- 构建图 (Graph)
- 编译并运行

运行: uv run python langgraph/01_basic_chat.py
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# 1. 定义状态
# 使用 Annotated + add_messages 让消息列表自动追加而不是覆盖
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
    )


def main() -> None:
    load_dotenv()
    llm = get_llm()

    # 2. 定义节点函数
    def chat_node(state: ChatState) -> dict:
        """对话节点：调用 LLM 生成回复"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 3. 构建图
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # 4. 编译并运行
    app = graph.compile()

    prompt = "用两句话介绍一下 LangGraph。"
    print(f"问题: {prompt}")
    print("-" * 40)
    
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})
    print(f"回答: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
