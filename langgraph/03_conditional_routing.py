"""
LangGraph 演示 03: 条件路由

演示如何使用条件边进行路由：
- add_conditional_edges() 根据条件选择不同的节点
- 实现一个简单的数学/对话路由器

运行: uv run python langgraph/03_conditional_routing.py
"""

from __future__ import annotations

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class RouterState(TypedDict):
    text: str
    answer: str


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


def route_by_content(state: RouterState) -> str:
    """路由函数：根据内容决定走哪个分支"""
    if re.search(r"\d", state["text"]):
        return "math"
    return "chat"


def solve_math(state: RouterState) -> dict:
    """数学计算节点：使用 eval 计算简单表达式"""
    match = re.search(r"[-+*/().0-9 ]+", state["text"])
    if not match:
        return {"answer": "无法识别数学表达式"}
    
    expression = match.group(0).strip()
    if not expression or not re.fullmatch(r"[-+*/().0-9 ]+", expression):
        return {"answer": "无法识别数学表达式"}
    
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"answer": f"{expression} = {result}"}
    except Exception:
        return {"answer": "计算出错"}


def main() -> None:
    load_dotenv()
    llm = get_llm()

    def chat_node(state: RouterState) -> dict:
        """对话节点：调用 LLM"""
        response = llm.invoke(state["text"])
        return {"answer": response.content}

    # 构建带条件路由的图
    graph = StateGraph(RouterState)
    
    # 添加节点
    graph.add_node("math", solve_math)
    graph.add_node("chat", chat_node)
    
    # 添加条件边：根据 route_by_content 的返回值选择节点
    graph.add_conditional_edges(
        START,
        route_by_content,
        {"math": "math", "chat": "chat"}
    )
    
    # 两个节点都连接到 END
    graph.add_edge("math", END)
    graph.add_edge("chat", END)

    app = graph.compile()

    # 测试数学问题
    print("测试 1: 数学问题")
    print("-" * 40)
    result = app.invoke({"text": "12 * 8 + 5 等于多少？", "answer": ""})
    print(f"输入: 12 * 8 + 5 等于多少？")
    print(f"输出: {result['answer']}")
    
    print("\n测试 2: 对话问题")
    print("-" * 40)
    result = app.invoke({"text": "用一句话介绍 LangGraph", "answer": ""})
    print(f"输入: 用一句话介绍 LangGraph")
    print(f"输出: {result['answer']}")


if __name__ == "__main__":
    main()
