"""
LangGraph 演示 04: 多节点工作流

演示如何构建包含多个顺序节点的工作流：
- 多个节点按顺序执行
- 节点之间通过状态传递数据
- 模拟一个"提取 -> 处理 -> 响应"的流程

运行: uv run python langgraph/04_multi_node_workflow.py
"""

from __future__ import annotations

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    question: str
    a: int | None
    b: int | None
    result: int | None
    answer: str


def add(a: int, b: int) -> int:
    """加法工具函数"""
    return a + b


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

    # 节点 1: 提取数字
    def extract_numbers(state: WorkflowState) -> dict:
        """从问题中提取两个数字"""
        numbers = re.findall(r"-?\d+", state["question"])
        if len(numbers) >= 2:
            return {"a": int(numbers[0]), "b": int(numbers[1])}
        return {"a": None, "b": None}

    # 节点 2: 执行计算
    def run_calculation(state: WorkflowState) -> dict:
        """执行加法计算"""
        if state["a"] is None or state["b"] is None:
            return {"result": None}
        return {"result": add(state["a"], state["b"])}

    # 节点 3: 生成回答
    def generate_response(state: WorkflowState) -> dict:
        """使用 LLM 生成自然语言回答"""
        if state["result"] is None:
            return {"answer": "抱歉，我无法从问题中找到两个数字。"}
        
        prompt = (
            f"问题: {state['question']}\n"
            f"计算结果: {state['a']} + {state['b']} = {state['result']}\n"
            "请用一句话回答这个问题。"
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    # 构建多节点工作流
    graph = StateGraph(WorkflowState)
    
    # 添加三个顺序节点
    graph.add_node("extract", extract_numbers)
    graph.add_node("calculate", run_calculation)
    graph.add_node("respond", generate_response)
    
    # 按顺序连接节点
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "calculate")
    graph.add_edge("calculate", "respond")
    graph.add_edge("respond", END)

    app = graph.compile()

    # 测试
    print("多节点工作流演示")
    print("=" * 40)
    
    question = "请问 19 加上 7 等于多少？"
    print(f"问题: {question}")
    print("-" * 40)
    
    result = app.invoke({
        "question": question,
        "a": None,
        "b": None,
        "result": None,
        "answer": ""
    })
    
    print(f"提取的数字: a={result['a']}, b={result['b']}")
    print(f"计算结果: {result['result']}")
    print(f"最终回答: {result['answer']}")


if __name__ == "__main__":
    main()
