"""
使用 LangGraph 调用智谱 AI 的测试文件
实现一个简单的 StateGraph 对话流
"""

import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 1. 定义状态 (State)
# 使用 Annotated 和 add_messages 告诉 LangGraph 如何处理消息列表的更新（追加而不是覆盖）
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 智谱 AI 的 OpenAI 兼容端点
ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"

def get_llm():
    """获取初始化后的 LLM"""
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")
    return ChatOpenAI(
        model="glm-4-flash",
        temperature=0.7,
        api_key=api_key,
        openai_api_base=ZHIPU_API_BASE
    )

# 3. 定义节点 (Nodes)
def chatbot(state: State):
    """简单的机器人节点，调用 LLM 并返回回答"""
    llm = get_llm()
    return {"messages": [llm.invoke(state["messages"])]}


# 4. 构建图 (Graph)
workflow = StateGraph(State)

# 添加节点
workflow.add_node("chatbot", chatbot)

# 添加边 (Edges)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 编译图
app = workflow.compile()


def test_simple_graph():
    """测试简单的图运行"""
    print("="*60)
    print(" LangGraph 基础对话测试 ")
    print("="*60)
    
    input_state = {"messages": [HumanMessage(content="你好，请介绍一下你自己。")]}
    
    print("\n--- 用户输入 ---")
    print(input_state["messages"][0].content)
    
    print("\n--- 正在运行 LangGraph ---")
    config = {"configurable": {"thread_id": "1"}}
    
    # 运行图并获取最终输出
    for event in app.stream(input_state, config):
        for value in event.values():
            print("\n--- 节点 [chatbot] 的输出 ---")
            print(value["messages"][-1].content)
            
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # 检查 API Key
    if not os.getenv("ZHIPUAI_API_KEY"):
        print("❌ 错误: 未设置 ZHIPUAI_API_KEY 环境变量")
    else:
        try:
            test_simple_graph()
        except Exception as e:
            print(f"❌ 运行失败: {e}")
