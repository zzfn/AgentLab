"""
使用 LangChain 调用智谱 AI 的测试文件（使用 OpenAI 兼容协议 + LCEL 语法）
使用前需要安装: uv add langchain langchain-openai
设置环境变量: export OPENAI_API_KEY="your_api_key"
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# OpenAI 兼容端点
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")


def get_chat_model(temperature=0.7, streaming=False):
    """创建聊天模型"""
    return ChatOpenAI(
        model="glm-4.7",
        temperature=temperature,
        streaming=streaming,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=OPENAI_API_BASE
    )


def test_basic_chat():
    """测试基本对话"""
    print("="*60)
    print("1. 测试基本对话功能")
    print("="*60)
    
    chat = get_chat_model()
    
    # 发送消息
    messages = [
        SystemMessage(content="你是一个有帮助的 AI 助手。"),
        HumanMessage(content="请用一句话介绍 LangChain。")
    ]
    
    response = chat.invoke(messages)
    print(f"\n问题: 请用一句话介绍 LangChain。")
    print(f"回答: {response.content}\n")


def test_prompt_template():
    """测试 Prompt 模板（使用 LCEL）"""
    print("="*60)
    print("2. 测试 Prompt 模板 (LCEL)")
    print("="*60)
    
    chat = get_chat_model()
    
    # 创建 Prompt 模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}。"),
        ("human", "{question}")
    ])
    
    # 使用 LCEL 创建链 (LangChain Expression Language)
    chain = template | chat | StrOutputParser()
    
    # 调用
    response = chain.invoke({
        "role": "Python 编程专家",
        "question": "如何使用列表推导式生成 1-10 的平方数列表？"
    })
    
    print(f"\n角色: Python 编程专家")
    print(f"问题: 如何使用列表推导式生成 1-10 的平方数列表？")
    print(f"回答: {response}\n")


def test_chain_with_lcel():
    """测试链式调用（使用 LCEL）"""
    print("="*60)
    print("3. 测试链式调用 (LCEL)")
    print("="*60)
    
    chat = get_chat_model(temperature=0.9)
    
    # 创建提示模板
    prompt = PromptTemplate.from_template("为{product}写一个有创意的广告语。只输出广告语，不要其他内容。")
    
    # 使用 LCEL 创建链
    chain = prompt | chat | StrOutputParser()
    
    # 运行链
    result = chain.invoke({"product": "智能手表"})
    print(f"\n产品: 智能手表")
    print(f"广告语: {result}\n")


def test_conversation_with_history():
    """测试对话记忆（使用 LCEL）"""
    print("="*60)
    print("4. 测试对话记忆 (LCEL)")
    print("="*60)
    
    chat = get_chat_model()
    
    # 使用消息历史进行对话
    messages = []
    
    # 第一轮对话
    messages.append(HumanMessage(content="我叫张三，我喜欢编程。"))
    response1 = chat.invoke(messages)
    messages.append(AIMessage(content=response1.content))
    
    print(f"\n第1轮 - 用户: 我叫张三，我喜欢编程。")
    print(f"第1轮 - AI: {response1.content}")
    
    # 第二轮对话（测试记忆）
    messages.append(HumanMessage(content="你还记得我的名字吗？我喜欢什么？"))
    response2 = chat.invoke(messages)
    
    print(f"\n第2轮 - 用户: 你还记得我的名字吗？我喜欢什么？")
    print(f"第2轮 - AI: {response2.content}\n")


def test_streaming():
    """测试流式输出"""
    print("="*60)
    print("5. 测试流式输出")
    print("="*60)
    
    chat = get_chat_model(streaming=True)
    
    messages = [
        SystemMessage(content="你是一个诗人。"),
        HumanMessage(content="写一首关于月光的短诗。")
    ]
    
    print("\n流式输出:")
    print("-"*60)
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-"*60 + "\n")


def test_multiple_chains():
    """测试多个链的组合"""
    print("="*60)
    print("6. 测试多个链的组合")
    print("="*60)
    
    chat = get_chat_model()
    
    # 第一个链：生成产品特点
    feature_prompt = PromptTemplate.from_template(
        "为{product}列出3个主要特点，用逗号分隔。只输出特点，不要其他内容。"
    )
    feature_chain = feature_prompt | chat | StrOutputParser()
    
    # 第二个链：根据特点生成广告语
    slogan_prompt = PromptTemplate.from_template(
        "根据这些产品特点：{features}，创作一个吸引人的广告语。"
    )
    slogan_chain = slogan_prompt | chat | StrOutputParser()
    
    # 组合链
    product = "智能手表"
    features = feature_chain.invoke({"product": product})
    slogan = slogan_chain.invoke({"features": features})
    
    print(f"\n产品: {product}")
    print(f"特点: {features}")
    print(f"广告语: {slogan}\n")


def test_batch_processing():
    """测试批量处理"""
    print("="*60)
    print("7. 测试批量处理")
    print("="*60)
    
    chat = get_chat_model()
    
    # 创建简单的链
    prompt = PromptTemplate.from_template("用一句话描述{topic}。")
    chain = prompt | chat | StrOutputParser()
    
    # 批量处理多个输入
    topics = [
        {"topic": "人工智能"},
        {"topic": "区块链"},
        {"topic": "量子计算"}
    ]
    
    print("\n批量处理结果:")
    results = chain.batch(topics)
    
    for i, (topic, result) in enumerate(zip(topics, results), 1):
        print(f"\n{i}. {topic['topic']}: {result}")
    
    print()


def main():
    """主函数"""
    print("\n" + "="*60)
    print(" "*10 + "LangChain + 智谱 AI 测试 (LCEL)")
    print("="*60 + "\n")
    
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 OPENAI_API_KEY 环境变量")
        print("\n请执行:")
        print('  export OPENAI_API_KEY="your_api_key"')
        return
    
    try:
        # 运行所有测试
        test_basic_chat()
        test_prompt_template()
        test_chain_with_lcel()
        test_conversation_with_history()
        test_streaming()
        test_multiple_chains()
        test_batch_processing()
        
        print("="*60)
        print("✅ 所有测试完成！")
        print("="*60)
        print("\n说明:")
        print("- 使用 LCEL (LangChain Expression Language) 语法")
        print("- 使用 | 管道操作符链接组件")
        print("- 兼容 LangChain 最新版本")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保:")
        print("1. 已安装依赖: uv add langchain langchain-openai")
        print("2. API Key 正确且有效")
        print("3. 网络连接正常")


if __name__ == "__main__":
    main()
