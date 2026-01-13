"""
智谱 AI API 测试文件
使用前需要安装: uv add zhipuai
设置环境变量: export ZHIPUAI_API_KEY="your_api_key"
"""

import os
from zhipuai import ZhipuAI


def test_chat():
    """测试智谱 AI 聊天功能"""
    # 初始化客户端
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")
    
    client = ZhipuAI(api_key=api_key)
    
    # 发送聊天请求
    response = client.chat.completions.create(
        model="glm-4-flash",  # 使用 glm-4-flash 模型
        messages=[
            {"role": "user", "content": "你好,请介绍一下你自己"}
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    
    # 打印响应
    print("AI 回复:")
    print(response.choices[0].message.content)
    print("\n" + "="*50)
    print(f"使用的模型: {response.model}")
    print(f"Token 使用情况: {response.usage}")


def test_stream_chat():
    """测试智谱 AI 流式聊天功能"""
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")
    
    client = ZhipuAI(api_key=api_key)
    
    # 发送流式聊天请求
    print("流式输出:")
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "user", "content": "写一首关于春天的诗"}
        ],
        stream=True,
    )
    
    # 逐个输出流式响应
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


def test_function_call():
    """测试智谱 AI 函数调用功能"""
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")
    
    client = ZhipuAI(api_key=api_key)
    
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    # 发送带工具的请求
    messages = [{"role": "user", "content": "北京今天天气怎么样?"}]
    
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        tools=tools,
    )
    
    print("AI 响应:")
    print(f"完成原因: {response.choices[0].finish_reason}")
    
    if response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message.tool_calls
        print(f"调用的工具: {tool_calls[0].function.name}")
        print(f"参数: {tool_calls[0].function.arguments}")


if __name__ == "__main__":
    print("="*50)
    print("智谱 AI API 测试")
    print("="*50 + "\n")
    
    try:
        # 测试基本聊天
        print("1. 测试基本聊天功能")
        print("-"*50)
        test_chat()
        print("\n")
        
        # 测试流式聊天
        print("2. 测试流式聊天功能")
        print("-"*50)
        test_stream_chat()
        print("\n")
        
        # 测试函数调用
        print("3. 测试函数调用功能")
        print("-"*50)
        test_function_call()
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. 已安装 zhipuai 包: uv add zhipuai")
        print("2. 已设置环境变量: export ZHIPUAI_API_KEY='your_api_key'")
        print("3. API Key 有效且有足够的额度")
