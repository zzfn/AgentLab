import os
import time
import httpx
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

# 1. 直接用 httpx (模拟 curl)
def test_httpx():
    print("\n--- 测试 httpx (模拟 curl) ---")
    start = time.time()
    # 确保 url 正确，通常 base_url 可能已经包含了 /v1
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True 
    }
    try:
        with httpx.stream("POST", url, headers=headers, json=data, timeout=30) as r:
            ttft = None
            for line in r.iter_lines():
                if line:
                    if not ttft:
                        ttft = time.time() - start
                        print(f"TTFT: {ttft:.4f}s")
                    break
    except Exception as e:
        print(f"httpx 发生错误: {e}")

# 2. 用 LangChain
def test_langchain():
    print("\n--- 测试 LangChain ---")
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="glm-4.7")
    start = time.time()
    ttft = None
    try:
        for chunk in llm.stream("你好"):
            if not ttft:
                ttft = time.time() - start
                print(f"TTFT: {ttft:.4f}s")
            break
    except Exception as e:
        print(f"LangChain 发生错误: {e}")

# 3. 用 LangChain (强制 HTTP/1.1)
def test_langchain_http1():
    print("\n--- 测试 LangChain (强制 HTTP/1.1) ---")
    import httpx
    client = httpx.Client(http2=False)
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="glm-4.7", http_client=client)
    start = time.time()
    ttft = None
    try:
        for chunk in llm.stream("你好"):
            if not ttft:
                ttft = time.time() - start
                print(f"TTFT: {ttft:.4f}s")
            break
    except Exception as e:
        print(f"LangChain HTTP1 发生错误: {e}")

if __name__ == "__main__":
    if not api_key or not base_url:
        print("错误: 请确保环境变量 OPENAI_API_KEY 和 OPENAI_API_BASE 已设置")
    else:
        test_httpx()
        test_langchain()
        test_langchain_http1()
