"""
DSPy 基础示例
演示如何使用 DSPy 的基本功能
"""

import dspy
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    # 配置语言模型 - 使用 DeepSeek
    lm = dspy.LM(
        'deepseek/deepseek-chat',
        api_key=os.getenv('DEEPSEEK_API_KEY')
    )
    dspy.configure(lm=lm)

    # 示例 1: 基本的 Signature
    # Signature 定义了任务的输入和输出
    class BasicQA(dspy.Signature):
        """回答问题"""
        question = dspy.InputField(desc="用户的问题")
        answer = dspy.OutputField(desc="问题的答案")

    # 使用 Predict 模块
    qa = dspy.Predict(BasicQA)

    # 进行预测
    question_text = "什么是 DSPy？"
    response = qa(question=question_text)
    print("问题:", question_text)
    print("答案:", response.answer)
    print("-" * 50)

    # 示例 2: 简化的 Signature 语法
    # 使用字符串定义 Signature 更加简洁
    generate = dspy.ChainOfThought("question -> answer")

    question_text2 = "为什么我们需要 DSPy 这样的框架？"
    response = generate(question=question_text2)
    print("\n问题:", question_text2)
    print("推理过程:", response.reasoning)
    print("答案:", response.answer)
    print("-" * 50)

    # 示例 3: 多个输入字段
    class Translation(dspy.Signature):
        """翻译文本"""
        text = dspy.InputField(desc="要翻译的文本")
        source_lang = dspy.InputField(desc="源语言")
        target_lang = dspy.InputField(desc="目标语言")
        translation = dspy.OutputField(desc="翻译后的文本")

    translator = dspy.Predict(Translation)

    original_text = "Hello, how are you?"
    result = translator(
        text=original_text,
        source_lang="英语",
        target_lang="中文"
    )
    print("\n原文:", original_text)
    print("翻译:", result.translation)

if __name__ == "__main__":
    main()
