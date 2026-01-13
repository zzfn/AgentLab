"""
DSPy 思维链 (Chain of Thought) 示例
演示如何使用 ChainOfThought 进行复杂推理
"""

import dspy
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    # 配置语言模型 - 使用 DeepSeek
    lm = dspy.LM(
        'openai/glm-4.7',
        api_key=os.getenv('OPENAI_API_KEY'),
                api_base=os.getenv('OPENAI_API_BASE'),
    )
    dspy.configure(lm=lm)

    # 示例 1: 基本的思维链
    print("=" * 60)
    print("示例 1: 基本的思维链推理")
    print("=" * 60)

    class MathProblem(dspy.Signature):
        """解决数学问题"""
        problem = dspy.InputField(desc="数学问题")
        solution = dspy.OutputField(desc="问题的解答")

    # ChainOfThought 会自动生成推理步骤
    math_solver = dspy.ChainOfThought(MathProblem)

    problem_text = "如果一个商店有 15 个苹果，卖出了 7 个，又进货 12 个，现在有多少个苹果？"
    result = math_solver(problem=problem_text)
    print("\n问题:", problem_text)
    print("\n推理步骤:", result.reasoning)
    print("\n解答:", result.solution)

    # 示例 2: 复杂的推理任务
    print("\n" + "=" * 60)
    print("示例 2: 逻辑推理")
    print("=" * 60)

    class LogicReasoning(dspy.Signature):
        """进行逻辑推理"""
        premises = dspy.InputField(desc="前提条件")
        conclusion = dspy.OutputField(desc="逻辑结论")

    reasoner = dspy.ChainOfThought(LogicReasoning)

    premises = """
    1. 所有的猫都是哺乳动物
    2. 所有的哺乳动物都需要氧气
    3. 加菲是一只猫
    """

    result = reasoner(premises=premises)
    print("\n前提:", premises)
    print("\n推理过程:", result.reasoning)
    print("\n结论:", result.conclusion)

    # 示例 3: 比较 Predict 和 ChainOfThought
    print("\n" + "=" * 60)
    print("示例 3: Predict vs ChainOfThought 对比")
    print("=" * 60)

    question = "为什么天空是蓝色的？"

    # 使用 Predict（不显示推理过程）
    simple_qa = dspy.Predict("question -> answer")
    result1 = simple_qa(question=question)

    print("\n使用 Predict:")
    print("答案:", result1.answer)

    # 使用 ChainOfThought（显示推理过程）
    cot_qa = dspy.ChainOfThought("question -> answer")
    result2 = cot_qa(question=question)

    print("\n使用 ChainOfThought:")
    print("推理过程:", result2.reasoning)
    print("答案:", result2.answer)

    # 示例 4: 自定义思维链模块
    print("\n" + "=" * 60)
    print("示例 4: 多步骤推理")
    print("=" * 60)

    class StepByStepAnalysis(dspy.Signature):
        """逐步分析问题"""
        problem = dspy.InputField(desc="需要分析的问题")
        analysis = dspy.OutputField(desc="详细的分析结果")

    analyzer = dspy.ChainOfThought(StepByStepAnalysis)

    problem_text = "一个创业公司应该如何选择合适的编程语言？"
    result = analyzer(problem=problem_text)

    print("\n问题:", problem_text)
    print("\n分析过程:", result.reasoning)
    print("\n结论:", result.analysis)

if __name__ == "__main__":
    main()
