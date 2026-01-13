"""
DSPy ReAct (Reasoning + Acting) 示例
演示如何使用 ReAct 模式构建智能体
ReAct 结合了推理和行动，让模型能够逐步思考并采取行动
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

    print("=" * 70)
    print("DSPy ReAct 模式：推理 + 行动")
    print("=" * 70)

    # 示例 1: 带工具的 ReAct Agent
    print("\n📋 示例 1: 基础 ReAct 与工具调用")
    print("-" * 70)

    # 定义工具函数
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            # 安全的计算（只允许基本运算）
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    def search_info(query: str) -> str:
        """搜索信息（模拟知识库）"""
        knowledge = {
            "python": "Python是一种高级编程语言，由Guido van Rossum创建于1991年",
            "dspy": "DSPy是斯坦福大学开发的语言模型编程框架，用于优化提示词",
            "react": "ReAct是一种结合推理(Reasoning)和行动(Acting)的AI范式",
        }
        query_lower = query.lower()
        for key, value in knowledge.items():
            if key in query_lower:
                return value
        return "未找到相关信息"

    # 将工具转换为 DSPy 工具格式
    tools = [
        dspy.Tool(
            func=calculate,
            name="calculate",
            desc="计算数学表达式，输入格式如: 100*0.8-20"
        ),
        dspy.Tool(
            func=search_info,
            name="search_info",
            desc="搜索知识库获取信息"
        ),
    ]

    # 定义 ReAct Signature
    class Question(dspy.Signature):
        """回答问题，必要时使用工具"""
        question = dspy.InputField(desc="用户的问题")
        answer = dspy.OutputField(desc="最终答案")

    # 创建 ReAct Agent
    react_agent = dspy.ReAct(Question, tools=tools)

    # 测试问题
    questions = [
        "计算 100 * 0.8 - 20 等于多少？",
        "告诉我关于Python的信息",
        "DSPy是什么？",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n问题 {i}: {q}")
        try:
            result = react_agent(question=q)
            print(f"答案: {result.answer}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 70)

    # 示例 2: 自定义 ReAct 风格模块
    print("\n\n📋 示例 2: 模拟 ReAct 推理过程")
    print("-" * 70)

    class ReActStyle(dspy.Module):
        """
        模拟 ReAct 的思考-行动-观察循环
        虽然不是真正的 ReAct，但演示了类似的推理模式
        """
        def __init__(self):
            super().__init__()

            class ThinkAndAct(dspy.Signature):
                """逐步推理并给出答案"""
                question = dspy.InputField(desc="问题")
                thinking = dspy.OutputField(desc="思考过程（分步骤）")
                answer = dspy.OutputField(desc="最终答案")

            self.solver = dspy.ChainOfThought(ThinkAndAct)

        def forward(self, question):
            result = self.solver(question=question)
            return result

    react_style = ReActStyle()

    problem = "一个班级有30个学生，其中60%是女生，女生中又有50%戴眼镜。问戴眼镜的女生有多少人？"
    print(f"\n问题: {problem}")

    result = react_style(question=problem)
    print(f"\n思考过程:\n{result.thinking}")
    print(f"\n答案: {result.answer}")

    # 示例 3: 多步推理问题
    print("\n\n📋 示例 3: 复杂的多步推理")
    print("-" * 70)

    class ComplexReasoning(dspy.Signature):
        """需要多步推理的复杂问题"""
        problem = dspy.InputField(desc="复杂问题")
        step_by_step = dspy.OutputField(desc="逐步解答过程")
        final_answer = dspy.OutputField(desc="最终答案")

    complex_solver = dspy.ChainOfThought(ComplexReasoning)

    problem = """
    有三个盒子：
    - 盒子A：5个红球，3个蓝球
    - 盒子B：4个红球，4个蓝球
    - 盒子C：3个红球，5个蓝球

    如果随机选择一个盒子，然后从中随机取一个球，
    问：取到红球的概率是多少？
    """

    print(f"\n问题: {problem.strip()}")
    result = complex_solver(problem=problem)
    print(f"\n解答过程:\n{result.step_by_step}")
    print(f"\n最终答案: {result.final_answer}")

    # 示例 4: ReAct 的工作原理说明
    print("\n\n" + "=" * 70)
    print("💡 ReAct 模式的工作原理")
    print("=" * 70)
    print("""
ReAct (Reasoning + Acting) 模式:

1. **核心概念**
   ReAct 将推理(Reasoning)和行动(Acting)交织在一起
   循环模式: Thought → Action → Observation → Thought → ...

2. **与其他方法的对比**

   a) 标准提示 (Standard Prompting):
      Question → Answer
      - 直接回答，没有推理过程

   b) ChainOfThought (CoT):
      Question → Reasoning → Answer
      - 有推理过程，但一次性完成

   c) ReAct:
      Question → Thought₁ → Action₁ → Observation₁
              → Thought₂ → Action₂ → Observation₂
              → ... → Answer
      - 可以在推理过程中采取行动
      - 基于观察结果调整思路

3. **ReAct 的优势**
   ✓ 可以使用外部工具（计算器、搜索引擎、数据库）
   ✓ 更适合需要信息检索的任务
   ✓ 推理过程可解释、可追踪
   ✓ 可以纠正中间错误

4. **工具 (Tools) 的作用**
   - ReAct 需要定义可用的工具
   - 工具格式: dspy.Tool(func=函数, name=名称, desc=描述)
   - 模型会根据问题决定是否调用工具
   - 工具返回的结果会影响后续推理

5. **适用场景**
   ✓ 需要外部工具的任务
   ✓ 多步骤问题求解
   ✓ 需要信息检索的问答
   ✓ 复杂的决策任务
   ✓ AI Agent 系统

6. **不适用场景**
   × 简单的问答（过度设计）
   × 纯创意任务（不需要工具）
   × 实时性要求高（多次调用慢）

7. **ReAct 工作流程示例**
   ```
   用户: "今天北京的天气如何，适合户外运动吗？"

   Thought₁: 我需要先查询北京的天气
   Action₁: search_weather("北京")
   Observation₁: 北京今天晴天，温度25°C，空气质量良好

   Thought₂: 天气不错，可以判断是否适合户外运动
   Action₂: finish(answer="今天北京天气很好...")
   ```

8. **DSPy ReAct 的使用要点**
   a) 定义工具函数
   b) 创建 Tool 对象列表
   c) 创建 ReAct 模块：dspy.ReAct(signature, tools=tools)
   d) 调用模块处理问题

9. **实际应用案例**
   - 客服机器人（查询订单、检索FAQ）
   - 研究助手（搜索论文、整理信息）
   - 数据分析助手（查询数据库、执行计算）
   - 任务自动化（调用API、操作系统）

10. **最佳实践**
    - 工具描述要清晰
    - 工具数量适中（3-10个）
    - 工具功能单一明确
    - 添加错误处理
    - 记录工具调用历史
    """)

    # 示例 5: 实际建议
    print("\n" + "=" * 70)
    print("🚀 如何在实际项目中使用 ReAct")
    print("=" * 70)
    print("""
实现步骤:

1. **分析需求**
   - 任务是否需要外部工具？
   - 需要哪些工具？
   - 推理步骤是否复杂？

2. **设计工具**
   - 每个工具负责一个明确的功能
   - 工具接口简单明了
   - 添加参数验证和错误处理

3. **实现工具**
   ```python
   def my_tool(param: str) -> str:
       \"\"\"工具说明\"\"\"
       try:
           # 工具逻辑
           result = do_something(param)
           return f"结果: {result}"
       except Exception as e:
           return f"错误: {e}"
   ```

4. **创建 ReAct Agent**
   ```python
   tools = [
       dspy.Tool(func=my_tool, name="my_tool", desc="工具描述"),
       # 更多工具...
   ]

   class MyTask(dspy.Signature):
       question = dspy.InputField()
       answer = dspy.OutputField()

   agent = dspy.ReAct(MyTask, tools=tools)
   ```

5. **测试和优化**
   - 测试各种问题类型
   - 检查工具调用是否正确
   - 优化工具描述
   - 添加更多示例

6. **监控和维护**
   - 记录工具使用频率
   - 分析失败案例
   - 持续改进工具
   - 更新工具文档
    """)

if __name__ == "__main__":
    main()
