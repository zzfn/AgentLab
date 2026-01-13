"""
DSPy ProgramOfThought 示例
演示如何让模型生成并执行代码来解决问题
ProgramOfThought: 模型生成代码 → 执行代码 → 得到答案
"""

import dspy
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def safe_execute(code: str) -> str:
    """
    安全执行生成的代码
    只允许基本的数学运算和数据操作
    """
    try:
        # 创建受限的命名空间
        allowed_builtins = {
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'len': len,
            'round': round,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'set': set,
            'range': range,
        }

        # 执行代码
        local_vars = {}
        exec(code, {"__builtins__": allowed_builtins}, local_vars)

        # 查找结果变量（通常是 'result' 或 'answer'）
        if 'result' in local_vars:
            return str(local_vars['result'])
        elif 'answer' in local_vars:
            return str(local_vars['answer'])
        else:
            # 返回最后一个变量的值
            if local_vars:
                last_var = list(local_vars.values())[-1]
                return str(last_var)
            return "代码执行成功，但没有返回结果"

    except Exception as e:
        return f"执行错误: {str(e)}"

def main():
    # 配置语言模型 - 使用 DeepSeek
    lm = dspy.LM(
        'deepseek/deepseek-chat',
        api_key=os.getenv('DEEPSEEK_API_KEY')
    )
    dspy.configure(lm=lm)

    print("=" * 70)
    print("DSPy ProgramOfThought: 通过代码解决问题")
    print("=" * 70)

    # 示例 1: 基础数学问题
    print("\n📋 示例 1: 基础数学计算")
    print("-" * 70)

    class MathProblem(dspy.Signature):
        """
        生成Python代码来解决数学问题
        代码应该定义一个变量 'result' 存储答案
        """
        problem = dspy.InputField(desc="数学问题描述")
        code = dspy.OutputField(desc="Python代码（必须定义result变量）")

    # 使用 ChainOfThought 生成代码
    code_generator = dspy.ChainOfThought(MathProblem)

    problem = "一个商店原价100元的商品打8折，然后用20元优惠券，最后支付多少？"
    print(f"\n问题: {problem}")

    # 生成代码
    result = code_generator(problem=problem)
    print(f"\n生成的代码:")
    print(result.code)
    print(f"\n推理过程: {result.reasoning}")

    # 执行代码
    output = safe_execute(result.code)
    print(f"\n执行结果: {output}")

    # 示例 2: 列表处理问题
    print("\n\n📋 示例 2: 数据处理")
    print("-" * 70)

    class DataProcessing(dspy.Signature):
        """生成Python代码处理数据"""
        task = dspy.InputField(desc="数据处理任务")
        code = dspy.OutputField(desc="Python代码")

    processor = dspy.Predict(DataProcessing)

    task = "有一个列表 [3, 7, 1, 9, 2, 5]，计算所有偶数的和。定义result变量存储答案。"
    print(f"\n任务: {task}")

    result = processor(task=task)
    print(f"\n生成的代码:")
    print(result.code)

    output = safe_execute(result.code)
    print(f"\n执行结果: {output}")

    # 示例 3: 完整的 ProgramOfThought 模块
    print("\n\n📋 示例 3: 自定义 ProgramOfThought 模块")
    print("-" * 70)

    class ProgramOfThought(dspy.Module):
        """
        完整的 Program of Thought 模块
        1. 理解问题
        2. 生成代码
        3. 执行代码
        4. 返回结果
        """
        def __init__(self):
            super().__init__()

            # 代码生成器
            class GenerateCode(dspy.Signature):
                """生成Python代码解决问题"""
                problem = dspy.InputField(desc="问题描述")
                reasoning = dspy.OutputField(desc="解题思路")
                code = dspy.OutputField(desc="Python代码，必须定义result变量")

            self.generate_code = dspy.ChainOfThought(GenerateCode)

        def forward(self, problem):
            # 生成代码
            generation = self.generate_code(problem=problem)

            # 执行代码
            execution_result = safe_execute(generation.code)

            return dspy.Prediction(
                reasoning=generation.reasoning,
                code=generation.code,
                result=execution_result
            )

    # 使用自定义模块
    pot = ProgramOfThought()

    problems = [
        "计算1到100的所有奇数之和",
        "有5个红球和3个蓝球，随机取2个球，两个都是红球的概率是多少？（用分数表示）",
        "斐波那契数列的第10个数是多少？（第1个是1，第2个是1）",
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n问题 {i}: {problem}")
        result = pot(problem=problem)
        print(f"\n解题思路: {result.reasoning}")
        print(f"\n生成的代码:")
        print(result.code)
        print(f"\n答案: {result.result}")
        print("-" * 70)

    # 示例 4: ProgramOfThought vs ChainOfThought 对比
    print("\n\n📋 示例 4: ProgramOfThought vs ChainOfThought")
    print("-" * 70)

    problem = "计算 15! (15的阶乘) 的值"

    # ChainOfThought
    print("\n使用 ChainOfThought:")
    cot = dspy.ChainOfThought("problem -> answer")
    result_cot = cot(problem=problem)
    print(f"推理: {result_cot.reasoning}")
    print(f"答案: {result_cot.answer}")

    # ProgramOfThought
    print("\n使用 ProgramOfThought:")
    result_pot = pot(problem=problem)
    print(f"推理: {result_pot.reasoning}")
    print(f"代码: {result_pot.code}")
    print(f"答案: {result_pot.result}")

    # 示例 5: 复杂的多步骤问题
    print("\n\n📋 示例 5: 复杂问题求解")
    print("-" * 70)

    complex_problem = """
    一个班级有30名学生，数学考试成绩如下（满分100分）：
    85, 92, 78, 95, 88, 76, 90, 83, 91, 87,
    79, 94, 86, 82, 89, 93, 77, 84, 96, 81,
    88, 92, 85, 90, 87, 83, 91, 86, 94, 89

    计算：
    1. 平均分
    2. 最高分和最低分
    3. 90分以上的人数

    将结果存储在result变量中（字典格式）
    """

    print(f"问题:\n{complex_problem}")

    result = pot(problem=complex_problem)
    print(f"\n生成的代码:")
    print(result.code)
    print(f"\n结果: {result.result}")

    # 说明
    print("\n\n" + "=" * 70)
    print("💡 ProgramOfThought 的特点和优势")
    print("=" * 70)
    print("""
ProgramOfThought (程序思维) 的核心思想:

1. **工作流程**
   问题 → 生成代码 → 执行代码 → 得到精确答案

2. **与 ChainOfThought 的区别**

   ChainOfThought:
   - 模型用自然语言推理
   - 可能出现计算错误
   - 适合需要解释的问题

   ProgramOfThought:
   - 模型生成代码
   - 代码执行确保计算准确
   - 适合需要精确计算的问题

3. **优势**
   ✓ 计算精确：避免模型的数学错误
   ✓ 复杂逻辑：处理复杂的计算和数据操作
   ✓ 可验证：代码可以审查和调试
   ✓ 可扩展：可以调用外部库

4. **适用场景**
   - 数学计算问题
   - 数据分析任务
   - 算法问题求解
   - 统计计算
   - 逻辑推理（需要精确计算）

5. **不适用场景**
   - 需要主观判断的问题
   - 创意性任务
   - 开放式问答
   - 情感分析等

6. **实现要点**
   a) 代码生成
      - 清晰的提示词
      - 指定结果变量名
      - 提供代码规范

   b) 代码执行
      - 安全沙箱环境
      - 限制危险操作
      - 错误处理

   c) 结果提取
      - 从执行环境获取结果
      - 格式化输出
      - 错误恢复

7. **安全考虑**
   ⚠️ 执行生成的代码有风险！必须：
   - 使用受限的执行环境
   - 禁止文件操作、网络访问
   - 设置超时限制
   - 验证代码内容
   - 记录所有执行

8. **最佳实践**
   - 明确指定结果变量名（如 'result'）
   - 在提示中提供代码示例
   - 添加代码验证步骤
   - 结合错误重试机制
   - 考虑代码优化

9. **实际应用示例**
   - 自动化数据分析
   - 数学辅导系统
   - 财务计算工具
   - 科学计算助手
   - 编程教学系统

10. **进阶用法**
    - 结合工具调用（如NumPy、Pandas）
    - 多步骤代码生成
    - 代码优化和重构
    - 单元测试生成
    """)

if __name__ == "__main__":
    main()
