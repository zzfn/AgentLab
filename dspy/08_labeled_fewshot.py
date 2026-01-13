"""
DSPy LabeledFewShot 优化器示例
演示如何使用标注数据作为 Few-Shot 示例来优化模型
LabeledFewShot 是最简单直接的优化器，直接使用标注数据作为示例
"""

import dspy
import os
from dotenv import load_dotenv
from dspy.teleprompt import LabeledFewShot

# 加载环境变量
load_dotenv()

def main():
    # 配置语言模型 - 使用环境变量
    lm = dspy.LM(
        'openai/glm-4.7',
        api_key=os.getenv('OPENAI_API_KEY')
    )
    dspy.configure(lm=lm)

    print("=" * 70)
    print("DSPy LabeledFewShot: 使用标注数据优化")
    print("=" * 70)

    # 示例 1: 基本用法 - 情感分类
    print("\n📋 示例 1: 基础 LabeledFewShot")
    print("-" * 70)

    class SentimentAnalysis(dspy.Signature):
        """分析文本情感"""
        text = dspy.InputField(desc="评论文本")
        sentiment = dspy.OutputField(desc="情感：积极、消极或中性")

    # 准备标注数据（作为示例）
    labeled_examples = [
        dspy.Example(
            text="这个产品质量非常好，我很满意！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="太失望了，完全不值这个价格。",
            sentiment="消极"
        ).with_inputs("text"),
        dspy.Example(
            text="还可以，没什么特别的感觉。",
            sentiment="中性"
        ).with_inputs("text"),
        dspy.Example(
            text="物流很快，包装也很好，推荐购买！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="质量太差了，用了一天就坏了。",
            sentiment="消极"
        ).with_inputs("text"),
    ]

    print(f"\n准备了 {len(labeled_examples)} 个标注示例")

    # 创建未优化的模型
    print("\n测试未优化模型:")
    base_model = dspy.Predict(SentimentAnalysis)
    test_text = "服务态度很好，但价格有点贵。"
    result = base_model(text=test_text)
    print(f"输入: {test_text}")
    print(f"预测: {result.sentiment}")

    # 使用 LabeledFewShot 优化
    print("\n" + "-" * 70)
    print("使用 LabeledFewShot 优化（选择3个示例）...")

    optimizer = LabeledFewShot(k=3)  # k=3 表示使用3个示例

    optimized_model = optimizer.compile(
        student=dspy.Predict(SentimentAnalysis),
        trainset=labeled_examples
    )

    print("✓ 优化完成")

    # 测试优化后的模型
    print("\n测试优化后模型:")
    result = optimized_model(text=test_text)
    print(f"输入: {test_text}")
    print(f"预测: {result.sentiment}")

    # 查看添加的示例
    if hasattr(optimized_model, 'demos'):
        print(f"\n优化器添加了 {len(optimized_model.demos)} 个示例:")
        for i, demo in enumerate(optimized_model.demos, 1):
            print(f"\n示例 {i}:")
            print(f"  文本: {demo.text}")
            print(f"  情感: {demo.sentiment}")

    # 示例 2: 对比不同 k 值的效果
    print("\n\n📋 示例 2: 不同 k 值的影响")
    print("-" * 70)

    test_cases = [
        "这家餐厅环境很好，菜品也不错。",
        "价格太高，性价比很低。",
        "普通水平，没有惊喜。",
    ]

    for k_value in [0, 2, 5]:
        print(f"\n使用 k={k_value} 个示例:")

        if k_value == 0:
            model = dspy.Predict(SentimentAnalysis)
            print("（未使用 Few-Shot 示例）")
        else:
            optimizer = LabeledFewShot(k=k_value)
            model = optimizer.compile(
                student=dspy.Predict(SentimentAnalysis),
                trainset=labeled_examples
            )

        for test_text in test_cases:
            result = model(text=test_text)
            print(f"  '{test_text}' → {result.sentiment}")

    # 示例 3: 更复杂的任务 - 问答
    print("\n\n📋 示例 3: 问答任务的 LabeledFewShot")
    print("-" * 70)

    class QuestionAnswer(dspy.Signature):
        """回答关于编程的问题"""
        question = dspy.InputField(desc="问题")
        answer = dspy.OutputField(desc="简洁的答案")

    # 准备标注的问答对
    qa_examples = [
        dspy.Example(
            question="什么是变量？",
            answer="变量是用来存储数据的容器，可以保存不同类型的值。"
        ).with_inputs("question"),
        dspy.Example(
            question="什么是函数？",
            answer="函数是一段可重复使用的代码块，接受输入参数并返回结果。"
        ).with_inputs("question"),
        dspy.Example(
            question="什么是循环？",
            answer="循环是重复执行代码的结构，直到满足某个条件为止。"
        ).with_inputs("question"),
        dspy.Example(
            question="什么是类？",
            answer="类是对象的蓝图，定义了对象的属性和方法。"
        ).with_inputs("question"),
    ]

    print(f"准备了 {len(qa_examples)} 个问答示例")

    # 优化问答模型
    print("\n优化问答模型...")
    qa_optimizer = LabeledFewShot(k=3)
    qa_model = qa_optimizer.compile(
        student=dspy.ChainOfThought(QuestionAnswer),
        trainset=qa_examples
    )
    print("✓ 优化完成")

    # 测试
    print("\n测试问答:")
    test_questions = [
        "什么是数组？",
        "什么是递归？",
    ]

    for question in test_questions:
        result = qa_model(question=question)
        print(f"\n问题: {question}")
        print(f"答案: {result.answer}")

    # 示例 4: LabeledFewShot vs BootstrapFewShot
    print("\n\n📋 示例 4: LabeledFewShot vs BootstrapFewShot 对比")
    print("-" * 70)

    print("""
LabeledFewShot 特点:
✓ 直接使用标注数据作为示例
✓ 简单、快速、不需要额外的 LM 调用
✓ 随机或按顺序选择 k 个示例
✓ 适合数据质量高的场景

BootstrapFewShot 特点:
✓ 动态生成示例（使用模型自己生成）
✓ 选择高质量的示例（基于评估指标）
✓ 需要额外的 LM 调用（成本更高）
✓ 可能获得更好的性能

选择建议:
- 如果有高质量的标注数据 → 使用 LabeledFewShot
- 如果需要更好的性能且不在意成本 → 使用 BootstrapFewShot
- 可以先尝试 LabeledFewShot，效果不好再用 BootstrapFewShot
    """)

    # 示例 5: 自定义选择策略
    print("\n\n📋 示例 5: 查看 LabeledFewShot 的内部机制")
    print("-" * 70)

    print("""
LabeledFewShot 的工作原理:

1. **输入**:
   - student: 要优化的模型（如 dspy.Predict）
   - trainset: 标注数据集
   - k: 选择几个示例

2. **处理过程**:
   - 从 trainset 中选择 k 个示例
   - 默认随机选择（可以配置）
   - 将这些示例添加到模型的上下文中

3. **输出**:
   - 返回优化后的模型
   - 模型内部包含选中的示例

4. **提示词结构**:
   ```
   [任务描述]

   示例 1:
   输入: ...
   输出: ...

   示例 2:
   输入: ...
   输出: ...

   ...

   现在回答:
   输入: [用户的实际输入]
   输出:
   ```

5. **优点**:
   - 实现简单
   - 无需额外计算
   - 效果立竿见影
   - 可控性强

6. **局限**:
   - 示例选择可能不是最优的
   - 不会动态调整示例
   - 依赖标注数据的质量
    """)

    # 实际应用建议
    print("\n" + "=" * 70)
    print("💡 实际应用建议")
    print("=" * 70)
    print("""
1. **准备高质量示例**
   - 覆盖不同场景
   - 答案准确且格式统一
   - 避免噪声和错误

2. **选择合适的 k 值**
   - 太少（k=1-2）: 可能不够代表性
   - 适中（k=3-5）: 通常效果较好
   - 太多（k>10）: 可能超出上下文限制

3. **示例质量 > 数量**
   - 3个高质量示例 > 10个低质量示例
   - 确保示例多样性
   - 覆盖边界情况

4. **结合评估使用**
   - 评估不同 k 值的效果
   - 对比优化前后的性能
   - 迭代改进示例集

5. **何时使用 LabeledFewShot**
   ✓ 有标注数据
   ✓ 需要快速原型
   ✓ 成本敏感
   ✓ 数据质量高

6. **何时考虑其他优化器**
   × 标注数据不够好
   × 需要最佳性能
   × 有充足的计算预算
    """)

if __name__ == "__main__":
    main()
