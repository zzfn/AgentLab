"""
DSPy 优化器 (Optimizer) 示例
演示如何使用优化器自动改进提示词
"""

import dspy
import os
from dotenv import load_dotenv

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
    print("DSPy 优化器示例：自动优化提示词")
    print("=" * 70)

    # 步骤 1: 定义任务 Signature
    class EmotionClassifier(dspy.Signature):
        """分析文本的情感倾向"""
        text = dspy.InputField(desc="要分析的文本")
        sentiment = dspy.OutputField(desc="情感分类：积极、消极或中性")

    # 步骤 2: 准备训练数据
    print("\n📚 准备训练数据...")
    trainset = [
        dspy.Example(
            text="这个产品真的太棒了，我非常喜欢！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="质量很差，完全不值这个价格。",
            sentiment="消极"
        ).with_inputs("text"),
        dspy.Example(
            text="还可以，没有特别的感觉。",
            sentiment="中性"
        ).with_inputs("text"),
        dspy.Example(
            text="服务态度很好，体验不错！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="太失望了，再也不会买了。",
            sentiment="消极"
        ).with_inputs("text"),
    ]

    print(f"✓ 准备了 {len(trainset)} 条训练样本")

    # 步骤 3: 创建未优化的模型
    print("\n🔧 创建未优化的基础模型...")
    unoptimized_model = dspy.Predict(EmotionClassifier)

    # 测试未优化的模型
    test_text = "这家餐厅的食物很美味，环境也不错。"
    print(f"\n测试文本: {test_text}")
    result = unoptimized_model(text=test_text)
    print(f"未优化模型预测: {result.sentiment}")

    # 打印未优化的提示词
    print("\n" + "=" * 70)
    print("📋 未优化的提示词（查看 LM 历史记录）")
    print("=" * 70)
    if hasattr(lm, 'history') and lm.history:
        last_call = lm.history[-1]
        print("\n发送给模型的消息:")
        if 'messages' in last_call:
            for msg in last_call['messages']:
                print(f"\n[{msg.get('role', 'unknown')}]:")
                print(msg.get('content', ''))
        elif 'prompt' in last_call:
            print(last_call['prompt'])

    # 步骤 4: 定义评估指标
    def validate_sentiment(example, pred, trace=None):
        """评估预测是否正确"""
        return example.sentiment.strip() == pred.sentiment.strip()

    # 步骤 5: 使用优化器优化模型
    print("\n" + "=" * 70)
    print("🚀 开始优化...")
    print("=" * 70)
    print("使用 BootstrapFewShot 优化器自动生成示例和优化提示词\n")

    # 创建优化器
    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(
        metric=validate_sentiment,
        max_bootstrapped_demos=3,  # 最多生成3个示例
        max_labeled_demos=3,        # 最多使用3个标注示例
    )

    # 优化模型
    print("正在优化...")
    optimized_model = optimizer.compile(
        dspy.Predict(EmotionClassifier),
        trainset=trainset
    )
    print("✓ 优化完成！")

    # 步骤 6: 测试优化后的模型
    print("\n" + "=" * 70)
    print("🧪 测试优化后的模型")
    print("=" * 70)

    test_cases = [
        "这家餐厅的食物很美味，环境也不错。",
        "价格太贵了，性价比不高。",
        "普通的产品，没什么特别的。",
    ]

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_text}")
        result = optimized_model(text=test_text)
        print(f"优化后预测: {result.sentiment}")

    # 步骤 7: 打印优化后的提示词
    print("\n" + "=" * 70)
    print("📋 优化后的提示词（包含自动生成的示例）")
    print("=" * 70)

    if hasattr(lm, 'history') and lm.history:
        last_call = lm.history[-1]
        print("\n发送给模型的消息:")
        if 'messages' in last_call:
            for msg in last_call['messages']:
                print(f"\n[{msg.get('role', 'unknown')}]:")
                content = msg.get('content', '')
                print(content)
        elif 'prompt' in last_call:
            print(last_call['prompt'])

    # 步骤 8: 查看模型的内部状态（包含优化后的示例）
    print("\n" + "=" * 70)
    print("🔍 优化器添加的示例")
    print("=" * 70)

    if hasattr(optimized_model, 'demos') and optimized_model.demos:
        print(f"\n优化器自动添加了 {len(optimized_model.demos)} 个示例:\n")
        for i, demo in enumerate(optimized_model.demos, 1):
            print(f"示例 {i}:")
            print(f"  输入: {demo.text}")
            print(f"  输出: {demo.sentiment}")
            print()
    else:
        print("\n提示：优化后的模型可能使用了内部优化，但没有显式的 demos 属性")

    # 步骤 9: 比较说明
    print("\n" + "=" * 70)
    print("💡 优化前后的区别")
    print("=" * 70)
    print("""
优化前:
- 只有任务描述和字段说明
- 模型需要根据少量信息进行推理
- 可能不稳定或不准确

优化后:
- 自动添加了高质量的示例（Few-Shot Learning）
- 模型可以从示例中学习任务模式
- 提高了准确性和一致性
- 示例是通过 Bootstrap 方法自动生成的

BootstrapFewShot 优化器的工作原理:
1. 使用训练数据生成高质量的示例
2. 选择最有代表性的示例作为 Few-Shot 提示
3. 将这些示例添加到提示词中
4. 提高模型在该任务上的表现
    """)

if __name__ == "__main__":
    main()
