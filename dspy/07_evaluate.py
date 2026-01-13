"""
DSPy Evaluate（评估系统）示例
演示如何系统化地评估模型性能
"""

import dspy
import os
from dotenv import load_dotenv
from dspy.evaluate import Evaluate

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
    print("DSPy Evaluate: 系统化评估")
    print("=" * 70)

    # 示例 1: 基本评估 - 情感分类
    print("\n📋 示例 1: 基本评估流程")
    print("-" * 70)

    # 定义任务
    class SentimentClassification(dspy.Signature):
        """分析文本情感"""
        text = dspy.InputField(desc="要分析的文本")
        sentiment = dspy.OutputField(desc="情感：积极、消极、中性")

    # 创建模型
    sentiment_model = dspy.Predict(SentimentClassification)

    # 准备测试数据
    test_set = [
        dspy.Example(
            text="这个产品质量非常好，我很满意！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="太失望了，完全不值这个价格。",
            sentiment="消极"
        ).with_inputs("text"),
        dspy.Example(
            text="还可以，没什么特别的。",
            sentiment="中性"
        ).with_inputs("text"),
        dspy.Example(
            text="服务态度很好，物流也很快！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="质量太差了，不推荐购买。",
            sentiment="消极"
        ).with_inputs("text"),
    ]

    print(f"\n准备了 {len(test_set)} 条测试数据")

    # 定义评估指标
    def accuracy_metric(example, pred, trace=None):
        """计算准确率"""
        return example.sentiment.strip() == pred.sentiment.strip()

    # 创建评估器
    evaluator = Evaluate(
        devset=test_set,
        metric=accuracy_metric,
        num_threads=1,  # 单线程执行
        display_progress=True,
        display_table=5  # 显示前5条结果
    )

    # 运行评估
    print("\n开始评估...")
    result = evaluator(sentiment_model)
    print(f"\n✓ 准确率: {result.score:.1%}")

    # 示例 2: 多指标评估
    print("\n\n📋 示例 2: 多指标评估")
    print("-" * 70)

    class QuestionAnswer(dspy.Signature):
        """回答问题"""
        question = dspy.InputField(desc="问题")
        answer = dspy.OutputField(desc="答案")

    qa_model = dspy.ChainOfThought(QuestionAnswer)

    # 测试数据
    qa_test_set = [
        dspy.Example(
            question="Python是什么时候发布的？",
            answer="1991年"
        ).with_inputs("question"),
        dspy.Example(
            question="谁创建了Python？",
            answer="Guido van Rossum"
        ).with_inputs("question"),
        dspy.Example(
            question="DSPy是哪个大学开发的？",
            answer="斯坦福大学"
        ).with_inputs("question"),
    ]

    # 定义多个评估指标
    def exact_match(example, pred, trace=None):
        """精确匹配"""
        return example.answer.strip() in pred.answer.strip()

    def has_key_info(example, pred, trace=None):
        """包含关键信息"""
        # 检查答案是否包含关键词
        key_words = example.answer.strip().split()
        return any(word in pred.answer for word in key_words)

    def answer_length_check(example, pred, trace=None):
        """答案长度检查（不要太长）"""
        return len(pred.answer) <= 200

    # 分别评估
    print("\n评估指标 1: 精确匹配")
    evaluator1 = Evaluate(
        devset=qa_test_set,
        metric=exact_match,
        num_threads=1,
    )
    result1 = evaluator1(qa_model)
    print(f"精确匹配得分: {result1.score:.1%}")

    print("\n评估指标 2: 包含关键信息")
    evaluator2 = Evaluate(
        devset=qa_test_set,
        metric=has_key_info,
        num_threads=1,
    )
    result2 = evaluator2(qa_model)
    print(f"关键信息得分: {result2.score:.1%}")

    print("\n评估指标 3: 答案长度检查")
    evaluator3 = Evaluate(
        devset=qa_test_set,
        metric=answer_length_check,
        num_threads=1,
    )
    result3 = evaluator3(qa_model)
    print(f"长度检查得分: {result3.score:.1%}")

    # 示例 3: 评估优化前后的性能
    print("\n\n📋 示例 3: 对比优化前后的性能")
    print("-" * 70)

    # 准备更多训练数据
    trainset = [
        dspy.Example(
            text="这家餐厅太棒了！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="服务太差了。",
            sentiment="消极"
        ).with_inputs("text"),
        dspy.Example(
            text="一般般。",
            sentiment="中性"
        ).with_inputs("text"),
        dspy.Example(
            text="超级喜欢！",
            sentiment="积极"
        ).with_inputs("text"),
        dspy.Example(
            text="非常糟糕。",
            sentiment="消极"
        ).with_inputs("text"),
    ]

    # 未优化的模型
    unoptimized = dspy.Predict(SentimentClassification)

    # 使用 BootstrapFewShot 优化
    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=2,
    )

    print("\n正在优化模型...")
    optimized = optimizer.compile(
        dspy.Predict(SentimentClassification),
        trainset=trainset
    )
    print("✓ 优化完成")

    # 评估未优化模型
    print("\n评估未优化模型...")
    evaluator = Evaluate(
        devset=test_set,
        metric=accuracy_metric,
        num_threads=1,
    )
    result_before = evaluator(unoptimized)
    score_before = result_before.score
    print(f"未优化模型准确率: {score_before:.1%}")

    # 评估优化后模型
    print("\n评估优化后模型...")
    result_after = evaluator(optimized)
    score_after = result_after.score
    print(f"优化后模型准确率: {score_after:.1%}")

    # 性能提升
    improvement = score_after - score_before
    print(f"\n性能提升: {improvement:+.1%}")

    # 示例 4: 自定义复杂评估指标
    print("\n\n📋 示例 4: 自定义复杂评估指标")
    print("-" * 70)

    class TranslationTask(dspy.Signature):
        """翻译任务"""
        text = dspy.InputField(desc="英文文本")
        translation = dspy.OutputField(desc="中文翻译")

    translator = dspy.Predict(TranslationTask)

    translation_test = [
        dspy.Example(
            text="Hello, how are you?",
            translation="你好，你好吗？"
        ).with_inputs("text"),
        dspy.Example(
            text="Good morning!",
            translation="早上好！"
        ).with_inputs("text"),
    ]

    def translation_quality(example, pred, trace=None):
        """
        复杂的翻译质量评估
        考虑多个因素：长度、关键词、流畅性
        """
        score = 0.0

        # 因素1: 长度合理性（0-0.3分）
        expected_len = len(example.translation)
        actual_len = len(pred.translation)
        len_ratio = min(actual_len, expected_len) / max(actual_len, expected_len)
        score += len_ratio * 0.3

        # 因素2: 包含关键概念（0-0.4分）
        # 简化：检查是否是非空翻译
        if pred.translation and len(pred.translation) > 0:
            score += 0.4

        # 因素3: 基本正确性（0-0.3分）
        # 简化：检查是否包含中文字符
        if any('\u4e00' <= char <= '\u9fff' for char in pred.translation):
            score += 0.3

        return score

    print("\n评估翻译质量（自定义复杂指标）...")
    evaluator = Evaluate(
        devset=translation_test,
        metric=translation_quality,
        num_threads=1,
    )
    result = evaluator(translator)
    print(f"翻译质量得分: {result.score:.1%}")

    # 说明
    print("\n\n" + "=" * 70)
    print("💡 Evaluate 系统的特点")
    print("=" * 70)
    print("""
DSPy Evaluate 的核心功能:

1. **系统化评估**
   - 标准化的评估流程
   - 支持批量测试数据
   - 自动计算聚合指标

2. **评估指标 (Metric)**
   - 函数签名: metric(example, pred, trace=None) -> score
   - 返回值可以是布尔值（True/False）或浮点数（0.0-1.0）
   - 可以自定义复杂的评估逻辑

3. **常见评估指标类型**
   - 准确率 (Accuracy): 精确匹配
   - 包含检查: 是否包含关键信息
   - 格式检查: 输出格式是否正确
   - 相似度: 语义相似度、编辑距离等
   - 复合指标: 多个因素加权

4. **Evaluate 参数和返回值**
   参数:
   - devset: 测试数据集
   - metric: 评估指标函数
   - num_threads: 并行线程数
   - display_progress: 显示进度条
   - display_table: 显示结果表格

   返回值 (EvaluationResult):
   - score: 平均得分（浮点数 0.0-1.0）
   - results: 所有样本的详细结果列表

5. **实际应用场景**
   - 模型性能基准测试
   - 优化前后性能对比
   - A/B 测试不同模型
   - 持续监控模型性能
   - 发现模型弱点

6. **最佳实践**
   - 准备高质量的测试集
   - 定义清晰的评估标准
   - 使用多个指标综合评估
   - 定期评估模型性能
   - 记录评估结果用于追踪

7. **评估 + 优化的完整流程**
   ```
   1. 准备数据集（训练集 + 测试集）
   2. 定义评估指标
   3. 评估基础模型性能
   4. 使用优化器优化模型
   5. 评估优化后性能
   6. 对比分析结果
   7. 迭代改进
   ```
    """)

if __name__ == "__main__":
    main()
