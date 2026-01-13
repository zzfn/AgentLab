"""
DSPy 输出约束和验证示例
演示如何通过自定义逻辑来约束和验证模型输出
虽然 DSPy 可能没有内置的 assertions 模块，但我们可以手动实现类似功能
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
    print("DSPy 输出约束和验证")
    print("=" * 70)

    # 示例 1: 手动实现输出约束 - 长度约束
    print("\n📋 示例 1: 输出长度约束（手动实现）")
    print("-" * 70)

    class ShortSummary(dspy.Signature):
        """生成简短摘要"""
        text = dspy.InputField(desc="原文")
        summary = dspy.OutputField(desc="简短摘要（不超过50字）")

    class SummaryWithRetry(dspy.Module):
        def __init__(self, max_retries=3):
            super().__init__()
            self.generate = dspy.ChainOfThought(ShortSummary)
            self.max_retries = max_retries

        def forward(self, text):
            for attempt in range(self.max_retries):
                result = self.generate(text=text)

                # 检查长度约束
                if len(result.summary) <= 50:
                    print(f"✓ 第 {attempt + 1} 次尝试成功（{len(result.summary)}字）")
                    return result
                else:
                    print(f"✗ 第 {attempt + 1} 次尝试失败（{len(result.summary)}字，超过50字限制）")
                    if attempt < self.max_retries - 1:
                        # 修改输入，强调长度要求
                        text = f"{text}\n\n重要：摘要必须控制在50字以内，当前过长，请重新生成更简洁的版本。"

            # 如果所有尝试都失败，返回最后一次结果并截断
            print(f"⚠ 达到最大重试次数，强制截断到50字")
            result.summary = result.summary[:50] + "..."
            return result

    summary_module = SummaryWithRetry(max_retries=3)

    long_text = """
    人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    它包括机器学习、深度学习、自然语言处理等多个子领域。近年来，大语言模型的发展
    使得AI在文本理解和生成方面取得了重大突破。DSPy框架正是为了更好地编程这些
    语言模型而诞生的工具。
    """

    print(f"\n原文长度: {len(long_text.strip())}字")
    result = summary_module(text=long_text)
    print(f"\n最终摘要: {result.summary}")
    print(f"摘要长度: {len(result.summary)}字")

    # 示例 2: 格式约束
    print("\n\n📋 示例 2: 格式约束 - 确保特定格式")
    print("-" * 70)

    class StructuredResponse(dspy.Signature):
        """生成结构化的响应，必须包含'结论:'前缀"""
        question = dspy.InputField(desc="问题")
        answer = dspy.OutputField(desc="答案")

    class StructuredModule(dspy.Module):
        def __init__(self, max_retries=2):
            super().__init__()
            self.generate = dspy.Predict(StructuredResponse)
            self.max_retries = max_retries

        def forward(self, question):
            enhanced_question = f"{question}\n\n要求：答案必须以'结论:'开头。"

            for attempt in range(self.max_retries):
                result = self.generate(question=enhanced_question)

                # 检查格式
                if "结论:" in result.answer or result.answer.startswith("结论"):
                    print(f"✓ 格式正确（第 {attempt + 1} 次尝试）")
                    return result
                else:
                    print(f"✗ 格式错误（第 {attempt + 1} 次尝试）: 缺少'结论:'前缀")
                    if attempt < self.max_retries - 1:
                        enhanced_question = f"{question}\n\n严格要求：答案必须以'结论:'开头！之前的回答不符合格式要求。"

            # 如果都失败，手动添加前缀
            if not ("结论:" in result.answer or result.answer.startswith("结论")):
                result.answer = f"结论: {result.answer}"
                print(f"⚠ 手动添加'结论:'前缀")

            return result

    structured_module = StructuredModule(max_retries=2)

    question = "为什么Python很流行？"
    print(f"\n问题: {question}")
    result = structured_module(question=question)
    print(f"答案: {result.answer}")

    # 示例 3: 内容约束 - 确保是有效值
    print("\n\n📋 示例 3: 内容约束 - 确保输出是有效值")
    print("-" * 70)

    class ProductReview(dspy.Signature):
        """分析产品评论"""
        review = dspy.InputField(desc="产品评论")
        sentiment = dspy.OutputField(desc="情感分析")
        confidence = dspy.OutputField(desc="置信度百分比")

    class ReviewAnalyzer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.analyze = dspy.ChainOfThought(ProductReview)

        def forward(self, review):
            # 在提示中明确指定有效值
            enhanced_review = f"""
评论: {review}

要求：
1. sentiment 必须是以下之一: '积极'、'消极'、'中性'
2. confidence 必须是 0-100 之间的数字，格式如: 85%
"""

            result = self.analyze(review=enhanced_review)

            # 验证情感值
            valid_sentiments = ["积极", "消极", "中性"]
            if result.sentiment not in valid_sentiments:
                print(f"⚠ 情感值 '{result.sentiment}' 不在有效范围内，修正为'中性'")
                result.sentiment = "中性"

            # 验证置信度
            try:
                confidence_str = result.confidence.replace("%", "").strip()
                confidence_val = int(confidence_str)
                if not (0 <= confidence_val <= 100):
                    print(f"⚠ 置信度 {confidence_val} 超出范围，修正为 50")
                    result.confidence = "50%"
            except ValueError:
                print(f"⚠ 置信度格式错误，设置为 50%")
                result.confidence = "50%"

            return result

    review_analyzer = ReviewAnalyzer()

    reviews = [
        "这个产品质量很好，非常满意！",
        "价格太贵了，性价比不高。",
        "还行吧，没什么特别的。",
    ]

    for i, review in enumerate(reviews, 1):
        print(f"\n评论 {i}: {review}")
        result = review_analyzer(review=review)
        print(f"情感: {result.sentiment}")
        print(f"置信度: {result.confidence}")
        print(f"推理: {result.reasoning[:80]}...")

    # 示例 4: 复合验证 - 多个约束条件
    print("\n\n📋 示例 4: 复合约束 - 多条件验证")
    print("-" * 70)

    class EmailGenerator(dspy.Signature):
        """生成专业邮件"""
        topic = dspy.InputField(desc="邮件主题")
        recipient = dspy.InputField(desc="收件人")
        email = dspy.OutputField(desc="邮件内容")

    class EmailModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(EmailGenerator)

        def forward(self, topic, recipient):
            # 详细的提示词，包含所有要求
            enhanced_topic = f"""
主题: {topic}
收件人: {recipient}

要求:
1. 必须包含对 {recipient} 的称呼
2. 邮件长度在 50-300 字之间
3. 必须有礼貌的结尾（如：祝好、谢谢、期待等）
4. 语气要专业、礼貌
"""

            result = self.generate(topic=enhanced_topic, recipient=recipient)
            email_content = result.email

            # 验证各项约束
            issues = []

            # 约束1：检查称呼
            if recipient not in email_content and "您好" not in email_content:
                issues.append("缺少称呼")

            # 约束2：检查长度
            length = len(email_content)
            if length < 50:
                issues.append(f"长度过短（{length}字）")
            elif length > 300:
                issues.append(f"长度过长（{length}字）")

            # 约束3：检查结尾
            closing_phrases = ["祝好", "谢谢", "期待", "感谢", "此致", "敬礼"]
            has_closing = any(phrase in email_content for phrase in closing_phrases)
            if not has_closing:
                issues.append("缺少礼貌结尾")

            # 报告验证结果
            if issues:
                print(f"⚠ 发现问题: {', '.join(issues)}")
            else:
                print(f"✓ 所有约束都满足")

            return result

    email_module = EmailModule()

    print("\n生成专业邮件:")
    result = email_module(
        topic="项目进度汇报",
        recipient="张经理"
    )
    print(f"\n{result.email}")
    print(f"\n邮件长度: {len(result.email)}字")

    # 示例 5: 使用提示词工程强化约束
    print("\n\n📋 示例 5: 通过提示词工程强化约束")
    print("-" * 70)

    class StrictTranslation(dspy.Signature):
        """
        严格翻译任务
        要求：
        1. 翻译必须准确
        2. 必须是中文
        3. 不要添加额外解释
        """
        english = dspy.InputField(desc="英文文本")
        chinese = dspy.OutputField(desc="中文翻译（仅翻译结果，不要解释）")

    translator = dspy.Predict(StrictTranslation)

    test_texts = [
        "Hello, how are you?",
        "Good morning!",
        "Thank you very much.",
    ]

    print("\n翻译测试:")
    for text in test_texts:
        result = translator(english=text)
        print(f"\n英文: {text}")
        print(f"中文: {result.chinese}")

        # 简单验证：检查是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in result.chinese)
        if has_chinese:
            print("✓ 包含中文字符")
        else:
            print("✗ 未检测到中文字符")

    # 说明
    print("\n\n" + "=" * 70)
    print("💡 输出约束的实现策略")
    print("=" * 70)
    print("""
虽然 DSPy 可能没有内置的 Assertions 功能，但我们可以通过以下方式实现约束:

1. **提示词工程**
   - 在 Signature 的文档字符串中明确要求
   - 在 InputField 的描述中强调约束
   - 使用清晰、明确的指令

2. **后处理验证**
   - 在 Module 的 forward 方法中检查输出
   - 验证格式、长度、内容等
   - 不符合要求时进行修正

3. **重试机制**
   - 检测到违反约束时重新生成
   - 在重试时强化约束描述
   - 设置最大重试次数避免无限循环

4. **兜底策略**
   - 达到最大重试次数后的备用方案
   - 手动修正不符合要求的输出
   - 记录失败情况用于改进

5. **约束类型和实现**

   a) 长度约束:
      - 提示: "摘要不超过50字"
      - 验证: len(output) <= 50
      - 修正: output[:50] + "..."

   b) 格式约束:
      - 提示: "必须以'结论:'开头"
      - 验证: output.startswith("结论:")
      - 修正: f"结论: {output}"

   c) 值域约束:
      - 提示: "必须是'积极'、'消极'或'中性'"
      - 验证: output in valid_values
      - 修正: default_value

   d) 结构约束:
      - 提示: "必须包含称呼和结尾"
      - 验证: has_greeting and has_closing
      - 修正: 添加缺失部分

6. **最佳实践**

   ✓ 在 Signature 文档字符串中明确约束
   ✓ 在字段描述中重复约束要求
   ✓ 实现验证逻辑检查输出
   ✓ 提供友好的错误信息
   ✓ 设置合理的重试次数
   ✓ 实现兜底修正策略
   ✓ 记录约束违规情况

7. **权衡考虑**

   优点:
   - 提高输出质量和一致性
   - 确保符合业务规则
   - 增强系统可靠性

   缺点:
   - 增加代码复杂度
   - 可能需要多次 API 调用
   - 过严格可能影响创造性

8. **实际应用场景**

   - API 响应格式验证
   - 业务规则强制执行
   - 数据质量控制
   - 合规性检查
   - 用户体验优化

9. **调试和监控**

   - 记录验证失败的情况
   - 统计重试次数
   - 分析常见违规模式
   - 持续优化提示词
   - 调整约束策略

10. **进阶技巧**

    - 使用评估指标量化约束效果
    - 结合优化器改进约束满足率
    - A/B 测试不同的约束策略
    - 动态调整约束严格程度
    - 基于反馈持续改进
    """)

if __name__ == "__main__":
    main()
