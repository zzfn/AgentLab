"""
DSPy RAG (Retrieval Augmented Generation) 示例
演示如何构建检索增强生成系统
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
        api_key=os.getenv('OPENAI_API_KEY')
    )
    dspy.configure(lm=lm)

    # 示例 1: 简单的上下文增强回答
    print("=" * 60)
    print("示例 1: 基于上下文的问答")
    print("=" * 60)

    class ContextQA(dspy.Signature):
        """基于给定上下文回答问题"""
        context = dspy.InputField(desc="背景知识")
        question = dspy.InputField(desc="问题")
        answer = dspy.OutputField(desc="基于上下文的答案")

    qa_with_context = dspy.ChainOfThought(ContextQA)

    # 模拟检索到的文档
    context = """
    DSPy 是斯坦福大学开发的一个框架，用于编程语言模型。
    它的核心思想是将提示工程转变为更系统化的过程。
    DSPy 提供了 Signature、Module 和 Optimizer 等核心抽象。
    使用 DSPy，开发者可以声明式地定义任务，而不是手工编写提示。
    """

    question_text = "DSPy 的核心思想是什么？"
    result = qa_with_context(
        context=context,
        question=question_text
    )

    print("\n上下文:", context.strip())
    print("\n问题:", question_text)
    print("\n推理:", result.reasoning)
    print("\n答案:", result.answer)

    # 示例 2: 简化的 RAG 模块
    print("\n" + "=" * 60)
    print("示例 2: 自定义 RAG 模块")
    print("=" * 60)

    class SimpleRAG(dspy.Module):
        def __init__(self):
            super().__init__()
            # 使用 ChainOfThought 进行推理
            self.generate_answer = dspy.ChainOfThought(ContextQA)

        def forward(self, question, documents):
            # 将文档组合成上下文
            context = "\n\n".join(documents)

            # 生成答案
            result = self.generate_answer(context=context, question=question)

            return dspy.Prediction(
                context=context,
                answer=result.answer,
                reasoning=result.reasoning
            )

    # 模拟知识库
    knowledge_base = [
        "Python 是一种高级编程语言，由 Guido van Rossum 在 1991 年发布。它强调代码可读性。",
        "Python 广泛用于数据科学、机器学习、Web 开发和自动化任务。",
        "Python 有丰富的标准库和第三方包生态系统，如 NumPy、Pandas 和 Django。",
    ]

    rag = SimpleRAG()

    result = rag(
        question="Python 的主要应用领域有哪些？",
        documents=knowledge_base
    )

    print("\n问题: Python 的主要应用领域有哪些？")
    print("\n检索到的文档:")
    for i, doc in enumerate(knowledge_base, 1):
        print(f"{i}. {doc}")
    print("\n推理过程:", result.reasoning)
    print("\n答案:", result.answer)

    # 示例 3: 多跳推理 RAG
    print("\n" + "=" * 60)
    print("示例 3: 多跳推理（需要综合多个文档）")
    print("=" * 60)

    class MultiHopQA(dspy.Signature):
        """需要综合多个信息源的问答"""
        context = dspy.InputField(desc="多个相关文档")
        question = dspy.InputField(desc="需要综合分析的问题")
        answer = dspy.OutputField(desc="综合答案")
        supporting_facts = dspy.OutputField(desc="支持答案的关键事实")

    multi_hop_qa = dspy.ChainOfThought(MultiHopQA)

    documents = """
    文档1: 机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。
    文档2: DSPy 是一个用于编程语言模型的框架，它使用机器学习来优化提示。
    文档3: 语言模型可以通过 DSPy 进行系统化的优化和改进。
    """

    question_text2 = "DSPy 和机器学习之间有什么关系？"
    result = multi_hop_qa(
        context=documents,
        question=question_text2
    )

    print("\n文档:", documents.strip())
    print("\n问题:", question_text2)
    print("\n推理:", result.reasoning)
    print("\n答案:", result.answer)
    print("\n支持事实:", result.supporting_facts)

    # 示例 4: 实际应用建议
    print("\n" + "=" * 60)
    print("提示: 在实际应用中，你需要:")
    print("=" * 60)
    print("""
    1. 集成真实的检索系统（如向量数据库）
       - 使用 ChromaDB、Pinecone 或 Weaviate
       - 实现语义搜索来检索相关文档

    2. 配置 DSPy 的 Retriever
       - dspy.Retrieve() 可以与各种检索后端集成

    3. 优化检索和生成
       - 使用 DSPy 的优化器（如 BootstrapFewShot）
       - 评估和迭代改进系统性能

    4. 示例代码:
       ```python
       # 配置检索器
       colbertv2 = dspy.ColBERTv2(url='http://your-server')
       dspy.settings.configure(rm=colbertv2)

       # 使用内置的 RAG 模块
       class RAG(dspy.Module):
           def __init__(self, num_passages=3):
               super().__init__()
               self.retrieve = dspy.Retrieve(k=num_passages)
               self.generate = dspy.ChainOfThought(ContextQA)

           def forward(self, question):
               context = self.retrieve(question).passages
               return self.generate(context=context, question=question)
       ```
    """)

if __name__ == "__main__":
    main()
