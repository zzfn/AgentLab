# DSPy 学习示例目录

本目录包含了 DSPy 框架的完整学习示例，从基础到高级，涵盖了主要特性。

## 📚 示例列表

### 基础部分

#### 01_basic.py - DSPy 基础
**学习内容：**
- Signature 定义（任务的输入输出）
- InputField 和 OutputField
- Predict 预测器
- 简化的 Signature 语法
- ChainOfThought 基础使用

**运行：**
```bash
python examples/01_basic.py
```

---

#### 02_chain_of_thought.py - 思维链推理
**学习内容：**
- ChainOfThought 详细用法
- 数学问题求解
- 逻辑推理
- Predict vs ChainOfThought 对比
- 多步骤推理

**运行：**
```bash
python examples/02_chain_of_thought.py
```

---

#### 03_rag.py - 检索增强生成
**学习内容：**
- 基于上下文的问答
- 自定义 RAG 模块
- 多跳推理
- 文档检索模拟
- RAG 最佳实践

**运行：**
```bash
python examples/03_rag.py
```

---

### 优化器部分

#### 04_optimization.py - BootstrapFewShot 优化器
**学习内容：**
- 训练数据准备
- BootstrapFewShot 优化器使用
- 评估指标定义
- 优化前后对比
- 查看优化后的提示词

**核心概念：**
- 自动生成高质量示例
- Few-Shot Learning
- 提示词优化

**运行：**
```bash
python examples/04_optimization.py
```

---

#### 08_labeled_fewshot.py - LabeledFewShot 优化器
**学习内容：**
- 直接使用标注数据作为示例
- 不同 k 值的影响
- 与 BootstrapFewShot 的对比
- 选择策略

**核心概念：**
- 最简单的优化器
- 直接使用标注数据
- 快速原型开发

**运行：**
```bash
python examples/08_labeled_fewshot.py
```

---

### 高级功能部分

#### 05_react_agent.py - ReAct 智能体
**学习内容：**
- ReAct（Reasoning + Acting）模式
- 工具（Tools）定义和使用
- 工具调用
- 与 ChainOfThought 的对比
- Agent 系统构建

**核心概念：**
- 思考-行动-观察循环
- 外部工具集成
- 智能体系统

**运行：**
```bash
python examples/05_react_agent.py
```

---

#### 06_assertions.py - 输出约束和验证
**学习内容：**
- 手动实现输出约束
- 长度约束
- 格式约束
- 内容约束
- 复合约束
- 重试机制

**核心概念：**
- 输出质量控制
- 验证和修正
- 兜底策略

**注意：** DSPy 可能没有内置的 assertions 模块，本示例展示如何手动实现类似功能。

**运行：**
```bash
python examples/06_assertions.py
```

---

#### 07_evaluate.py - 评估系统
**学习内容：**
- Evaluate 系统使用
- 单指标评估
- 多指标评估
- 自定义评估指标
- 优化前后性能对比

**核心概念：**
- 系统化评估
- 性能基准测试
- 持续监控

**运行：**
```bash
python examples/07_evaluate.py
```

---

#### 09_program_of_thought.py - 代码生成求解
**学习内容：**
- ProgramOfThought 模式
- 代码生成
- 安全代码执行
- 数学计算
- 数据处理
- 与 ChainOfThought 对比

**核心概念：**
- 通过代码解决问题
- 精确计算
- 安全沙箱

**运行：**
```bash
python examples/09_program_of_thought.py
```

---

## 🎯 学习路径建议

### 初学者路径
1. **01_basic.py** - 理解基础概念
2. **02_chain_of_thought.py** - 学习推理模式
3. **03_rag.py** - 了解 RAG 应用
4. **04_optimization.py** - 学习优化器
5. **07_evaluate.py** - 学习评估方法

### 进阶路径
6. **08_labeled_fewshot.py** - 更多优化器
7. **06_assertions.py** - 输出约束
8. **09_program_of_thought.py** - 代码生成
9. **05_react_agent.py** - Agent 系统

---

## 📊 功能对比表

| 示例 | Signature | Predict | ChainOfThought | Module | Optimizer | Evaluate | Tools |
|------|-----------|---------|----------------|--------|-----------|----------|-------|
| 01_basic | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 02_chain_of_thought | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 03_rag | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 04_optimization | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| 05_react_agent | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |
| 06_assertions | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 07_evaluate | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| 08_labeled_fewshot | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| 09_program_of_thought | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

---

## 🔑 核心概念总结

### 1. Signature（签名）
定义任务的输入和输出：
```python
class MyTask(dspy.Signature):
    """任务描述"""
    input_field = dspy.InputField(desc="输入说明")
    output_field = dspy.OutputField(desc="输出说明")
```

### 2. 预测器（Predictors）
- **Predict**: 基础预测器
- **ChainOfThought**: 带推理过程的预测器
- **ReAct**: 带工具调用的智能体
- **ProgramOfThought**: 生成代码来解决问题

### 3. Module（模块）
自定义的组合模块：
```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Signature)

    def forward(self, input):
        return self.predictor(input=input)
```

### 4. 优化器（Optimizers）
- **BootstrapFewShot**: 自动生成示例优化
- **LabeledFewShot**: 使用标注数据优化
- **MIPRO**: 多指令提示优化（未包含）

### 5. 评估（Evaluate）
系统化评估模型性能：
```python
evaluator = dspy.Evaluate(
    devset=test_data,
    metric=my_metric,
    num_threads=1
)
result = evaluator(model)
```

---

## 💡 最佳实践

### 1. 开发流程
1. 定义清晰的 Signature
2. 选择合适的预测器
3. 准备训练/测试数据
4. 定义评估指标
5. 评估基础性能
6. 选择优化器优化
7. 评估优化后性能
8. 迭代改进

### 2. Signature 设计
- 文档字符串要清晰描述任务
- 字段描述要具体
- 使用有意义的字段名
- 考虑添加约束说明

### 3. 优化器选择
- 有高质量标注数据 → LabeledFewShot
- 需要最佳性能 → BootstrapFewShot
- 快速原型 → LabeledFewShot
- 成本敏感 → LabeledFewShot

### 4. 评估策略
- 准备高质量测试集
- 使用多个评估指标
- 定期评估性能
- 记录结果用于追踪

---

## 🚀 下一步

学完这些示例后，你可以：

1. **构建实际应用**
   - 问答系统
   - 文本分类
   - 信息抽取
   - Agent 系统

2. **探索更多特性**
   - MIPRO 优化器
   - 实际检索器集成（ColBERT, ChromaDB）
   - 多 LM 配置
   - 并行处理

3. **深入学习**
   - 阅读 DSPy 官方文档
   - 研究论文
   - 参与社区讨论
   - 贡献代码

---

## 📖 参考资源

- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy 论文](https://arxiv.org/abs/2310.03714)
- [DSPy 文档](https://dspy-docs.vercel.app/)

---

## ❓ 常见问题

### Q: 应该先学习哪个示例？
A: 按顺序学习 01 → 02 → 03，这些是基础。

### Q: 优化器和评估必须一起使用吗？
A: 不是必须，但强烈建议。评估帮助你量化优化效果。

### Q: ReAct 和 ChainOfThought 有什么区别？
A: ReAct 可以调用外部工具，ChainOfThought 只是展示推理过程。

### Q: 如何选择合适的优化器？
A: 根据数据质量、时间预算和性能要求选择。见"优化器选择"部分。

---

**祝学习愉快！🎉**
