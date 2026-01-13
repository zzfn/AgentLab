# AgentLab

🚀 **AI Agent Learning Lab - 智能体学习实验室**

一个用于学习和实验 AI Agent 技术的项目，集成智谱 AI 和 LangChain。

## 功能特性

- ✅ 智谱 AI 原生 SDK 调用
- ✅ LangChain + 智谱 AI 集成 (OpenAI 兼容协议)
- ✅ LCEL (LangChain Expression Language) 语法
- ✅ 流式输出
- ✅ 对话记忆
- ✅ 批量处理
- ✅ 多链组合

## 快速开始

### 1. 安装依赖

```bash
uv add langchain langchain-openai
```

### 2. 设置 API Key

```bash
export ZHIPUAI_API_KEY="your_api_key_here"
```

### 3. 运行测试

**测试智谱 AI 原生 SDK:**

```bash
uv run python test_zhipu.py
```

**测试 LangChain 集成:**

```bash
uv run python test_langchain_zhipu.py
```

## 项目结构

```
agentlab/
├── main.py                      # 主程序入口
├── test_zhipu.py                # 智谱 AI 原生 SDK 测试
├── test_langchain_zhipu.py      # LangChain + 智谱 AI 测试
├── pyproject.toml               # 项目配置
└── README.md                    # 项目说明
```

## 技术栈

- **Python 3.11+**
- **智谱 AI (GLM-4-Flash)** - 国产大语言模型
- **LangChain** - AI 应用开发框架
- **uv** - 快速的 Python 包管理器

## 学习资源

- [智谱 AI 开放平台](https://open.bigmodel.cn/)
- [LangChain 官方文档](https://python.langchain.com/)
- [LCEL 文档](https://python.langchain.com/docs/expression_language/)

## License

MIT
