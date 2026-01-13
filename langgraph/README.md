# LangGraph 演示

学习 LangGraph 的基础用法，按顺序阅读和运行以下演示：

## 演示列表

| 文件                        | 内容         | 学习要点                    |
| --------------------------- | ------------ | --------------------------- |
| `01_basic_chat.py`          | 基础对话     | State, Node, Graph, Compile |
| `02_streaming.py`           | 流式输出     | `app.stream()` 用法         |
| `03_conditional_routing.py` | 条件路由     | `add_conditional_edges()`   |
| `04_multi_node_workflow.py` | 多节点工作流 | 顺序节点、状态传递          |

## 环境配置

设置环境变量（或在项目根目录创建 `.env` 文件）：

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_API_BASE="https://your-api-endpoint.com"
```

## 运行

```bash
uv run python langgraph/01_basic_chat.py
uv run python langgraph/02_streaming.py
uv run python langgraph/03_conditional_routing.py
uv run python langgraph/04_multi_node_workflow.py
```

## LangGraph 核心概念

```
┌─────────────────────────────────────────┐
│                 Graph                   │
│  ┌─────┐    ┌─────┐    ┌─────┐         │
│  │START│───▶│Node │───▶│ END │         │
│  └─────┘    └─────┘    └─────┘         │
│                │                        │
│                ▼                        │
│            [State]                      │
└─────────────────────────────────────────┘
```

- **State**: TypedDict，定义工作流中传递的数据结构
- **Node**: 处理函数，接收 State 返回更新
- **Edge**: 节点之间的连接
- **Conditional Edge**: 根据条件选择下一个节点
