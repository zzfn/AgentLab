# LangGraph 演示

使用 OpenAI 兼容的 API 端点。需要设置环境变量：

```
OPENAI_API_KEY=your_key_here
OPENAI_API_BASE=https://your-api-endpoint.com
```

也可以在项目根目录创建 `.env` 文件。

运行演示：

```
python langgraph/demo_basic_chat.py
python langgraph/demo_tool_calling.py
python langgraph/demo_router.py
```
