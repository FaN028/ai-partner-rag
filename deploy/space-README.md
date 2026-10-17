---
title: 私有知识库 RAG + Agent 问答
emoji: 📚
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 私有知识库 RAG + Agent 问答系统

上传 txt / md / pdf / docx 构建自己的知识库，提问时先检索、再作答，并在回答下方标出命中的原文片段。

两个子系统：

- **RAG 检索链路** —— BGE 中文向量化 → 余弦粗排 → bge-reranker 精排 → 上下文注入 → 流式生成 → 答案溯源
- **Agent 工具调度** —— 由模型自己决定调不调工具（知识库检索 / 数学计算 / 当前时间），调用轨迹可展开复盘

技术栈：Python · Streamlit · FastAPI · DeepSeek API · sentence-transformers · Chroma / NumPy 双后端 · Docker

源码：https://github.com/FaN028/rag-knowledge-qa

> 本空间使用免费 CPU 实例，首次打开需要 30–60 秒加载模型；实例空闲一段时间会休眠，再次打开同样需要等待。
> 在空间设置里添加环境变量 `DEEPSEEK_API_KEY` 后即可提问。
