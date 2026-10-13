# ============================================================
# Dockerfile —— 私有知识库 RAG + Agent 问答系统
# ============================================================
# ⚠️ 说明：本文件未经实际构建验证。首次使用请先执行：
#     docker build -t rag-agent .
#     docker run --rm -p 8000:8000 --env-file .env rag-agent
# 如构建失败，请把报错贴出来再修。
# ============================================================
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先单独拷贝依赖清单：只要它没变，修改代码不会触发重新安装（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 下载模型（约 500MB，走 ModelScope 国内镜像），构建时间较长。
# 若本地已有 models/ 目录并打算用 -v 挂载，可加 --build-arg DOWNLOAD_MODELS=false 跳过。
ARG DOWNLOAD_MODELS=true
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        python download_model.py && python download_reranker.py; \
    fi

EXPOSE 8000

# 默认启动 HTTP 接口服务（FastAPI）。
# 如需 Web 界面，覆盖启动命令：
#   docker run ... rag-agent streamlit run app.py --server.port 8000 --server.address 0.0.0.0
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
