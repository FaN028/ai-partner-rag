# ============================================================
# Dockerfile —— 私有知识库 RAG + Agent 问答系统
# ============================================================
# 两阶段构建：
#   builder：装依赖（含 torch 这类大包）并下载模型
#   runtime：只拷虚拟环境与模型，不带编译器与构建缓存
#
# 常用命令：
#   docker build -t rag-agent .
#   docker build -t rag-agent --build-arg DOWNLOAD_MODELS=false .   # 模型走挂载
#   docker run --rm -p 8000:8000 --env-file .env rag-agent
#   docker compose up -d                                            # API + Web 界面
# ============================================================

# ---------------- 构建阶段 ----------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖装进独立 venv，方便整目录拷进运行阶段
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 先单独拷贝依赖清单：只要它没变，改代码不会触发重新安装
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 下载模型（走 ModelScope 国内镜像，两个模型合计约 3.4G，去重后约 2.2G）。
# 若打算用 -v 挂载本地 models/，加 --build-arg DOWNLOAD_MODELS=false 跳过。
ARG DOWNLOAD_MODELS=true
COPY download_model.py download_reranker.py ./
# 两个模型都同时带了 safetensors 与 pytorch_model.bin（各 1.1G），留一份就够：
# 有 safetensors 就删掉 .bin，镜像直接小 1.2G。
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        python download_model.py && python download_reranker.py; \
    fi \
    && mkdir -p models \
    && for d in models/*/; do \
         if [ -f "$d/model.safetensors" ] && [ -f "$d/pytorch_model.bin" ]; then \
           rm -f "$d/pytorch_model.bin"; \
         fi; \
       done \
    && du -sh models

# ---------------- 运行阶段 ----------------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    RAG_VECTOR_BACKEND=chroma \
    WEB_CONCURRENCY=1 \
    PORT=8000

# sentence-transformers 依赖的 torch 在 slim 镜像里缺 libgomp1，补上
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/models /app/models

# 以非 root 用户运行：容器被拿下时，攻击者拿到的也只是普通权限
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/sessions \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# 就绪探针：直接打 /health，顺带确认向量后端没被静默降级
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# 默认启动 HTTP 接口服务；Web 界面用 docker compose 的 ui 服务覆盖启动命令
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips='*'"]
