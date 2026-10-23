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
# torch 先单独装 CPU 版：Linux 下 pip 默认装 CUDA 版，会连带 3.2G 的
# nvidia-* 库，而容器内无 GPU，这部分纯属体积浪费。先装 CPU 版后，
# requirements 里的 sentence-transformers 会直接复用，不再拉 CUDA 依赖。
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# 下载模型（走 ModelScope 国内镜像，原始约 3.4G）。
# 若打算用 -v 挂载本地 models/，加 --build-arg DOWNLOAD_MODELS=false 跳过。
ARG DOWNLOAD_MODELS=true
COPY download_model.py download_reranker.py ./
# 同一份权重常带多个副本，只留 sentence-transformers 实际会读的那份：
#   · pytorch_model.bin 与 model.safetensors 二选一（各 1.1G）
#   · onnx/ 目录只在 optimum / onnxruntime 推理时才用得上，本项目不读（1.1G）
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        python download_model.py && python download_reranker.py; \
    fi \
    && mkdir -p models \
    && for d in models/*/; do \
         if [ -f "$d/model.safetensors" ] && [ -f "$d/pytorch_model.bin" ]; then \
           rm -f "$d/pytorch_model.bin"; \
         fi; \
         rm -rf "$d/onnx"; \
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

# 以非 root 用户运行：容器被拿下时，攻击者拿到的也只是普通权限。
# 注意顺序：用户与目录必须在拷贝大目录「之前」建好，且不要事后 chown -R /app。
# chown 会改写每个文件的元数据，Docker 的写时复制会把整个模型目录重新落成一个新层
# ——实测因此白白多出 2.3G 的镜像体积。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/sessions \
    && chown -R appuser:appuser /app

# COPY --chown 让文件直接落成 appuser 所有，避免额外一层
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app/models /app/models

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# 就绪探针：直接打 /health，顺带确认向量后端没被静默降级
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# 默认启动 HTTP 接口服务；Web 界面用 docker compose 的 ui 服务覆盖启动命令
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips='*'"]
