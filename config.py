# ============================================================
# config.py —— 所有"常量"集中在这里，方便统一修改
# ============================================================
import os
from pathlib import Path

APP_TITLE = "私有知识库 RAG + Agent 问答系统"
SESSIONS_DIR = Path("sessions")
KNOWLEDGE_DIR = Path("data/knowledge")   # 服务端知识库落盘目录（按客户端分文件夹）
LOGO_PATH = Path("resources/WLive48x48.png")

# 默认昵称 / 性格
DEFAULT_NICK_NAME = "小A"
DEFAULT_NATURE = "活泼开朗的女生"

# —— 对话模式 ——
MODE_KB = "知识库模式"      # 每次提问均预先检索知识库
MODE_AGENT = "Agent 模式"   # 由模型决定是否调用工具

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-pro"

# —— 知识库（切块）参数 ——
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 4

# —— 模型路径 ——
MODEL_PATH = "models/bge-small-zh-v1.5"        # embedding 模型
RERANK_MODEL_PATH = "models/bge-reranker-base" # rerank 模型
RERANK_TOP_N = 20                               # 粗排先取多少个候选

# rerank 模型常驻约 1.5GB 内存。低配机器（2G 云主机 / 免费 CPU 实例）设 RAG_RERANK=0
# 关掉精排：只做向量粗排，精度略降，但内存占用少一半，冷启动也快得多。
RERANK_ENABLED = os.getenv("RAG_RERANK", "1").strip().lower() not in ("0", "false", "no")

# —— 向量存储后端 ——
# numpy（默认）：全量余弦扫描，零外部依赖，单元测试与单机演示用它
# chroma       ：HNSW 近似最近邻 + 持久化；未装依赖时直接报错，避免静默降级
# auto         ：装了 chromadb 就用 chroma，否则退回 numpy
VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "numpy").strip().lower()
