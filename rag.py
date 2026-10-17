# ============================================================
# rag.py —— 知识库（RAG）相关：模型加载、切块、解析、检索、注入
# ============================================================
import re
from io import BytesIO

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder

import config


# —— 加载模型（@st.cache_resource = 只加载一次，之后复用）——
@st.cache_resource
def load_model():
    """加载中文 embedding 模型。"""
    return SentenceTransformer(config.MODEL_PATH)


@st.cache_resource
def load_reranker():
    """加载 rerank（重排）模型，用于精排。"""
    return CrossEncoder(config.RERANK_MODEL_PATH)


model = load_model()

# rerank 模型常驻内存约 1.5GB，低配机器（如 2G 内存的云主机）会被它压垮。
# 因此改成按需加载，并允许用 RAG_RERANK=0 整体关掉——关掉后只做向量粗排，
# 精度略降但内存占用直接少一半。
_reranker = None


def get_reranker():
    """取 rerank 模型；未启用时返回 None。"""
    global _reranker
    if not config.RERANK_ENABLED:
        return None
    if _reranker is None:
        _reranker = load_reranker()
    return _reranker


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """把一段长文本切成若干固定长度、彼此有重叠的小片段。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_text(filename: str, raw: bytes) -> str | None:
    """把上传文件（txt/md/pdf/docx）的原始字节，转成纯文本。"""
    name = filename.lower()

    if name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore").strip() or None

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                st.warning("读取 PDF 需要先安装 pypdf：请在终端运行 `pip install pypdf`。")
                return None
        try:
            reader = PdfReader(BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip() or None
        except Exception as exc:
            st.warning(f"解析 PDF 失败：{exc}")
            return None

    if name.endswith(".docx"):
        try:
            import docx
        except ImportError:
            st.warning("读取 Word 需要先安装 python-docx：请在终端运行 `pip install python-docx`。")
            return None
        try:
            document = docx.Document(BytesIO(raw))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return text.strip() or None
        except Exception as exc:
            st.warning(f"解析 Word 失败：{exc}")
            return None

    st.warning(f"暂不支持的文件类型：{filename}")
    return None


def cosine_similarity(a, b) -> float:
    """算两个向量的余弦相似度，越接近 1 表示意思越像。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_chunks(
    query: str,
    chunks: list[str] | None = None,
    vectors=None,
    top_k: int | None = None,
    searcher=None,
) -> list[str]:
    """两阶段检索：先向量粗排筛出候选，再用 rerank 精排取 top_k。

    粗排有两条路：
    - 传入 searcher(query_vector, limit) 时，交给向量存储后端（可能是
      Chroma 的 HNSW 索引）处理，此时无需把全量片段搬进内存；
    - 不传 searcher 时，退化为对 chunks / vectors 的全量余弦扫描。

    这样界面模式（数据就在 session_state 里）与服务端模式（数据在向量库）
    共用同一段精排与拼装逻辑。
    """
    if top_k is None:
        top_k = config.TOP_K

    q_vec = model.encode(query)

    if searcher is not None:
        candidates = list(searcher(q_vec, config.RERANK_TOP_N))
    else:
        if chunks is None or vectors is None or not len(chunks) or not len(vectors):
            return []
        scored = []
        for idx, chunk in enumerate(chunks):
            score = cosine_similarity(q_vec, vectors[idx])
            scored.append((score, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        candidates = [chunk for _, _, chunk in scored[:config.RERANK_TOP_N]]

    if not candidates:
        return []

    # 精排：rerank 对候选重新打分；未启用 rerank 时直接取粗排前 top_k
    ranker = get_reranker()
    if ranker is None:
        return candidates[:top_k]

    pairs = [[query, chunk] for chunk in candidates]
    rerank_scores = ranker.predict(pairs)

    ranked = sorted(zip(candidates, rerank_scores), key=lambda item: -item[1])
    return [chunk for chunk, _ in ranked[:top_k]]


def build_context_block(
    query: str,
    chunks: list[str] | None = None,
    vectors=None,
    searcher=None,
):
    """检索并拼出知识库上下文，同时返回命中的片段（用于显示引用来源）。"""
    selected = retrieve_chunks(query, chunks, vectors, searcher=searcher)
    if not selected:
        return "", []

    parts = [f"【片段 {i + 1}】\n{chunk}" for i, chunk in enumerate(selected)]
    context = (
        "下面是用户上传的\"知识库\"中，与本次问题最相关的片段：\n"
        "如果用户的问题与这些内容相关，请优先依据这些片段作答；\n"
        "如果无关，请忽略它们，照常以知识库助手身份聊天。\n\n"
        + "\n\n".join(parts)
    )
    return context, selected
