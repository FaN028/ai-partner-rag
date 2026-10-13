# ============================================================
# knowledge.py —— 服务端知识库：按客户端隔离 + 落盘持久化
# ============================================================
import contextvars
import json
import re
from pathlib import Path

import numpy as np

import config

# 当前请求所属的客户端标识。
# 工具函数（search_knowledge）无法从模型那里拿到客户端信息，
# 因此由接口层在每次请求开始时写入，工具函数再读取。
_current_client: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_client", default="default"
)


def normalize_client_id(raw: str | None) -> str:
    """把客户端标识规范化为安全的目录名；缺省或非法时归入 default。"""
    if not raw:
        return "default"
    safe = re.sub(r"[^0-9a-zA-Z_-]", "", raw)[:40]
    return safe or "default"


def set_current_client(client_id: str | None) -> None:
    """设置当前请求的客户端标识（由接口层调用）。"""
    _current_client.set(normalize_client_id(client_id))


def current_client() -> str:
    """读取当前请求的客户端标识。"""
    return _current_client.get()


class KnowledgeBase:
    """单个客户端的知识库：片段、向量与来源清单，并支持落盘与恢复。"""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.chunks: list[str] = []
        self.vectors: list = []
        self.sources: list[str] = []

    # ---------------- 目录 ----------------

    @property
    def directory(self) -> Path:
        return config.KNOWLEDGE_DIR / self.client_id

    # ---------------- 增删查 ----------------

    @property
    def ready(self) -> bool:
        """是否已有可检索的内容。"""
        return len(self.chunks) > 0 and len(self.vectors) > 0

    def add(self, chunks: list[str], vectors, source: str = "") -> int:
        """追加片段与向量，返回追加后的片段总数。"""
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)
        if source and source not in self.sources:
            self.sources.append(source)
        return len(self.chunks)

    def clear(self) -> None:
        """清空内存中的数据，并删除已落盘的文件。"""
        self.chunks.clear()
        self.vectors.clear()
        self.sources.clear()
        if self.directory.exists():
            for file in self.directory.iterdir():
                if file.is_file():
                    file.unlink()

    def info(self) -> dict:
        return {"sources": list(self.sources), "chunks": len(self.chunks)}

    # ---------------- 持久化 ----------------

    def save(self) -> None:
        """把片段、来源与向量写入磁盘。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.directory / "sources.json").write_text(
            json.dumps(self.sources, ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(self.directory / "vectors.npy", np.asarray(self.vectors, dtype="float32"))

    def load(self) -> bool:
        """从磁盘恢复；数据不存在或已损坏时返回 False。"""
        chunks_file = self.directory / "chunks.json"
        vectors_file = self.directory / "vectors.npy"
        if not chunks_file.exists() or not vectors_file.exists():
            return False

        try:
            chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            vectors = list(np.load(vectors_file))
            sources_file = self.directory / "sources.json"
            sources = (
                json.loads(sources_file.read_text(encoding="utf-8"))
                if sources_file.exists()
                else []
            )
        except (OSError, ValueError):
            return False

        if len(chunks) != len(vectors):
            return False

        self.chunks = chunks
        self.vectors = vectors
        self.sources = sources
        return True


# ---------------- 客户端 → 知识库 的注册表 ----------------

_STORES: dict[str, KnowledgeBase] = {}


def get_store(client_id: str | None = None) -> KnowledgeBase:
    """取指定客户端的知识库；未显式指定时使用当前请求的客户端。

    首次访问某个客户端时，会尝试从磁盘恢复其历史数据。
    """
    key = normalize_client_id(client_id if client_id is not None else current_client())

    if key not in _STORES:
        store = KnowledgeBase(key)
        store.load()
        _STORES[key] = store

    return _STORES[key]


def forget_all() -> None:
    """清空内存中的注册表（测试用；不影响磁盘数据）。"""
    _STORES.clear()
