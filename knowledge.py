# ============================================================
# knowledge.py —— 服务端知识库：按客户端隔离 + 后端可切换的持久化
# ============================================================
import contextvars
import re
from pathlib import Path

import config
import vectorstore

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
    """单个客户端的知识库。

    只负责「按客户端隔离」这一层语义，真正的存取与检索交给
    vectorstore 里的后端（NumPy 全量扫描 / Chroma HNSW），
    因此换后端不会牵动这里的任何一行调用代码。
    """

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        # 懒加载：第一次真正用到时才建后端（Chroma 会开连接，代价不为零）
        self._store = None

    # ---------------- 目录与后端 ----------------

    @property
    def directory(self) -> Path:
        return config.KNOWLEDGE_DIR / self.client_id

    @property
    def store(self):
        """按客户端隔离的向量存储实例。"""
        if self._store is None:
            self._store = vectorstore.build_store(self.client_id, self.directory)
        return self._store

    @property
    def backend(self) -> str:
        """当前后端名称：numpy / chroma（便于健康检查与日志）。"""
        return self.store.backend

    # ---------------- 只读视图 ----------------

    @property
    def chunks(self) -> list[str]:
        return self.store.chunks()

    @property
    def vectors(self) -> list:
        return self.store.vectors()

    @property
    def sources(self) -> list[str]:
        return self.store.sources()

    @property
    def ready(self) -> bool:
        """是否已有可检索的内容。"""
        return self.store.count() > 0

    def info(self) -> dict:
        return {"sources": self.sources, "chunks": self.store.count()}

    def count(self) -> int:
        """当前片段总数（健康检查与统计用）。"""
        return self.store.count()

    # ---------------- 增删查 ----------------

    def add(self, chunks: list[str], vectors, source: str = "") -> int:
        """追加片段与向量，返回追加后的片段总数。"""
        return self.store.add(chunks, vectors, source)

    def search(self, query_vector, limit: int) -> list[str]:
        """按向量相似度取候选片段，供上层 rerank 精排。"""
        return self.store.query(query_vector, limit)

    def clear(self) -> None:
        """清空该客户端的知识库（含已落盘数据）。"""
        self.store.clear()

    # ---------------- 持久化 ----------------

    def save(self) -> None:
        """落盘；Chroma 后端自身持久化，这里只保证接口一致。"""
        self.store.persist()

    def load(self) -> bool:
        """从磁盘恢复；无数据或已损坏时返回 False。"""
        return self.store.load()


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
