# ============================================================
# vectorstore.py —— 向量存储抽象层：Chroma / NumPy 双后端
# ============================================================
"""为什么要有这一层？

早期实现把「片段 + 向量」直接塞进 Python 列表，每次检索都对全量向量做
一遍余弦扫描（O(N)）。数据量小的时候没问题，片段上千后单次检索就开始
肉眼可见地变慢，而且重启只能靠 .npy 全量读回内存。

这里把「存」和「查」抽成一组统一方法，交给两个实现：

- NumpyVectorStore  —— 纯 NumPy 全量扫描，零外部依赖。默认后端，
                       单元测试与单机演示都不需要装数据库。
- ChromaVectorStore —— Chroma 持久化 + HNSW 近似最近邻，生产用。
                       通过环境变量 RAG_VECTOR_BACKEND=chroma 打开。

上层（knowledge.KnowledgeBase）只依赖这组方法，换后端不改调用方。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

import config


class VectorStoreError(RuntimeError):
    """向量存储不可用（例如显式指定了 chroma 但没装依赖）。"""


def _as_matrix(vectors) -> np.ndarray:
    """把任意形状的向量集合规整成 (N, D) 的 float32 矩阵。"""
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.size == 0:
        return np.zeros((0, 0), dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """批量余弦相似度；分母为 0 时退化为 0，避免除零告警。"""
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    denominator[denominator == 0] = 1e-12
    return (matrix @ query) / denominator


# ============================================================
# 后端一：NumPy 全量扫描（默认，零依赖）
# ============================================================
class NumpyVectorStore:
    """把片段与向量放在内存里，检索时全量算余弦相似度后取 Top-N。"""

    backend = "numpy"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._chunks: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._sources: list[str] = []

    # ---------------- 写 ----------------

    def add(self, chunks: list[str], vectors, source: str = "") -> int:
        matrix = _as_matrix(vectors)
        if len(chunks) != len(matrix):
            raise VectorStoreError(
                f"片段数({len(chunks)})与向量数({len(matrix)})不一致，拒绝写入。"
            )
        self._chunks.extend(chunks)
        self._vectors.extend(list(matrix))
        if source and source not in self._sources:
            self._sources.append(source)
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._sources.clear()
        if self.directory.exists():
            for file in self.directory.iterdir():
                if file.is_file():
                    file.unlink()

    # ---------------- 读 ----------------

    def chunks(self) -> list[str]:
        return list(self._chunks)

    def vectors(self) -> list[np.ndarray]:
        return list(self._vectors)

    def sources(self) -> list[str]:
        return list(self._sources)

    def count(self) -> int:
        return len(self._chunks)

    def query(self, vector, limit: int) -> list[str]:
        """返回与 vector 最相近的前 limit 个片段（按相似度降序）。"""
        if not self._chunks or limit <= 0:
            return []
        query = np.asarray(vector, dtype="float32").reshape(-1)
        matrix = np.vstack(self._vectors)
        if matrix.shape[1] != query.shape[0]:
            raise VectorStoreError(
                f"查询向量维度({query.shape[0]})与库内维度({matrix.shape[1]})不一致。"
            )
        scores = _cosine_scores(query, matrix)
        # 同分时按原始顺序，保证结果可复现
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self._chunks[i] for i in order[:limit]]

    # ---------------- 持久化 ----------------

    def persist(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "chunks.json").write_text(
            json.dumps(self._chunks, ensure_ascii=False), encoding="utf-8"
        )
        (self.directory / "sources.json").write_text(
            json.dumps(self._sources, ensure_ascii=False), encoding="utf-8"
        )
        matrix = np.vstack(self._vectors) if self._vectors else np.zeros((0, 0), "float32")
        np.save(self.directory / "vectors.npy", matrix.astype("float32"))

    def load(self) -> bool:
        chunks_file = self.directory / "chunks.json"
        vectors_file = self.directory / "vectors.npy"
        if not chunks_file.exists() or not vectors_file.exists():
            return False
        try:
            chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            matrix = np.load(vectors_file)
            sources_file = self.directory / "sources.json"
            sources = (
                json.loads(sources_file.read_text(encoding="utf-8"))
                if sources_file.exists()
                else []
            )
        except (OSError, ValueError):
            return False

        vectors = list(matrix) if matrix.size else []
        if len(chunks) != len(vectors):
            return False

        self._chunks = chunks
        self._vectors = vectors
        self._sources = sources
        return True


# ============================================================
# 后端二：Chroma（持久化 + HNSW 近似最近邻）
# ============================================================
class ChromaVectorStore:
    """片段与向量交给 Chroma 持久化，检索走 HNSW 索引。"""

    backend = "chroma"

    # Chroma（Rust 实现）单次写入上限为 5461 条，超过会抛 InternalError:
    # "Batch size of N is greater than max batch size of 5461"。
    # 这里取一个留有余量的值，写入时分批提交。
    BATCH_LIMIT = 4000

    def __init__(self, directory: Path, client_id: str) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise VectorStoreError(
                "未安装 chromadb，无法使用 chroma 后端：pip install chromadb"
            ) from exc

        self.directory = Path(directory)
        self.client_id = client_id
        self._client = chromadb.PersistentClient(path=str(self.directory / "chroma"))
        self._collection = self._open_collection()
        # 用于统计本次进程写入的片段数，避免每次 info() 都打到数据库
        self._added = 0

    def _open_collection(self):
        """按客户端建独立集合；集合名只允许字母数字与 -_，故先做一次清洗。"""
        name = f"kb_{self.client_id}"[:60]
        try:
            return self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # 兼容不同版本的 Chroma：老版本不认 metadata 里的 hnsw 配置
            return self._client.get_or_create_collection(name=name)

    # ---------------- 写 ----------------

    def add(self, chunks: list[str], vectors, source: str = "") -> int:
        matrix = _as_matrix(vectors)
        if len(chunks) != len(matrix):
            raise VectorStoreError(
                f"片段数({len(chunks)})与向量数({len(matrix)})不一致，拒绝写入。"
            )
        if not chunks:
            return self.count()

        total = len(chunks)
        added = 0
        for start in range(0, total, self.BATCH_LIMIT):
            end = min(start + self.BATCH_LIMIT, total)
            batch_ids = [uuid.uuid4().hex for _ in range(end - start)]
            metadatas = [
                {"source": source or "", "order": i} for i in range(start, end)
            ]
            self._collection.add(
                ids=batch_ids,
                documents=list(chunks[start:end]),
                embeddings=matrix[start:end].tolist(),
                metadatas=metadatas,
            )
            added += end - start
        self._added += added
        return self.count()

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection.name)
        except Exception:
            pass
        self._collection = self._open_collection()
        self._added = 0

    # ---------------- 读 ----------------

    def chunks(self) -> list[str]:
        result = self._collection.get(include=["documents"])
        return list(result.get("documents") or [])

    def vectors(self) -> list[np.ndarray]:
        result = self._collection.get(include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            return []
        matrix = np.asarray(embeddings, dtype="float32")
        return list(matrix) if matrix.size else []

    def sources(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        seen: list[str] = []
        for meta in result.get("metadatas") or []:
            name = (meta or {}).get("source") or ""
            if name and name not in seen:
                seen.append(name)
        return seen

    def count(self) -> int:
        return int(self._collection.count())

    def query(self, vector, limit: int) -> list[str]:
        total = self.count()
        if total == 0 or limit <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype="float32").tolist()],
            n_results=min(limit, total),
            include=["documents"],
        )
        documents = (result.get("documents") or [[]])[0]
        return [doc for doc in documents if doc]

    # ---------------- 持久化 ----------------

    def persist(self) -> None:
        """Chroma 自身即持久化存储，这里只需保证目录存在（接口对齐）。"""
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self) -> bool:
        return self.count() > 0


# ============================================================
# 工厂：按配置挑后端
# ============================================================
def build_store(client_id: str, directory: Path):
    """根据 config.VECTOR_BACKEND 创建后端实例。

    - numpy（默认）：零依赖，全量扫描
    - chroma：HNSW 检索；未安装依赖时按 auto 语义回退到 numpy
    """
    backend = (config.VECTOR_BACKEND or "numpy").strip().lower()

    if backend in ("chroma", "auto"):
        try:
            return ChromaVectorStore(directory, client_id)
        except VectorStoreError:
            if backend == "chroma":
                # 显式指定了却没装依赖，属于配置错误，直接报出来而不是静默降级
                raise
            # auto：有就用，没有就退回去
    return NumpyVectorStore(directory)
