# ============================================================
# test_vectorstore.py —— 向量存储抽象层：双后端行为一致性
#               默认跑 NumPy 后端；装了 chromadb 才跑 Chroma 用例
# ============================================================
import numpy as np
import pytest

import config
import vectorstore

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------- NumPy 后端 ----------------

def test_numpy_query_returns_nearest_first(tmp_path):
    store = vectorstore.NumpyVectorStore(tmp_path)
    store.add(["猫", "狗", "汽车"], [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], source="a.md")

    assert store.query([1.0, 0.0], limit=2) == ["猫", "狗"]


def test_numpy_query_respects_limit_and_empty_store(tmp_path):
    store = vectorstore.NumpyVectorStore(tmp_path)
    assert store.query([1.0, 0.0], limit=3) == []

    store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert len(store.query([1.0, 0.0], limit=1)) == 1
    assert store.query([1.0, 0.0], limit=0) == []


def test_numpy_rejects_length_mismatch(tmp_path):
    store = vectorstore.NumpyVectorStore(tmp_path)

    with pytest.raises(vectorstore.VectorStoreError):
        store.add(["只有一段"], [[0.1, 0.2], [0.3, 0.4]])


def test_numpy_rejects_dimension_mismatch(tmp_path):
    store = vectorstore.NumpyVectorStore(tmp_path)
    store.add(["一段"], [[0.1, 0.2]], source="a.md")

    with pytest.raises(vectorstore.VectorStoreError):
        store.query([0.1, 0.2, 0.3], limit=1)


def test_numpy_persist_and_load_roundtrip(tmp_path):
    store = vectorstore.NumpyVectorStore(tmp_path)
    store.add(["片段一", "片段二"], [[0.1, 0.2], [0.3, 0.4]], source="a.md")
    store.add(["片段三"], [[0.5, 0.6]], source="b.md")
    store.persist()

    reopened = vectorstore.NumpyVectorStore(tmp_path)
    assert reopened.load() is True
    assert reopened.chunks() == ["片段一", "片段二", "片段三"]
    assert reopened.sources() == ["a.md", "b.md"]
    assert reopened.vectors()[0].shape == (2,)


def test_numpy_clear_removes_files(tmp_path):
    store = vectorstore.NumpyVectorStore(tmp_path)
    store.add(["片段"], [[0.1]], source="a.md")
    store.persist()
    assert any(tmp_path.iterdir())

    store.clear()

    assert store.count() == 0
    assert not any(tmp_path.iterdir())


def test_numpy_load_without_files_returns_false(tmp_path):
    assert vectorstore.NumpyVectorStore(tmp_path).load() is False


def test_build_store_defaults_to_numpy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_BACKEND", "numpy")

    store = vectorstore.build_store("alice", tmp_path / "alice")

    assert store.backend == "numpy"


# ---------------- Chroma 后端 ----------------

try:
    import chromadb  # noqa: F401
    HAS_CHROMA = True
except ImportError:  # pragma: no cover - 取决于运行环境
    HAS_CHROMA = False

requires_chroma = pytest.mark.skipif(not HAS_CHROMA, reason="未安装 chromadb，跳过 Chroma 用例")


@requires_chroma
def test_chroma_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_BACKEND", "chroma")

    store = vectorstore.build_store("alice", tmp_path / "alice")
    assert store.backend == "chroma"

    store.add(["年假", "报销", "加班"], [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], source="手册.md")
    assert store.count() == 3
    assert store.sources() == ["手册.md"]
    assert store.query([1.0, 0.0], limit=1) == ["年假"]

    store.clear()
    assert store.count() == 0


@requires_chroma
def test_chroma_splits_large_batch(tmp_path, monkeypatch):
    """Chroma 单批有上限（5461），写入必须分批，否则大批量入库会直接报错。"""
    monkeypatch.setattr(config, "VECTOR_BACKEND", "chroma")
    store = vectorstore.build_store("carol", tmp_path / "carol")
    monkeypatch.setattr(store, "BATCH_LIMIT", 2)

    chunks = [f"片段{i}" for i in range(5)]
    vectors = [[float(i), 1.0] for i in range(5)]

    assert store.add(chunks, vectors, source="big.md") == 5
    assert store.count() == 5


@requires_chroma
def test_chroma_survives_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_BACKEND", "chroma")
    directory = tmp_path / "bob"

    first = vectorstore.build_store("bob", directory)
    first.add(["片段一"], [[1.0, 0.0]], source="a.md")

    second = vectorstore.build_store("bob", directory)

    assert second.count() == 1
    assert second.chunks() == ["片段一"]
    assert np.asarray(second.vectors()[0]).shape == (2,)
