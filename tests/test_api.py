# ============================================================
# test_api.py —— api.py 的接口测试：状态码、校验、序列化
#               不联网、不加载 embedding 模型
# ============================================================
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api
import tools

SAMPLE_TRACE = [
    {"round": 1, "tool": "calculate", "args": {"expression": "1+1"}, "result": "2"},
]


@pytest.fixture
def client(monkeypatch):
    """提供 TestClient，并把外部依赖替换掉：不真的调用 DeepSeek。"""
    monkeypatch.setattr(api, "_client", object())
    monkeypatch.setattr(
        api,
        "run_agent",
        lambda *_args, **_kwargs: ("答复", SAMPLE_TRACE),
    )
    return TestClient(api.app)


# ---------------- 健康检查 ----------------

def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"]


# ---------------- /chat ----------------

def test_chat_returns_answer_and_trace(client):
    response = client.post("/chat", json={"question": "1+1"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "答复"
    assert body["trace"][0]["tool"] == "calculate"
    assert body["trace"][0]["args"] == {"expression": "1+1"}


def test_chat_accepts_history(client):
    response = client.post(
        "/chat",
        json={
            "question": "那它呢",
            "history": [{"role": "user", "content": "我叫小明"}],
        },
    )

    assert response.status_code == 200


def test_chat_without_question_returns_422(client):
    """缺必填字段由 Pydantic 自动拦截，无需手写校验代码。"""
    response = client.post("/chat", json={"history": []})

    assert response.status_code == 422


def test_chat_returns_503_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(api, "_client", None)

    response = TestClient(api.app).post("/chat", json={"question": "hi"})

    assert response.status_code == 503


def test_chat_returns_502_when_agent_raises(monkeypatch):
    monkeypatch.setattr(api, "_client", object())

    def boom(*_args, **_kwargs):
        raise RuntimeError("上游炸了")

    monkeypatch.setattr(api, "run_agent", boom)

    response = TestClient(api.app).post("/chat", json={"question": "hi"})

    assert response.status_code == 502
    assert "上游炸了" in response.json()["detail"]


# ---------------- /knowledge ----------------

def test_knowledge_status_starts_empty(client):
    tools.clear_knowledge()

    response = client.get("/knowledge")

    assert response.status_code == 200
    assert response.json() == {"sources": [], "chunks": 0}


def test_upload_knowledge_chunks_and_stores(client, monkeypatch):
    tools.clear_knowledge()

    # 用假的 rag 模块替换真实模块，避免加载数百 MB 的 embedding 模型
    fake_rag = SimpleNamespace(
        chunk_text=lambda text: [text[i:i + 5] for i in range(0, len(text), 5)],
        model=SimpleNamespace(encode=lambda chunks: [[0.0]] * len(chunks)),
    )
    monkeypatch.setitem(sys.modules, "rag", fake_rag)

    response = client.post(
        "/knowledge",
        json={"source": "demo.md", "content": "0123456789"},
    )

    assert response.status_code == 200
    assert response.json() == {"sources": ["demo.md"], "chunks": 2}

    tools.clear_knowledge()


def test_upload_knowledge_rejects_empty_content(client, monkeypatch):
    tools.clear_knowledge()
    fake_rag = SimpleNamespace(
        chunk_text=lambda _text: [],
        model=SimpleNamespace(encode=lambda chunks: [[0.0]] * len(chunks)),
    )
    monkeypatch.setitem(sys.modules, "rag", fake_rag)

    response = client.post("/knowledge", json={"source": "empty.md", "content": "   "})

    assert response.status_code == 400


def test_delete_knowledge_clears_store(client):
    tools.add_knowledge(["片段"], [[0.1]], source="tmp.md")

    response = client.delete("/knowledge")

    assert response.status_code == 200
    assert response.json() == {"sources": [], "chunks": 0}


def test_knowledge_is_isolated_between_clients(client, monkeypatch):
    """不同 X-Client-Id 应看到各自的知识库，互不干扰。"""
    tools.clear_knowledge()
    fake_rag = SimpleNamespace(
        chunk_text=lambda text: [text],
        model=SimpleNamespace(encode=lambda chunks: [[0.0]] * len(chunks)),
    )
    monkeypatch.setitem(sys.modules, "rag", fake_rag)

    client.post(
        "/knowledge",
        json={"source": "a.md", "content": "alice 的资料"},
        headers={"X-Client-Id": "alice"},
    )

    alice = client.get("/knowledge", headers={"X-Client-Id": "alice"}).json()
    bob = client.get("/knowledge", headers={"X-Client-Id": "bob"}).json()

    assert alice == {"sources": ["a.md"], "chunks": 1}
    assert bob == {"sources": [], "chunks": 0}


def test_cors_headers_are_returned(client):
    response = client.get("/health", headers={"Origin": "http://example.com"})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
