# ============================================================
# test_knowledge.py —— knowledge.py 的单元测试：客户端隔离与落盘恢复
# ============================================================
import knowledge
from tools import add_knowledge, clear_knowledge, knowledge_info


def test_stores_are_isolated_per_client():
    knowledge.set_current_client("alice")
    add_knowledge(["alice 的资料"], [[0.1, 0.2]], source="a.md")

    knowledge.set_current_client("bob")

    assert knowledge_info() == {"sources": [], "chunks": 0}


def test_data_survives_restart():
    knowledge.set_current_client("alice")
    add_knowledge(["片段一", "片段二"], [[0.1, 0.2], [0.3, 0.4]], source="a.md")

    # 模拟"服务重启"：清空内存注册表，再重新取
    knowledge.forget_all()

    store = knowledge.get_store("alice")
    assert store.info() == {"sources": ["a.md"], "chunks": 2}
    assert len(store.vectors) == 2
    assert store.vectors[0].shape == (2,)


def test_clear_removes_persisted_files():
    knowledge.set_current_client("alice")
    add_knowledge(["x"], [[0.1]], source="a.md")
    directory = knowledge.get_store("alice").directory
    assert directory.exists()

    clear_knowledge()

    assert knowledge_info() == {"sources": [], "chunks": 0}
    assert not any(directory.iterdir())


def test_normalize_client_id_blocks_unsafe_characters():
    assert knowledge.normalize_client_id("../../etc/passwd") == "etcpasswd"
    assert knowledge.normalize_client_id("") == "default"
    assert knowledge.normalize_client_id(None) == "default"
    assert knowledge.normalize_client_id("user_1") == "user_1"


def test_load_returns_false_without_saved_data():
    assert knowledge.KnowledgeBase("nobody").load() is False
