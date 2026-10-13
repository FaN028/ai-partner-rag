# ============================================================
# conftest.py —— 全测试共用的夹具
# ============================================================
import pytest

import config
import knowledge


@pytest.fixture(autouse=True)
def isolated_knowledge_dir(tmp_path, monkeypatch):
    """把知识库落盘目录指向临时目录，并清空内存注册表。

    这样每个用例都在干净的环境里跑，不会污染仓库的 data/ 目录，
    也不会互相看到彼此写入的知识库。
    """
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path / "knowledge")
    knowledge.forget_all()
    yield
    knowledge.forget_all()
