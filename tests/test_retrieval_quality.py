# ============================================================
# test_retrieval_quality.py —— 检索质量回归测试（需要本地模型）
# ============================================================
"""检索质量回归测试。

需要真实 bge 模型（约 1.2G），因此与其余测试分层：

- 默认 `pytest`                  → 跳过 slow，保持秒级反馈
- 改动检索链路后 `pytest -m slow` → 跑真实模型，确认检索质量未退化

两者目的不同：常规用例用替身验证逻辑，本组验证语义检索的实际效果。
"""
from pathlib import Path

import pytest

import evaluate

pytestmark = pytest.mark.slow

MODEL_DIRS = ("models/bge-small-zh-v1.5", "models/bge-reranker-base")


@pytest.fixture(scope="module", autouse=True)
def require_local_models():
    """缺模型时跳过而不是报错——CI 上不一定有这几个 G 的文件。"""
    missing = [d for d in MODEL_DIRS if not Path(d).exists()]
    if missing:
        pytest.skip(f"缺少本地模型：{', '.join(missing)}（先跑 download_model.py）")


def test_vector_retrieval_hits_most_questions():
    """语义检索要对「换了个说法」的问题也命中，命中率不低于 90%。"""
    assert evaluate.hit_rate(evaluate.vector_retrieve) >= 0.9


def test_rerank_does_not_hurt_recall():
    """精排只能提升排序，不应该把原本能召回的正确答案挤出去。"""
    vector_rate = evaluate.hit_rate(evaluate.vector_retrieve)
    rerank_rate = evaluate.hit_rate(evaluate.rerank_retrieve)

    assert rerank_rate >= vector_rate


def test_semantic_retrieval_beats_keyword():
    """这是整个项目的前提假设：语义检索优于关键词匹配。

    这条断言一旦失败，说明切块或 embedding 环节出了问题——
    要么模型没加载对，要么文本被截断了。
    """
    keyword_rate = evaluate.hit_rate(evaluate.keyword_retrieve)
    vector_rate = evaluate.hit_rate(evaluate.vector_retrieve)

    assert vector_rate > keyword_rate


@pytest.mark.parametrize("retrieve_fn", [evaluate.vector_retrieve, evaluate.rerank_retrieve])
def test_returns_at_most_top_k(retrieve_fn):
    """返回条数不能超过约定的 Top-K，且不能有重复下标。"""
    result = retrieve_fn(evaluate.TESTS[0][0])

    assert 0 < len(result) <= evaluate.TOP_K
    assert len(set(result)) == len(result)
