# -*- coding: utf-8 -*-
# ============================================================
# evaluate.py —— 评测三种检索方法的命中率
# ============================================================
"""目标：用「命中率」这个数据，证明哪种检索更好。

两种用法：

    python evaluate.py          # 打印对比表，给人看结果
    pytest -m slow              # 作为回归测试跑（见 tests/test_retrieval_quality.py）

模型按需加载、不放在模块顶层：测试收集阶段会 import 本模块，
若在导入时就加载 1.2G 模型，默认的快速测试（8 秒）会被拖成一分多钟。
"""
import re
from functools import lru_cache

import numpy as np

TOP_K = 3

# —— 知识库（10 个片段，模拟"公司手册"）——
CHUNKS = [
    "员工每年享有带薪年假5天，须在年底前使用完毕",     # 0
    "报销单需要部门经理签字后提交给财务部",           # 1
    "考勤打卡时间是每天早上九点到下午六点",           # 2
    "绩效评估每半年进行一次，结果影响年终奖金",        # 3
    "加班需要提前一天向主管申请",                     # 4
    "新员工试用期为三个月",                           # 5
    "病假需要提供医院证明",                           # 6
    "出差补贴标准是每天200元",                        # 7
    "离职需要提前一个月通知",                         # 8
    "年终奖在春节前发放",                             # 9
]

# —— 测试集：(问题, 正确答案片段的序号) ——
# 注意：前几题故意用"不同说法"，看关键词能不能搜到
TESTS = [
    ("一年能休息多少天", 0),     # "年假" vs "休息" —— 字不同
    ("报销找谁签字", 1),
    ("几点开始上班", 2),         # "上班" vs "打卡" —— 字不同
    ("多久考核一次", 3),         # "考核" vs "评估" —— 字不同
    ("加班要提前申请吗", 4),
    ("试用期多久", 5),
    ("生病请假要什么证明", 6),   # "生病" vs "病假" —— 字不同
    ("出差每天补贴多少", 7),
    ("辞职要提前多久说", 8),     # "辞职" vs "离职" —— 字不同
    ("年终奖什么时候发", 9),
]


@lru_cache(maxsize=1)
def load_models():
    """加载 embedding 与 rerank 模型（只加载一次）。"""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    return (
        SentenceTransformer("models/bge-small-zh-v1.5"),
        CrossEncoder("models/bge-reranker-base"),
    )


@lru_cache(maxsize=4)
def chunk_vectors(chunks: tuple):
    """片段向量只算一次，三种方法共用。"""
    return load_models()[0].encode(list(chunks))


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# —— 方法1：关键词检索 ——
def tokenize(text: str) -> list[str]:
    text = text.lower()
    terms = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def keyword_retrieve(query: str, chunks=None, top_k: int = TOP_K) -> list[int]:
    chunks = CHUNKS if chunks is None else chunks
    query_terms = set(tokenize(query))
    scored = []
    for idx, chunk in enumerate(chunks):
        chunk_terms = tokenize(chunk)
        hits = sum(1 for t in query_terms if t in chunk_terms)
        scored.append((hits / (len(chunk_terms) ** 0.5), idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for score, idx in scored[:top_k] if score > 0]


# —— 方法2：向量检索 ——
def vector_retrieve(query: str, chunks=None, top_k: int = TOP_K) -> list[int]:
    chunks = CHUNKS if chunks is None else chunks
    query_vector = load_models()[0].encode(query)
    vectors = chunk_vectors(tuple(chunks))
    scored = sorted(
        [(cosine(query_vector, v), i) for i, v in enumerate(vectors)],
        key=lambda x: -x[0],
    )
    return [i for _, i in scored[:top_k]]


# —— 方法3：向量 + rerank ——
def rerank_retrieve(query: str, chunks=None, coarse: int = 6, top_k: int = TOP_K) -> list[int]:
    chunks = CHUNKS if chunks is None else chunks
    query_vector = load_models()[0].encode(query)
    vectors = chunk_vectors(tuple(chunks))
    scored = sorted(
        [(cosine(query_vector, v), i) for i, v in enumerate(vectors)],
        key=lambda x: -x[0],
    )
    candidate_idx = [i for _, i in scored[:coarse]]
    scores = load_models()[1].predict([[query, chunks[i]] for i in candidate_idx])
    ranked = sorted(zip(candidate_idx, scores), key=lambda x: -x[1])
    return [i for i, _ in ranked[:top_k]]


def hit_rate(retrieve_fn, tests=None, top_k: int = TOP_K) -> float:
    """命中率：正确答案是否落在 Top-K 结果里。"""
    tests = TESTS if tests is None else tests
    hits = sum(1 for question, correct in tests if correct in retrieve_fn(question))
    return hits / len(tests)


def main() -> None:
    print("加载模型...")
    load_models()
    print("完成\n")

    print("=" * 50)
    print("测试题数：", len(TESTS), "，每次检索 Top", TOP_K)
    print("=" * 50)
    print(f"1) 关键词检索   命中率：{hit_rate(keyword_retrieve):.0%}")
    print(f"2) 向量检索     命中率：{hit_rate(vector_retrieve):.0%}")
    print(f"3) 向量 + rerank 命中率：{hit_rate(rerank_retrieve):.0%}")


if __name__ == "__main__":
    main()
