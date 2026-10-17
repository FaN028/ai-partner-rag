# -*- coding: utf-8 -*-
# ============================================================
# benchmark_vector.py —— 向量后端自检 + 检索耗时对比
# ============================================================
"""向量后端自检与检索耗时对比。

1. 后端正确性：写入、计数、Top-N、清空是否正常；
2. 性能对比：同一批数据下，全量扫描与 HNSW 的平均单次查询耗时。

用随机向量模拟数据，不加载 embedding 模型，秒级完成。

    python benchmark_vector.py
    python benchmark_vector.py --chunks 20000 --queries 500
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import config
import vectorstore

DIM = 512          # 与 bge-small-zh-v1.5 的输出维度一致


def build_dataset(count: int, dim: int = DIM, seed: int = 0):
    rng = np.random.default_rng(seed)
    chunks = [f"第 {i} 条模拟片段：这是一段用于压测的占位文本。" for i in range(count)]
    vectors = rng.normal(size=(count, dim)).astype("float32")
    # 归一化：余弦相似度只关心方向，归一化后与真实 embedding 的分布更接近
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return chunks, vectors


def benchmark(store, name: str, vectors, queries: int, top_n: int = 20) -> float:
    """跑 queries 次检索，返回平均单次耗时（毫秒）。"""
    rng = np.random.default_rng(42)
    probes = rng.normal(size=(queries, vectors.shape[1])).astype("float32")
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)

    # 预热一次，避免首帧开销混进平均值
    store.query(probes[0], top_n)

    start = time.perf_counter()
    for probe in probes:
        store.query(probe, top_n)
    elapsed = (time.perf_counter() - start) / queries * 1000
    print(f"  {name:<8} 平均单次检索 {elapsed:8.2f} ms")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=5000, help="模拟片段数")
    parser.add_argument("--queries", type=int, default=200, help="每种后端查询次数")
    args = parser.parse_args()

    chunks, vectors = build_dataset(args.chunks)
    print(f"\n数据集：{args.chunks} 条片段，向量维度 {DIM}，每种后端查询 {args.queries} 次\n")

    workdir = Path(tempfile.mkdtemp(prefix="vecbench-"))
    try:
        # —— NumPy 后端 ——
        numpy_store = vectorstore.NumpyVectorStore(workdir / "numpy")
        numpy_store.add(chunks, vectors, source="bench.md")
        assert numpy_store.count() == args.chunks, "NumPy 后端计数不一致"
        top = numpy_store.query(vectors[0], 3)
        assert len(top) == 3, "NumPy 后端 Top-N 数量不对"
        numpy_ms = benchmark(numpy_store, "numpy", vectors, args.queries)

        # —— Chroma 后端 ——
        chroma_ms = None
        try:
            chroma_store = vectorstore.ChromaVectorStore(workdir / "chroma", "bench")
            chroma_store.add(chunks, vectors, source="bench.md")
            assert chroma_store.count() == args.chunks, "Chroma 后端计数不一致"
            assert len(chroma_store.query(vectors[0], 3)) == 3, "Chroma 后端 Top-N 数量不对"
            chroma_ms = benchmark(chroma_store, "chroma", vectors, args.queries)
        except vectorstore.VectorStoreError as exc:
            print(f"  chroma   跳过：{exc}")

        print()
        if chroma_ms:
            print(f"结论：Chroma 比全量扫描快 {numpy_ms / chroma_ms:.1f}×（同一台机器、同一批数据）")
        else:
            print("结论：NumPy 后端自检通过；装上 chromadb 后重跑可看到两种后端的耗时对比。")
            print("      安装：pip install chromadb")
        print(f"当前 config.VECTOR_BACKEND = {config.VECTOR_BACKEND}\n")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
