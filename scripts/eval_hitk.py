#!/usr/bin/env python3
"""Hit@K evaluation: for each query, whether ANY expected source appears in top-K."""
import json
import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vector_store.store import get_store


def load_queries(path: str = "data/eval/queries.json") -> list:
    with open(path) as f:
        return json.load(f)


def hit_at_k(retrieved_sources: list, expected: list, k: int) -> bool:
    top_k = retrieved_sources[:k]
    for exp in expected:
        if any(exp in src for src in top_k):
            return True
    return False


def evaluate():
    queries = load_queries()
    print(f"Loaded {len(queries)} eval queries\n")

    store = get_store()

    Ks = [1, 3, 5, 10]
    methods = {
        "Hybrid": lambda q: [d.metadata.get("source", "") for d, _ in store.hybrid_search(q, k=max(Ks))],
        "Dense":  lambda q: [d.metadata.get("source", "") for d, _ in store._dense_search(q, k=max(Ks))],
        "BM25":   lambda q: [d.metadata.get("source", "") for d, _ in store._bm25_search(q, k=max(Ks))],
        "Ngram":  lambda q: [d.metadata.get("source", "") for d, _ in store._ngram_search(q, k=max(Ks))],
    }

    # header
    header = f"{'Method':<8s}" + "".join(f" Hit@{k:>2d}" for k in Ks)
    print(header)
    print("-" * len(header))

    by_type = {m: {} for m in methods}
    for m_name, search_fn in methods.items():
        total = [0] * len(Ks)
        type_hits = {}
        for item in queries:
            q = item["query"]
            expected = item["expected_sources"]
            qtype = item.get("type", "retrieve")

            retrieved = search_fn(q)
            for ki, k in enumerate(Ks):
                if hit_at_k(retrieved, expected, k):
                    total[ki] += 1

            if qtype not in type_hits:
                type_hits[qtype] = [0] * len(Ks)
            for ki, k in enumerate(Ks):
                if hit_at_k(retrieved, expected, k):
                    type_hits[qtype][ki] += 1

        n = len(queries)
        vals = " ".join(f"{t/n*100:5.1f}%" for t in total)
        print(f"{m_name:<8s} {vals}")
        by_type[m_name] = type_hits

    # per-type breakdown
    print(f"\n--- Per-type Hit@{max(Ks)} ---")
    types_in_data = set()
    for item in queries:
        types_in_data.add(item.get("type", "retrieve"))
    for qtype in sorted(types_in_data):
        count = sum(1 for item in queries if item.get("type", "retrieve") == qtype)
        print(f"\n  [{qtype}] (n={count}):")
        for m_name in methods:
            vals = " ".join(f"{by_type[m_name][qtype][ki]/count*100:5.1f}%" for ki, k in enumerate(Ks))
            print(f"    {m_name:<8s} {vals}")


if __name__ == "__main__":
    evaluate()
