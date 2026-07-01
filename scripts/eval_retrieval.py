#!/usr/bin/env python3
"""Eval script: run retrieval benchmark against human-annotated queries."""
import json
import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vector_store.store import get_store
from src.rag.retriever import MaritimeLawRetriever
from src.config import TOP_K


def load_queries(path: str = "data/eval/queries.json") -> list:
    with open(path) as f:
        return json.load(f)


def evaluate():
    queries = load_queries()
    print(f"Loaded {len(queries)} eval queries\n")

    retriever = MaritimeLawRetriever(top_k=TOP_K, rerank=False)

    total_recall = 0.0
    total_mrr = 0.0
    per_type = {}

    for i, item in enumerate(queries, 1):
        q = item["query"]
        expected = item["expected_sources"]
        qtype = item.get("type", "retrieve")

        results = retriever.retrieve(q)
        retrieved = [doc.metadata.get("source", "") for doc, _ in results]

        found = 0
        first_rank = []
        for exp in expected:
            for rank, src in enumerate(retrieved, 1):
                if exp in src:
                    found += 1
                    first_rank.append(rank)
                    break

        recall = found / len(expected)
        mrr = sum(1.0 / r for r in first_rank) / len(expected) if first_rank else 0.0

        total_recall += recall
        total_mrr += mrr

        if qtype not in per_type:
            per_type[qtype] = {"count": 0, "recall": 0.0, "mrr": 0.0}
        per_type[qtype]["count"] += 1
        per_type[qtype]["recall"] += recall
        per_type[qtype]["mrr"] += mrr

        status = "PASS" if found == len(expected) else "FAIL"
        print(f"[{status}] #{i:2d} {q[:50]:50s} recall={recall:.2f} mrr={mrr:.3f} expected={expected} got={retrieved[:3]}")

    n = len(queries)
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS (k={TOP_K})")
    print(f"{'='*60}")
    print(f"  Overall Recall@{TOP_K}: {total_recall/n:.4f}")
    print(f"  Overall MRR:          {total_mrr/n:.4f}")
    print()

    for qtype, data in sorted(per_type.items()):
        avg_recall = data["recall"] / data["count"]
        avg_mrr = data["mrr"] / data["count"]
        print(f"  [{qtype:10s}] count={data['count']:2d} recall={avg_recall:.4f} mrr={avg_mrr:.4f}")

    return total_recall / n, total_mrr / n


if __name__ == "__main__":
    evaluate()
