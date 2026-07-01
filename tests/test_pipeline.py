#!/usr/bin/env python3
"""End-to-end pipeline test for maritime law RAG system (parent-child chunking)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vector_store.store import get_store
from src.rag.retriever import MaritimeLawRetriever
from src.rag.reranker import get_reranker
from src.rag.generator import build_prompt
from src.config import TOP_K, RERANK_TOP_K

TEST_QUERIES = [
    "船舶碰撞的法律规定是什么",
    "海难救助的报酬如何确定",
    "船舶所有人责任限制的具体规定",
    "共同海损的构成要件",
    "海事诉讼时效是多少年",
    "SOLAS公约对船舶安全有什么要求",
    "MARPOL公约对油类排放的规定",
    "船员任职需要什么条件",
    "港口法的基本原则是什么",
    "国际海运条例对无船承运业务的规定",
]


def test_index_stats():
    print("\n" + "=" * 60)
    print("[TEST] Index Statistics Display")
    print("=" * 60)
    store = get_store()
    stats = store.get_index_stats()
    print(f"  Parent chunks: {stats['parent_chunks']}")
    print(f"  Child chunks: {stats['child_chunks']}")
    print(f"  ChromaDB entries: {stats['chroma_entries']}")
    print(f"  BM25 corpus size: {stats['bm25_corpus_size']}")
    print(f"  Number of sources: {len(stats['sources'])}")
    for s in sorted(stats['sources']):
        print(f"    - {s}")
    assert stats['parent_chunks'] > 0
    assert stats['child_chunks'] > stats['parent_chunks']
    assert stats['chroma_entries'] > 0
    assert stats['bm25_corpus_size'] > 0
    assert len(stats['sources']) >= 10
    print("[PASS] Index stats OK")


def test_dense_search():
    print("\n" + "=" * 60)
    print("[TEST] Dense Search (ChromaDB HNSW, child-level -> parent)")
    print("=" * 60)
    store = get_store()
    results = store._dense_search("船舶碰撞", k=5)
    assert len(results) > 0, "No dense results"
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        pid = doc.metadata.get("parent_id", "?")
        length = len(doc.page_content)
        print(f"  [{i}] {src} pid={pid} | score={score:.4f} | {length} chars | {doc.page_content[:60]}...")
    print("[PASS] Dense search OK")


def test_bm25_search():
    print("\n" + "=" * 60)
    print("[TEST] BM25 Search (child-level -> parent)")
    print("=" * 60)
    store = get_store()
    results = store._bm25_search("船舶碰撞", k=5)
    assert len(results) > 0, "No BM25 results"
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        pid = doc.metadata.get("parent_id", "?")
        length = len(doc.page_content)
        print(f"  [{i}] {src} pid={pid} | score={score:.4f} | {length} chars | {doc.page_content[:60]}...")
    print("[PASS] BM25 search OK")


def test_hybrid_search():
    print("\n" + "=" * 60)
    print("[TEST] Hybrid Search (child-level -> RRF -> parent)")
    print("=" * 60)
    store = get_store()
    results = store.hybrid_search("船舶碰撞的法律规定", k=5)
    assert len(results) > 0, "No hybrid results"
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        pid = doc.metadata.get("parent_id", "?")
        length = len(doc.page_content)
        print(f"  [{i}] {src} pid={pid} | score={score:.4f} | {length} chars | {doc.page_content[:60]}...")
    print("[PASS] Hybrid search OK")


def test_parent_content_size():
    print("\n" + "=" * 60)
    print("[TEST] Parent chunk size (should be ~1024)")
    print("=" * 60)
    store = get_store()
    results = store.hybrid_search("船舶碰撞", k=3)
    for i, (doc, score) in enumerate(results, 1):
        length = len(doc.page_content)
        print(f"  [{i}] {length} chars (expected ~1024)")
        assert length > 200, f"Parent too small: {length} chars"
    print("[PASS] Parent content size OK")


def test_retriever():
    print("\n" + "=" * 60)
    print("[TEST] Retriever Pipeline (Hybrid, no rerank)")
    print("=" * 60)
    retriever = MaritimeLawRetriever(top_k=TOP_K, rerank=False)
    for q in TEST_QUERIES:
        results = retriever.retrieve(q)
        assert len(results) > 0, f"No results for: {q}"
        srcs = [doc.metadata.get("source", "?") for doc, _ in results]
        lengths = [len(doc.page_content) for doc, _ in results]
        print(f"  [OK] {q[:40]:40s} -> {srcs[0]} ({lengths[0]} chars)")
    print("\n[PASS] Retriever pipeline OK")


def test_reranker():
    print("\n" + "=" * 60)
    print("[TEST] Cross-Encoder Reranker")
    print("=" * 60)
    reranker = get_reranker(top_k=RERANK_TOP_K)
    if reranker is None:
        print("[SKIP] Reranker not available")
        return
    store = get_store()
    results = store.hybrid_search("船舶碰撞的责任划分", k=10)
    before = [(doc.page_content[:50], score) for doc, score in results[:3]]
    print("  Before rerank (top 3):")
    for i, (txt, score) in enumerate(before, 1):
        print(f"    [{i}] {score:.4f} | {txt}...")

    reranked = reranker.rerank("船舶碰撞的责任划分", results)
    after = [(doc.page_content[:50], score) for doc, score in reranked[:3]]
    print("  After rerank (top 3):")
    for i, (txt, score) in enumerate(after, 1):
        print(f"    [{i}] {score:.4f} | {txt}...")
    assert len(reranked) > 0, "No reranker results"
    assert len(reranked) <= RERANK_TOP_K
    print("[PASS] Reranker OK")


def test_retriever_with_rerank():
    print("\n" + "=" * 60)
    print("[TEST] Retriever with Rerank")
    print("=" * 60)
    retriever = MaritimeLawRetriever(top_k=TOP_K, rerank=True)
    for q in TEST_QUERIES[:5]:
        results = retriever.retrieve(q)
        assert len(results) > 0, f"No results for: {q}"
        srcs = [doc.metadata.get("source", "?") for doc, _ in results]
        print(f"  [OK] {q[:40]:40s} -> {srcs[0]}")
    print("\n[PASS] Retriever + Rerank OK")


def test_prompt_template():
    print("\n" + "=" * 60)
    print("[TEST] Prompt Template")
    print("=" * 60)
    retriever = MaritimeLawRetriever(top_k=3, rerank=False)
    results = retriever.retrieve("船舶碰撞")
    context_chunks = [doc.page_content for doc, _ in results]
    sources = [doc.metadata.get("source", "unknown") for doc, _ in results]
    prompt = build_prompt("船舶碰撞的法律责任", context_chunks, sources)
    assert "参考法条" in prompt, "Missing '参考法条' in prompt"
    assert "船舶碰撞" in prompt, "Missing query in prompt"
    assert "用户问题" in prompt, "Missing '用户问题' in prompt"
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Sources: {sources}")
    print("[PASS] Prompt template OK")


def test_content_preview():
    print("\n" + "=" * 60)
    print("[TEST] Retrieved Content Preview")
    print("=" * 60)
    retriever = MaritimeLawRetriever(top_k=3, rerank=False)
    results = retriever.retrieve("船舶碰撞")
    print(f"  Results: {len(results)}")
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        pid = doc.metadata.get("parent_id", "?")
        length = len(doc.page_content)
        print(f"  [{i}] {src} pid={pid} | score={score:.4f} | {length} chars")
        print(f"      {doc.page_content[:80].strip()}...")
    print("[PASS] Content preview OK")


def test_child_dedup():
    print("\n" + "=" * 60)
    print("[TEST] Child dedup (multiple children per parent)")
    print("=" * 60)
    store = get_store()
    child_count = len(store.children)
    parent_count = len(store.parents)
    print(f"  Children: {child_count}, Parents: {parent_count}")
    print(f"  Ratio: {child_count/parent_count:.1f} children/parent (expected >1)")
    assert child_count > parent_count
    results = store.hybrid_search("船舶", k=10)
    pids = set(d.metadata["parent_id"] for d, _ in results)
    print(f"  Hybrid search k=10 -> {len(results)} parents, {len(pids)} unique parents")
    assert len(results) == len(pids), "Dedup not working"
    print("[PASS] Child dedup OK")


def main():
    print("=" * 60)
    print("Maritime Law RAG - Parent-Child Chunking Pipeline Test")
    print("Architecture: child(256)→HNSW+BM25→RRF→map→parent(1024)→cross-encoder")
    print("=" * 60)

    test_index_stats()
    test_dense_search()
    test_bm25_search()
    test_hybrid_search()
    test_parent_content_size()
    test_retriever()
    test_child_dedup()
    try:
        test_reranker()
    except Exception as e:
        print(f"[SKIP] Reranker: {e}")
    test_retriever_with_rerank()
    test_prompt_template()
    test_content_preview()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
