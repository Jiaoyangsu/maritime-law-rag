from src.data_collection.collector import collect_all
from src.document_processing.chunker import chunk_all
from src.vector_store.store import build_store


def main():
    print("=" * 50)
    print("Step 1: Collecting maritime law data...")
    print("=" * 50)
    texts = collect_all()

    print("\n" + "=" * 50)
    print("Step 2: Parent-Child chunking...")
    print("=" * 50)
    parent_chunks, child_chunks = chunk_all(texts)

    print("\n" + "=" * 50)
    print("Step 3: Building ChromaDB (HNSW) + BM25 hybrid store...")
    print("=" * 50)
    store = build_store(parent_chunks, child_chunks)

    stats = store.get_index_stats()
    print(f"\n[SUCCESS] Knowledge base built!")
    print(f"  Parent chunks: {stats['parent_chunks']}")
    print(f"  Child chunks: {stats['child_chunks']}")
    print(f"  ChromaDB entries: {stats['chroma_entries']}")
    print(f"  Sources: {len(stats['sources'])}")
    for s in stats['sources']:
        print(f"    - {s}")
    print(f"  Retrieval: ChromaDB HNSW (child-level) + BM25 (child-level) -> RRF -> map to parent")
    print(f"  Embedding: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')} via {os.getenv('EMBEDDING_PROVIDER', 'openai')}")
    print(f"  Reranker: {os.getenv('CROSS_ENCODER_MODEL', 'BAAI/bge-reranker-v2-m3')} (skip for Chinese if not cached)")
    print("Ready for Q&A!")
    print("\nTest commands:")
    print("  PYTHONPATH=. python scripts/run_agent.py")
    print("  PYTHONPATH=. python scripts/run_agent.py '船舶碰撞的法律规定是什么'")


if __name__ == "__main__":
    main()
