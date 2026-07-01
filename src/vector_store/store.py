from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle
from functools import lru_cache
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from src.config import (
    PROCESSED_DIR, CHROMA_DIR, RRF_K,
    HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH,
    CACHE_MAXSIZE, RETRY_SYNONYMS,
)

COLLECTION_NAME = "maritime_law_v2"


import re
import math
import threading
from collections import Counter
from functools import lru_cache


def _tokenize(text: str) -> List[str]:
    try:
        import jieba
        words = list(jieba.cut(text))
        return words + list(text)
    except ImportError:
        return list(text)


@lru_cache(maxsize=2048)
def _char_ngrams(text: str, n: int = 6) -> tuple:
    cleaned = re.sub(r"\s+", "", text)
    ngrams = set()
    for i in range(len(cleaned)):
        for size in range(1, min(n, len(cleaned) - i + 1) + 1):
            ngrams.add(cleaned[i:i+size])
    return tuple(sorted(ngrams))


def _ngram_tfidf_similarity(query: str, document: str) -> float:
    q_ngrams = set(_char_ngrams(query))
    d_ngrams = set(_char_ngrams(document))
    if not q_ngrams or not d_ngrams:
        return 0.0
    intersection = q_ngrams & d_ngrams
    if not intersection:
        return 0.0
    tfidf_denom = len(d_ngrams) * len(q_ngrams)
    if tfidf_denom == 0:
        return 0.0
    return math.log2(1 + len(intersection)) * len(intersection) / math.sqrt(tfidf_denom)


def _build_synonym_expansions(query: str) -> List[str]:
    queries = [query]
    for pair in RETRY_SYNONYMS:
        if ":" not in pair:
            continue
        a, b = pair.split(":", 1)
        if a in query:
            queries.append(query.replace(a, b))
        if b in query:
            queries.append(query.replace(b, a))
    return list(set(queries))


def _get_embedding_fn():
    from src.config import EMBEDDING_PROVIDER, EMBEDDING_MODEL, HF_ENDPOINT
    if EMBEDDING_PROVIDER == "sentence_transformer":
        try:
            import os
            os.environ["HF_ENDPOINT"] = HF_ENDPOINT
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        except Exception as e:
            print(f"[store] SentenceTransformer failed ({e}), falling back to ONNX")
    return ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


class HybridVectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.parents: Dict[int, Document] = {}
        self.children: List[Document] = []
        self._child_to_parent: List[int] = []
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embedding_fn = _get_embedding_fn()
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
                metadata={
                "hnsw:space": "cosine",
            },
        )
        self._next_child_id = 0

    def add_documents(self, parent_chunks: List[dict], child_chunks: List[dict]):
        if not child_chunks:
            return

        for chunk in parent_chunks:
            pid = chunk["parent_id"]
            self.parents[pid] = Document(
                page_content=chunk["text"],
                metadata={"source": chunk["source"], "parent_id": pid, "version": chunk.get("version", "1.0")},
            )

        child_texts = []
        chroma_ids = []
        chroma_metadatas = []
        for chunk in child_chunks:
            pid = chunk["parent_id"]
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["source"],
                    "child_id": chunk["child_id"],
                    "parent_id": pid,
                    "version": chunk.get("version", "1.0"),
                },
            )
            self.children.append(doc)
            self._child_to_parent.append(pid)
            child_texts.append(chunk["text"])
            chroma_ids.append(str(self._next_child_id))
            chroma_metadatas.append({
                "parent_id": pid,
                "source": chunk["source"],
                "version": chunk.get("version", "1.0"),
            })
            self._next_child_id += 1

        self._bm25_corpus.extend(child_texts)
        tokenized = [_tokenize(t) for t in child_texts]
        if self.bm25 is None:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = BM25Okapi(self.bm25.corpus + tokenized)

        self.collection.add(
            documents=child_texts,
            metadatas=chroma_metadatas,
            ids=chroma_ids,
        )

    def update_law(self, law_name: str, new_parent_chunks: List[dict], new_child_chunks: List[dict]):
        delete_ids = [
            str(i)
            for i, c in enumerate(self.children)
            if c.metadata.get("source") == law_name
        ]
        if delete_ids:
            self.collection.delete(ids=delete_ids)

        keep_children = [c for c in self.children if c.metadata.get("source") != law_name]
        keep_parents = {pid: doc for pid, doc in self.parents.items() if doc.metadata.get("source") != law_name}

        self.children = keep_children
        self.parents = keep_parents
        self._child_to_parent = [c.metadata["parent_id"] for c in self.children]

        if self._bm25_corpus:
            keep_indices = [
                i for i, c in enumerate(self.children)
                if c.metadata.get("source") != law_name
            ]
            self._bm25_corpus = [self._bm25_corpus[i] for i in keep_indices]

        bm25_was_none = self.bm25 is None
        if not bm25_was_none and self._bm25_corpus:
            self.bm25 = BM25Okapi([_tokenize(t) for t in self._bm25_corpus])
        elif not self._bm25_corpus:
            self.bm25 = None

        self.add_documents(new_parent_chunks, new_child_chunks)

    def delete_law(self, law_name: str):
        delete_ids = [
            str(i)
            for i, c in enumerate(self.children)
            if c.metadata.get("source") == law_name
        ]
        if delete_ids:
            self.collection.delete(ids=delete_ids)

        self.children = [c for c in self.children if c.metadata.get("source") != law_name]
        self.parents = {pid: doc for pid, doc in self.parents.items() if doc.metadata.get("source") != law_name}
        self._child_to_parent = [c.metadata["parent_id"] for c in self.children]
        self._bm25_corpus = [c.page_content for c in self.children]
        if self._bm25_corpus:
            self.bm25 = BM25Okapi([_tokenize(t) for t in self._bm25_corpus])
        else:
            self.bm25 = None

    def _child_results_to_parents(
        self, child_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        parent_best: Dict[int, Tuple[Document, float]] = {}
        for child_doc, score in child_results:
            pid = child_doc.metadata["parent_id"]
            if pid not in parent_best or score > parent_best[pid][1]:
                parent_best[pid] = (self.parents.get(pid, child_doc), score)
        results = sorted(parent_best.values(), key=lambda x: x[1], reverse=True)
        return results

    def _ngram_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        if not self.parents:
            return []
        query_lower = query.lower()
        scores = []
        for pid, doc in self.parents.items():
            sim = _ngram_tfidf_similarity(query_lower, doc.page_content.lower())
            if sim > 0:
                scores.append((doc, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    @lru_cache(maxsize=CACHE_MAXSIZE)
    def _cached_dense_query(self, query: str) -> str:
        return query

    def _dense_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        self._cached_dense_query(query)
        n_results = min(k * HNSW_EF_SEARCH // 50, self._next_child_id or 1)
        n_results = max(n_results, min(k, self._next_child_id or 1))
        n_results = min(n_results, self._next_child_id or 1)
        if n_results == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results["ids"] or not results["ids"][0]:
            return []
        child_results = []
        for idx_str, dist, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            score = 1.0 - float(dist)
            child_doc = Document(
                page_content="",
                metadata={"parent_id": meta["parent_id"], "source": meta.get("source", "unknown")},
            )
            child_results.append((child_doc, score))
        return self._child_results_to_parents(child_results)

    def _bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        if not self.bm25 or not self.children:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-k:][::-1]
        child_results = [
            (self.children[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]
        return self._child_results_to_parents(child_results)

    def hybrid_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in query)

        if has_chinese:
            ngram_results = self._ngram_search(query, k=k * 3)
            bm25_results = self._bm25_search(query, k=k * 3)

            bm25_weight, ngram_weight = 0.6, 0.4

            if not bm25_results and not ngram_results:
                return []
            if not bm25_results:
                return ngram_results[:k]
            if not ngram_results:
                return bm25_results[:k]

            bm25_by_pid = {doc.metadata["parent_id"]: (doc, score) for doc, score in bm25_results}
            ngram_by_pid = {doc.metadata["parent_id"]: (doc, score) for doc, score in ngram_results}

            all_pids = set(bm25_by_pid.keys()) | set(ngram_by_pid.keys())
            rrf_scores: List[Tuple[Document, float]] = []
            for pid in all_pids:
                doc = bm25_by_pid.get(pid, ngram_by_pid.get(pid))[0]
                bm25_rank = next(
                    (r for r, (d, _) in enumerate(bm25_results) if d.metadata["parent_id"] == pid),
                    None,
                )
                ngram_rank = next(
                    (r for r, (d, _) in enumerate(ngram_results) if d.metadata["parent_id"] == pid),
                    None,
                )
                score = 0.0
                if bm25_rank is not None:
                    score += bm25_weight / (RRF_K + bm25_rank + 1)
                if ngram_rank is not None:
                    score += ngram_weight / (RRF_K + ngram_rank + 1)
                rrf_scores.append((doc, score))
            rrf_scores.sort(key=lambda x: x[1], reverse=True)
            if not rrf_scores:
                result = self._bm25_search(query, k=k)
                if result:
                    return result
                return self._retry_with_synonyms(query, k)
            return rrf_scores[:k]

        dense_results = self._dense_search(query, k=k * 3)
        bm25_results = self._bm25_search(query, k=k * 3)

        bm25_weight, dense_weight = 0.5, 0.5

        if not dense_results and not bm25_results:
            return []
        if not dense_results:
            return bm25_results[:k]
        if not bm25_results:
            return dense_results[:k]

        dense_by_pid = {doc.metadata["parent_id"]: (doc, score) for doc, score in dense_results}
        bm25_by_pid = {doc.metadata["parent_id"]: (doc, score) for doc, score in bm25_results}

        all_pids = set(dense_by_pid.keys()) | set(bm25_by_pid.keys())
        rrf_scores: List[Tuple[Document, float]] = []
        for pid in all_pids:
            doc = dense_by_pid.get(pid, bm25_by_pid.get(pid))[0]
            dense_rank = next(
                (r for r, (d, _) in enumerate(dense_results) if d.metadata["parent_id"] == pid),
                None,
            )
            bm25_rank = next(
                (r for r, (d, _) in enumerate(bm25_results) if d.metadata["parent_id"] == pid),
                None,
            )
            score = 0.0
            if dense_rank is not None:
                score += dense_weight / (RRF_K + dense_rank + 1)
            if bm25_rank is not None:
                score += bm25_weight / (RRF_K + bm25_rank + 1)
            rrf_scores.append((doc, score))
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        if not rrf_scores:
            return self._retry_with_synonyms(query, k)
        return rrf_scores[:k]

    def _retry_with_synonyms(self, query: str, k: int) -> List[Tuple[Document, float]]:
        expanded = _build_synonym_expansions(query)
        for eq in expanded:
            if eq == query:
                continue
            results = self._dense_search(eq, k=k)
            if results:
                return results
            results = self._bm25_search(eq, k=k)
            if results:
                return results
        return []

    def get_index_stats(self) -> dict:
        count = self.collection.count()
        return {
            "parent_chunks": len(self.parents),
            "child_chunks": len(self.children),
            "chroma_entries": count,
            "bm25_corpus_size": len(self._bm25_corpus),
            "sources": list(set(d.metadata.get("source", "unknown") for d in self.parents.values())),
            "hnsw_params": {"M": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": HNSW_EF_SEARCH},
        }

    def save(self):
        path = PROCESSED_DIR / "parent_child_store.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "parents": self.parents,
                    "children": self.children,
                    "child_to_parent": self._child_to_parent,
                    "bm25_corpus": self._bm25_corpus,
                    "next_child_id": self._next_child_id,
                },
                f,
            )

    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        return self._bm25_search(query, k=k)

    def get_parents_by_source(self, law: str) -> Dict[int, Document]:
        return {
            pid: doc
            for pid, doc in self.parents.items()
            if law.lower() in doc.metadata.get("source", "").lower()
        }

    def load(self) -> bool:
        path = PROCESSED_DIR / "parent_child_store.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.parents = data["parents"]
        self.children = data["children"]
        self._child_to_parent = data["child_to_parent"]
        self._bm25_corpus = data["bm25_corpus"]
        self._next_child_id = data["next_child_id"]
        if self._bm25_corpus:
            tokenized = [_tokenize(t) for t in self._bm25_corpus]
            self.bm25 = BM25Okapi(tokenized)
        return True


_store = None
_store_lock = threading.Lock()


def get_store():
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = HybridVectorStore()
                loaded = _store.load()
                if not loaded:
                    print("[store] No existing store found, please run build_knowledge_base.py first")
    return _store


def build_store(parent_chunks: List[dict], child_chunks: List[dict]) -> HybridVectorStore:
    store = HybridVectorStore()
    store.add_documents(parent_chunks, child_chunks)
    store.save()
    stats = store.get_index_stats()
    print(f"[store] Built parent-child hybrid store: {stats['parent_chunks']} parents, {stats['child_chunks']} children")
    print(f"[store] Sources ({len(stats['sources'])}): {stats['sources']}")
    return store


def search_store(query: str, k: int = 5, hybrid: bool = True) -> List[Tuple[Document, float]]:
    store = get_store()
    if hybrid:
        return store.hybrid_search(query, k=k)
    return store._bm25_search(query, k=k)
