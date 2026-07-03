import os
import re
import math
from typing import List, Tuple, Set
from langchain.schema import Document
from src.config import CROSS_ENCODER_MODEL, HF_ENDPOINT


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text) if text else False)


def _tokenize(text: str) -> List[str]:
    try:
        import jieba
        return [w for w in jieba.cut(text) if len(w) > 1]
    except ImportError:
        return []


def _char_ngrams(text: str, n: int = 6) -> Set[str]:
    cleaned = re.sub(r"\s+", "", text)
    ngrams = set()
    for i in range(len(cleaned)):
        for size in range(1, min(n, len(cleaned) - i + 1) + 1):
            ngrams.add(cleaned[i:i+size])
    return ngrams


def _query_coverage(query: str, document: str) -> float:
    q_words = set(_tokenize(query))
    if not q_words:
        return 0.0
    doc_lower = document.lower()
    covered = sum(1 for w in q_words if w.lower() in doc_lower)
    return covered / len(q_words)


def _ngram_similarity(query: str, document: str) -> float:
    q_ngrams = _char_ngrams(query.lower())
    d_ngrams = _char_ngrams(document.lower())
    if not q_ngrams or not d_ngrams:
        return 0.0
    intersection = q_ngrams & d_ngrams
    if not intersection:
        return 0.0
    denom = len(d_ngrams) * len(q_ngrams)
    if denom == 0:
        return 0.0
    return math.log2(1 + len(intersection)) * len(intersection) / math.sqrt(denom)


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, top_k: int = 3):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
        self._load_model()

    def _load_model(self):
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception:
            self._model = None

    def rerank(
        self, query: str, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        if not results:
            return results

        has_chinese = _has_chinese(query)

        if self._model is not None:
            return self._cross_rerank(query, results)

        if has_chinese:
            return self._chinese_rerank(query, results)

        return results

    def _cross_rerank(
        self, query: str, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        pairs = [(query, doc.page_content) for doc, _ in results]
        scores = self._model.predict(pairs)
        scored = [(doc, float(score)) for (doc, _), score in zip(results, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]

    def _chinese_rerank(
        self, query: str, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        n = len(results)

        bm25_scores = [s for _, s in results]
        min_b, max_b = min(bm25_scores), max(bm25_scores)
        range_b = max_b - min_b if max_b != min_b else 1.0

        doc_has_cn = [_has_chinese(doc.page_content) for doc, _ in results]
        coverages = [_query_coverage(query, doc.page_content) for doc, _ in results]
        ngrams = [_ngram_similarity(query, doc.page_content) for doc, _ in results]

        scored = []
        for i in range(n):
            s = 0.8 * ((bm25_scores[i] - min_b) / range_b)
            if doc_has_cn[i]:
                s += 0.15 * coverages[i]
                s += 0.05 * ngrams[i]
            scored.append((results[i][0], s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]


_reranker = None


def get_reranker(top_k: int = 3):
    global _reranker
    if _reranker is None:
        try:
            _reranker = CrossEncoderReranker(top_k=top_k)
        except Exception as e:
            print(f"[reranker] Failed to load cross-encoder: {e}")
            _reranker = None
    return _reranker