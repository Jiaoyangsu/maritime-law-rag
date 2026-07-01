import os
from typing import List, Tuple
from langchain.schema import Document
from src.config import CROSS_ENCODER_MODEL, HF_ENDPOINT


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, top_k: int = 3):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
        self._load_model()

    def _load_model(self):
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, local_files_only=True)

    def rerank(
        self, query: str, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        if not results or self._model is None:
            return results
        pairs = [(query, doc.page_content) for doc, _ in results]
        scores = self._model.predict(pairs)
        scored = []
        for (doc, _), score in zip(results, scores):
            scored.append((doc, float(score)))
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
