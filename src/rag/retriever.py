import os
from typing import List, Tuple, Optional
from langchain.schema import Document
from src.vector_store.store import get_store
from src.rag.reranker import get_reranker
from src.config import TOP_K, RERANK_TOP_K, RERANK_ENABLED, HF_ENDPOINT

os.environ["HF_ENDPOINT"] = HF_ENDPOINT


class MaritimeLawRetriever:
    def __init__(
        self,
        top_k: int = TOP_K,
        hybrid: bool = True,
        rerank: bool = RERANK_ENABLED,
        rerank_top_k: int = RERANK_TOP_K,
    ):
        self.store = get_store()
        self.top_k = top_k
        self.hybrid = hybrid
        self.rerank = rerank
        self.rerank_top_k = rerank_top_k
        self._reranker = get_reranker(top_k=rerank_top_k) if rerank else None

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        if self.hybrid:
            results = self.store.hybrid_search(query, k=self.top_k)
        else:
            results = self.store._bm25_search(query, k=self.top_k)
        if self._reranker is not None:
            results = self._reranker.rerank(query, results)
        return results

    def retrieve_texts(self, query: str) -> List[str]:
        results = self.retrieve(query)
        return [doc.page_content for doc, _ in results]
