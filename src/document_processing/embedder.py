import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
from pathlib import Path
from src.config import PROCESSED_DIR


class TfidfEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            max_features=50000,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts)

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Embedder not fitted yet")
        return self.vectorizer.transform([query])

    def save(self, path: Optional[Path] = None):
        path = path or PROCESSED_DIR / "tfidf_vectorizer.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Optional[Path] = None):
        path = path or PROCESSED_DIR / "tfidf_vectorizer.pkl"
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TfidfEmbedder()
    return _embedder
