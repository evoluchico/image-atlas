"""Message embeddings for semantic convergence.

`sentence_transformers` is used when installed (fixed model from config).
Otherwise a deterministic TF-IDF + truncated SVD (LSA) embedding is fitted on
the pooled corpus. The backend used is always recorded in metric metadata,
because the two are not interchangeable.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, backend: str = "auto", model_name: str = "all-MiniLM-L6-v2", seed: int = 0):
        self.requested = backend
        self.model_name = model_name
        self.seed = seed
        self.backend = backend
        if backend in ("auto", "sentence_transformers"):
            try:
                import sentence_transformers  # noqa: F401

                self.backend = "sentence_transformers"
            except ImportError:
                if backend == "sentence_transformers":
                    raise
                self.backend = "tfidf_svd"

    def describe(self) -> dict:
        d = {"backend": self.backend}
        if self.backend == "sentence_transformers":
            d["model"] = self.model_name
        return d

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), float)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        X = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True).fit_transform(texts)
        k = int(min(100, X.shape[0] - 1, X.shape[1] - 1))
        if k < 2:
            E = X.toarray()
        else:
            E = TruncatedSVD(k, random_state=self.seed).fit_transform(X)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        return np.divide(E, norms, out=np.zeros_like(E), where=norms > 0)


def mean_pairwise_cosine_distance(centroids: np.ndarray) -> float:
    n = centroids.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    X = np.divide(centroids, norms, out=np.zeros_like(centroids), where=norms > 0)
    S = X @ X.T
    iu = np.triu_indices(n, 1)
    return float((1.0 - S[iu]).mean())
