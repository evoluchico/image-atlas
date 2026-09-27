"""Shared topic representation for topical convergence.

Topics are fitted ONCE on the pooled corpus of all runs of an experiment, so
topic k means the same thing in every condition; each message is then
assigned a topic distribution. The implementation is modular: any object
with `fit_transform(texts) -> (doc_topic, topic_terms)` can replace it (e.g.
a BERTopic wrapper).
"""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np


class TopicModel(Protocol):
    def fit_transform(self, texts: list[str]) -> tuple[np.ndarray, list[list[str]]]: ...

    def describe(self) -> dict: ...


class SklearnTopicModel:
    def __init__(self, method: str = "nmf", n_topics: int = 12, seed: int = 0, n_terms: int = 10):
        self.method = method
        self.n_topics = n_topics
        self.seed = seed
        self.n_terms = n_terms
        self.fitted_topics: int | None = None

    def describe(self) -> dict:
        return {"method": self.method, "n_topics_requested": self.n_topics, "n_topics": self.fitted_topics}

    def fit_transform(self, texts: list[str]) -> tuple[np.ndarray, list[list[str]]]:
        from sklearn.decomposition import NMF, LatentDirichletAllocation
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        k = int(max(2, min(self.n_topics, len(texts) // 5 or 2)))
        if self.method == "lda":
            vec = CountVectorizer(stop_words="english", min_df=2, max_df=0.95)
            X = vec.fit_transform(texts)
            model = LatentDirichletAllocation(k, random_state=self.seed, learning_method="batch")
        else:
            vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.95, sublinear_tf=True)
            X = vec.fit_transform(texts)
            model = NMF(k, random_state=self.seed, init="nndsvda", max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            W = model.fit_transform(X)
        self.fitted_topics = k
        terms = np.array(vec.get_feature_names_out())
        topic_terms = [list(terms[np.argsort(-row)[: self.n_terms]]) for row in model.components_]
        sums = W.sum(axis=1, keepdims=True)
        # Messages with no in-vocabulary terms get a uniform distribution.
        W = np.where(sums > 0, W / np.where(sums > 0, sums, 1), 1.0 / k)
        return W, topic_terms


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
