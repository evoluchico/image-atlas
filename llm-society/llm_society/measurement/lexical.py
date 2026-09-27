"""Lexical convergence and vocabulary diversity.

Raw message text is never modified; tokens are derived from it with a fixed
pipeline (Unicode NFKC normalisation, lower-casing, regex tokenisation,
optional lemmatisation, optional stopword removal).
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | {"s", "t", "don", "doesn", "isn", "it's", "i'm"}


class Preprocessor:
    def __init__(self, remove_stopwords: bool = True, lemmatise: bool = False):
        self.remove_stopwords = remove_stopwords
        self.lemmatise = False
        self._nlp = None
        if lemmatise:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                self.lemmatise = True
            except Exception:  # pragma: no cover - optional dependency
                warnings.warn("lemmatisation requested but spaCy/en_core_web_sm unavailable; skipping")

    def describe(self) -> dict:
        return {
            "normalisation": "NFKC",
            "lowercase": True,
            "tokeniser": TOKEN_RE.pattern,
            "lemmatised": self.lemmatise,
            "stopwords_removed": self.remove_stopwords,
        }

    def all_tokens(self, text: str) -> list[str]:
        norm = unicodedata.normalize("NFKC", text or "").lower().replace("’", "'")
        if self._nlp is not None:
            return [t.lemma_ for t in self._nlp(norm) if TOKEN_RE.fullmatch(t.text)]
        return TOKEN_RE.findall(norm)

    def content_tokens(self, text: str) -> list[str]:
        toks = self.all_tokens(text)
        if self.remove_stopwords:
            toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
        return toks


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mean_pairwise_jsd(P: np.ndarray) -> float:
    """Mean Jensen-Shannon divergence (base 2, in [0, 1]) over all row pairs.
    Rows of P must be probability distributions."""
    n = P.shape[0]
    if n < 2:
        return float("nan")
    total, count = 0.0, 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log2(P), 0.0)
        for i in range(n - 1):
            Q = P[i + 1 :]
            M = 0.5 * (P[i] + Q)
            logM = np.where(M > 0, np.log2(M), 0.0)
            kl_p = np.where(P[i] > 0, P[i] * (logP[i] - logM), 0.0).sum(axis=1)
            kl_q = np.where(Q > 0, Q * (logP[i + 1 :] - logM), 0.0).sum(axis=1)
            total += float((0.5 * (kl_p + kl_q)).sum())
            count += Q.shape[0]
    return total / count


def mean_pairwise_cosine_similarity(X: np.ndarray) -> float:
    n = X.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.divide(X, norms, out=np.zeros_like(X, dtype=float), where=norms > 0)
    S = Xn @ Xn.T
    iu = np.triu_indices(n, 1)
    return float(S[iu].mean())


def top_k(counter: Counter, k: int) -> set[str]:
    return {w for w, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}


def mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    n = len(sets)
    if n < 2:
        return float("nan")
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            u = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / len(u) if u else 0.0)
    return float(np.mean(vals))


def mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio (length-robust alternative to TTR)."""
    if not tokens:
        return float("nan")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    ratios = [len(counts) / window]
    for i in range(window, len(tokens)):
        out = tokens[i - window]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[tokens[i]] += 1
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


def lexical_round_metrics(
    agent_tokens: dict[str, list[str]], top_ks: list[int]
) -> dict[str, float]:
    """Metrics for one analysis window: agent -> content tokens in the window."""
    agents = [a for a, toks in agent_tokens.items() if toks]
    out: dict[str, float] = {}
    if len(agents) < 2:
        return out
    counters = [Counter(agent_tokens[a]) for a in agents]
    vocab = sorted(set().union(*counters))
    index = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(agents), len(vocab)))
    for r, c in enumerate(counters):
        for w, n in c.items():
            C[r, index[w]] = n
    P = C / C.sum(axis=1, keepdims=True)
    out["lexical_jsd"] = mean_pairwise_jsd(P)
    for k in top_ks:
        out[f"jaccard_top{k}"] = mean_pairwise_jaccard([top_k(c, k) for c in counters])
    pooled = C.sum(axis=0)
    total = pooled.sum()
    out["vocab_size"] = float(len(vocab))
    out["type_token_ratio"] = float(len(vocab) / total)
    out["mattr"] = float(np.nanmean([mattr(agent_tokens[a]) for a in agents]))
    h = _entropy_bits(pooled / total)
    out["lexical_entropy"] = h
    out["effective_vocab_size"] = float(2**h)
    out["shared_vocab_share"] = float(((C > 0).sum(axis=0) >= 2).mean())
    return out
