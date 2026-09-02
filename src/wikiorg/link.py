"""Related-page links: cosine kNN candidates re-ranked with Maximal Marginal Relevance.

For each document the ``n_candidates`` (10) nearest neighbours by cosine are taken in decreasing
similarity, then MMR picks ``n_links`` (5) of them greedily:

    argmax_d  λ · sim(q, d) − (1 − λ) · max_{s ∈ selected} sim(d, s)

Candidates are scanned in kNN order and a strictly better score is required to replace the
running best, so λ = 1 reproduces the kNN order exactly.
"""
from __future__ import annotations

import numpy as np

N_CANDIDATES = 10
N_LINKS = 5
LAMBDAS = (1.0, 0.7, 0.5)
HEADLINE_LAMBDA = 0.7


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def knn(X: np.ndarray, n_candidates: int = N_CANDIDATES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(idx, sims, S)``: per row the ``n_candidates`` nearest other rows (decreasing cosine,
    ties by index), their similarities, and the full similarity matrix."""
    X = _unit(X)
    S = X @ X.T
    n = S.shape[0]
    m = min(n_candidates, n - 1)
    Sx = S.copy()
    np.fill_diagonal(Sx, -np.inf)
    order = np.argsort(-Sx, axis=1, kind="stable")[:, :m]
    sims = np.take_along_axis(Sx, order, axis=1)
    return order, sims, S


def mmr_select(cand: np.ndarray, cand_sims: np.ndarray, S: np.ndarray, lam: float, n_links: int = N_LINKS) -> list[int]:
    selected: list[int] = []
    remaining = list(range(len(cand)))
    while remaining and len(selected) < n_links:
        best, best_score = None, -np.inf
        for j in remaining:
            rel = float(cand_sims[j])
            div = max((float(S[cand[j], cand[s]]) for s in selected), default=0.0)
            score = lam * rel - (1.0 - lam) * div
            if score > best_score:
                best, best_score = j, score
        selected.append(best)
        remaining.remove(best)
    return [int(cand[j]) for j in selected]


def related_links(X: np.ndarray, n_candidates: int = N_CANDIDATES, n_links: int = N_LINKS,
                  lam: float = HEADLINE_LAMBDA) -> tuple[list[list[int]], list[list[float]]]:
    """Per document: chosen neighbour indices (MMR order) and their cosine similarities."""
    order, sims, S = knn(X, n_candidates)
    links, link_sims = [], []
    for i in range(X.shape[0]):
        chosen = mmr_select(order[i], sims[i], S, lam, n_links)
        links.append(chosen)
        link_sims.append([float(S[i, j]) for j in chosen])
    return links, link_sims
