"""Spherical k-means (cosine), silhouette-based k selection, two-level clustering.

``spherical_kmeans``: rows of ``X`` are L2-normalised; centroids are initialised with
``sklearn.cluster.kmeans_plusplus`` (on unit vectors the Euclidean k-means++ seeding is the
cosine seeding, since ||x-c||² = 2 - 2·cos), points are assigned by argmax dot product, centroids
are the renormalised sums of their members, empty clusters are re-seeded with the worst-fit point.
``n_init`` restarts (seeds derived from ``seed`` and the restart index); the restart with the best
objective (mean cosine of each point to its centroid) wins.  Cluster ids are then renumbered by
decreasing size (ties: smallest member index) so the labelling is canonical.  Deterministic for a
given seed and input (float64 arithmetic).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import silhouette_score

K_GRID_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
K_GRID_EXTRA = (5, 10, 20)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def _restart_seed(seed: int, init: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(init)]).generate_state(1)[0])


def _relabel_by_size(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.bincount(labels, minlength=k)
    first = np.full(k, labels.shape[0], dtype=np.int64)
    for i, lab in enumerate(labels):
        if first[lab] == labels.shape[0]:
            first[lab] = i
    order = sorted(range(k), key=lambda c: (-sizes[c], first[c]))
    mapping = np.empty(k, dtype=np.int64)
    for new, old in enumerate(order):
        mapping[old] = new
    return mapping[labels], np.asarray(order)


def _single_run(X: np.ndarray, k: int, rs: int, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    n = X.shape[0]
    C, _ = kmeans_plusplus(X, k, random_state=rs)
    C = normalize_rows(C)
    labels = (X @ C.T).argmax(axis=1)
    prev_obj = -np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        M = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
        C = np.asarray(M @ X)
        counts = np.bincount(labels, minlength=k)
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            S = X @ normalize_rows(C).T
            fit = S[np.arange(n), labels]
            worst = np.argsort(fit, kind="stable")
            taken = 0
            for e in empty:
                while counts[labels[worst[taken]]] <= 1:
                    taken += 1
                p = worst[taken]
                taken += 1
                counts[labels[p]] -= 1
                C[labels[p]] -= X[p]
                labels[p] = e
                C[e] = X[p]
                counts[e] = 1
        C = normalize_rows(C)
        S = X @ C.T
        new_labels = S.argmax(axis=1)
        obj = float(S[np.arange(n), new_labels].mean())
        converged = bool(np.array_equal(new_labels, labels)) or (obj - prev_obj) < tol
        labels = new_labels
        prev_obj = obj
        if converged:
            break
    S = X @ C.T
    labels = S.argmax(axis=1)
    obj = float(S[np.arange(n), labels].mean())
    return labels, C, obj, n_iter


def spherical_kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 10, max_iter: int = 100,
                     tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns ``(labels, centroids, info)``; ``info`` has objective, n_iter, best_init."""
    X = normalize_rows(X)
    n = X.shape[0]
    if k < 1 or k > n:
        raise ValueError(f"k={k} must be in [1, n={n}]")
    if k == 1:
        C = normalize_rows(X.sum(axis=0, keepdims=True))
        return np.zeros(n, dtype=np.int64), C, {"objective": float((X @ C.T).mean()), "n_iter": 0, "best_init": 0, "k": 1}
    if k == n:
        return np.arange(n), X.copy(), {"objective": 1.0, "n_iter": 0, "best_init": 0, "k": k}
    best = None
    for init in range(n_init):
        labels, C, obj, n_iter = _single_run(X, k, _restart_seed(seed, init), max_iter, tol)
        if best is None or obj > best[2] + 1e-12:
            best = (labels, C, obj, n_iter, init)
    labels, C, obj, n_iter, init = best
    labels, order = _relabel_by_size(labels, k)
    C = C[order]
    return labels.astype(np.int64), C, {"objective": obj, "n_iter": int(n_iter), "best_init": int(init), "k": int(k),
                                        "n_init": int(n_init), "seed": int(seed)}


def k_grid(k_star: int, n: int, fractions=K_GRID_FRACTIONS, extra=K_GRID_EXTRA) -> list[int]:
    ks = {int(round(k_star * f)) for f in fractions} | set(extra)
    return sorted(k for k in ks if 2 <= k <= n - 1)


def choose_k_silhouette(X: np.ndarray, k_star: int, seed: int = 0, n_init: int = 10,
                        grid: list[int] | None = None, max_iter: int = 100) -> tuple[int, list[dict]]:
    """Best k on the grid ``round(k*·{.25,.5,.75,1,1.25,1.5,2}) ∪ {5,10,20}`` by cosine silhouette
    (ties -> smaller k).  Returns ``(k, curve)``."""
    X = normalize_rows(X)
    n = X.shape[0]
    grid = grid or k_grid(k_star, n)
    D = np.clip(1.0 - X @ X.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    curve = []
    best_k, best_s = None, -np.inf
    for k in grid:
        labels, _, info = spherical_kmeans(X, k, seed=seed, n_init=n_init, max_iter=max_iter)
        if len(np.unique(labels)) < 2:
            continue
        s = float(silhouette_score(D, labels, metric="precomputed"))
        curve.append({"k": int(k), "silhouette": round(s, 6), "objective": round(info["objective"], 6)})
        if s > best_s + 1e-12:
            best_k, best_s = k, s
    if best_k is None:
        raise ValueError("silhouette could not be computed on any k of the grid")
    return int(best_k), curve


def apportion_k(sizes: np.ndarray, k_total: int) -> np.ndarray:
    """Split ``k_total`` leaves over top clusters proportionally to size (>= 1 each, <= size)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    m = sizes.shape[0]
    if k_total < m:
        raise ValueError(f"k_total={k_total} < number of top clusters {m}")
    k_total = min(k_total, int(sizes.sum()))
    alloc = np.ones(m, dtype=np.int64)
    exact = k_total * sizes / sizes.sum()
    while alloc.sum() < k_total:
        room = alloc < sizes
        if not room.any():
            break
        gap = np.where(room, exact - alloc, -np.inf)
        alloc[int(np.argmax(gap))] += 1
    return alloc


def two_level(X: np.ndarray, k1: int, k_total: int, seed: int = 0, n_init: int = 10,
              max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Top level with ``k1`` clusters, then spherical k-means inside each top cluster with a
    proportional share of ``k_total`` leaves.  Returns ``(leaf_labels, top_labels, parent_of_leaf, info)``."""
    X = normalize_rows(X)
    top, _, top_info = spherical_kmeans(X, k1, seed=seed, n_init=n_init, max_iter=max_iter)
    sizes = np.bincount(top, minlength=k1)
    alloc = apportion_k(sizes, k_total)
    leaf = np.empty(X.shape[0], dtype=np.int64)
    parent: list[int] = []
    offset = 0
    sub_info = []
    for c in range(k1):
        members = np.flatnonzero(top == c)
        kc = int(min(alloc[c], members.size))
        if kc <= 1 or members.size <= 1:
            leaf[members] = offset
            parent.append(c)
            offset += 1
            sub_info.append({"top": c, "k": 1})
            continue
        sub, _, info = spherical_kmeans(X[members], kc, seed=seed, n_init=n_init, max_iter=max_iter)
        leaf[members] = offset + sub
        parent.extend([c] * kc)
        offset += kc
        sub_info.append({"top": c, "k": kc, "objective": round(info["objective"], 6)})
    return leaf, top, np.asarray(parent), {"k1": int(k1), "k_total": int(offset), "top": top_info, "sub": sub_info}
