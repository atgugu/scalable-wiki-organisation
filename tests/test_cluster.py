"""Spherical k-means vs sklearn KMeans on separated blobs; determinism; k selection; two-level."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from wikiorg.cluster import apportion_k, choose_k_silhouette, k_grid, normalize_rows, spherical_kmeans, two_level


def _blobs(k=5, n=400, d=32, seed=0):
    X, y = make_blobs(n_samples=n, centers=k, n_features=d, cluster_std=0.8, random_state=seed)
    return normalize_rows(X), y


def test_matches_sklearn_kmeans_on_blobs():
    X, y = _blobs()
    labels, C, info = spherical_kmeans(X, 5, seed=1)
    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(y, labels) > 0.9
    assert adjusted_rand_score(km.labels_, labels) > 0.9
    assert C.shape == (5, X.shape[1])
    np.testing.assert_allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-9)
    assert 0.9 < info["objective"] <= 1.0


def test_deterministic_and_best_of_n_init():
    X, _ = _blobs(seed=3)
    a, _, ia = spherical_kmeans(X, 5, seed=7)
    b, _, ib = spherical_kmeans(X, 5, seed=7)
    assert np.array_equal(a, b) and ia["objective"] == ib["objective"]
    single = spherical_kmeans(X, 5, seed=7, n_init=1)[2]["objective"]
    assert ia["objective"] >= single - 1e-12


def test_labels_canonical_by_size():
    X, _ = _blobs(k=3, n=90, seed=5)
    labels, _, _ = spherical_kmeans(X, 3, seed=0)
    sizes = np.bincount(labels)
    assert list(sizes) == sorted(sizes, reverse=True)
    assert set(labels) == {0, 1, 2}


def test_edge_cases():
    X, _ = _blobs(k=2, n=10, seed=1)
    lab1, C1, _ = spherical_kmeans(X, 1)
    assert set(lab1) == {0} and C1.shape == (1, X.shape[1])
    labn, _, _ = spherical_kmeans(X, 10)
    assert sorted(labn) == list(range(10))


def test_k_grid_and_silhouette_choice():
    grid = k_grid(18, 100)
    assert {5, 10, 18, 20} <= set(grid) and all(2 <= k <= 99 for k in grid) and grid == sorted(set(grid))
    X, _ = _blobs(k=5, n=300, seed=11)
    k, curve = choose_k_silhouette(X, 5, seed=1, n_init=3)
    assert k == 5
    assert {c["k"] for c in curve} == set(k_grid(5, 300))


def test_two_level_consistency():
    X, _ = _blobs(k=6, n=240, seed=2)
    leaf, top, parent, info = two_level(X, 3, 6, seed=1, n_init=3)
    assert len(parent) == info["k_total"] == leaf.max() + 1
    assert set(top) == {0, 1, 2}
    for i in range(len(leaf)):
        assert parent[leaf[i]] == top[i]
    assert list(apportion_k(np.array([100, 50, 10]), 6)) == [3, 2, 1]
