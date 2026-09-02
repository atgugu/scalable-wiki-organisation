"""kNN candidates + MMR: lambda = 1 reproduces the kNN order; diversity kicks in for lambda < 1."""
import numpy as np

from wikiorg.link import knn, mmr_select, related_links


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_lambda_one_is_knn_order():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 16))
    order, sims, S = knn(X, 10)
    links, link_sims = related_links(X, n_candidates=10, n_links=5, lam=1.0)
    for i in range(60):
        assert links[i] == list(order[i][:5])
        assert i not in links[i] and len(set(links[i])) == 5
        assert all(a >= b for a, b in zip(link_sims[i], link_sims[i][1:]))


def test_knn_excludes_self_and_is_sorted():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 8))
    order, sims, S = knn(X, 10)
    assert order.shape == (20, 10)
    for i in range(20):
        assert i not in order[i]
        assert all(a >= b for a, b in zip(sims[i], sims[i][1:]))


def test_mmr_prefers_diverse_candidate():
    e = np.eye(4)
    q = e[0]
    c1 = _unit(e[0] + 0.10 * e[1])
    c2 = _unit(e[0] + 0.11 * e[1])   # near-duplicate of c1, second by relevance
    c3 = _unit(e[0] + 0.30 * e[2])   # less relevant, but diverse
    X = np.stack([q, c1, c2, c3])
    order, sims, S = knn(X, 3)
    assert list(order[0]) == [1, 2, 3]
    assert mmr_select(order[0], sims[0], S, lam=1.0, n_links=2) == [1, 2]
    assert mmr_select(order[0], sims[0], S, lam=0.5, n_links=2) == [1, 3]
    assert mmr_select(order[0], sims[0], S, lam=0.0, n_links=2) == [1, 3]


def test_links_within_candidates_for_all_lambdas():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 12))
    order, _, _ = knn(X, 10)
    for lam in (1.0, 0.7, 0.5):
        links, _ = related_links(X, lam=lam)
        for i in range(40):
            assert set(links[i]) <= set(order[i]) and len(links[i]) == 5
