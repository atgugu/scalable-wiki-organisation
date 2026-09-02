"""Partition metrics: sklearn parity on random partitions, purity family on a
hand-built contingency table, singleton handling and the random baseline."""
import numpy as np
import pytest
from sklearn import metrics as skm

from wikiorg.eval.categories import (contingency, evaluate_categories, harmonic_purity, inverse_purity,
                                     partition_metrics, per_document_metric_values, purity, random_partition_baseline)
from wikiorg.eval.state import Category, Page, WikiState, gold_from_rows


@pytest.mark.parametrize("seed", range(50))
def test_wrappers_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 200))
    kp, kg = int(rng.integers(1, 12)), int(rng.integers(1, 12))
    pred = [f"p{i}" for i in rng.integers(0, kp, size=n)]
    gold = [f"g{i}" for i in rng.integers(0, kg, size=n)]
    m = partition_metrics(pred, gold)
    assert m["nmi"] == pytest.approx(skm.normalized_mutual_info_score(gold, pred))
    assert m["ari"] == pytest.approx(skm.adjusted_rand_score(gold, pred))
    assert m["ami"] == pytest.approx(skm.adjusted_mutual_info_score(gold, pred))
    h, c, v = skm.homogeneity_completeness_v_measure(gold, pred)
    assert (m["homogeneity"], m["completeness"], m["v_measure"]) == pytest.approx((h, c, v))
    assert m["v_measure"] == pytest.approx(m["nmi"])  # arithmetic NMI == V-measure
    assert 0 <= m["purity"] <= 1 and 0 <= m["inverse_purity"] <= 1 and 0 <= m["p1"] <= 1


def _labels_from_table(table):
    pred, gold = [], []
    for i, row in enumerate(table):
        for j, n in enumerate(row):
            pred += [f"c{i}"] * n
            gold += [f"g{j}"] * n
    return pred, gold


def test_purity_family_hand_case():
    table = [[4, 3, 0], [2, 2, 0], [0, 0, 6]]  # rows = clusters, cols = classes, N = 17
    pred, gold = _labels_from_table(table)
    C = contingency(pred, gold)
    assert C.sum() == 17 and sorted(C.max(axis=1).tolist()) == [2, 4, 6]
    assert purity(pred, gold) == pytest.approx(12 / 17)
    assert inverse_purity(pred, gold) == pytest.approx(13 / 17)
    p, ip = 12 / 17, 13 / 17
    assert harmonic_purity(pred, gold) == pytest.approx(2 * p * ip / (p + ip))
    pdv = per_document_metric_values(pred, gold)
    assert pdv["purity"].mean() == pytest.approx(12 / 17)
    assert pdv["inverse_purity"].mean() == pytest.approx(13 / 17)


def test_perfect_and_degenerate_partitions():
    gold = ["a"] * 5 + ["b"] * 5
    assert partition_metrics(gold, gold)["nmi"] == pytest.approx(1.0)
    assert partition_metrics(gold, gold)["p1"] == pytest.approx(1.0)
    one = partition_metrics(["x"] * 10, gold)
    assert one["nmi"] == pytest.approx(0.0) and one["purity"] == pytest.approx(0.5) and one["inverse_purity"] == 1.0
    assert partition_metrics([], [])["n"] == 0


def test_singletons_and_random_baseline():
    rows = [{"id": f"d{i}", "title": f"D{i}", "l1": "x", "l2": f"g{i % 4}"} for i in range(200)]
    gold = gold_from_rows(rows)
    pages = {f"d{i}": Page(title=f"D{i}", category=(f"c{i % 4}" if i % 25 else None)) for i in range(200)}
    st = WikiState("B", pages, {f"c{j}": Category(label=f"c{j}") for j in range(4)})
    res = evaluate_categories(st, gold, n_random=100, seed=1)
    l2 = res["levels"]["l2"]
    assert res["k_produced"] == 4 and res["n_uncategorised"] == 8 and l2["n_singletons"] == 8
    assert l2["k_pred"] == 12  # 4 clusters + 8 singletons
    assert 0.8 < l2["nmi"] < 1.0 and l2["purity"] == 1.0
    rb = l2["random_baseline"]
    assert rb["n_draws"] == 100 and rb["mode"] == "permute"
    assert abs(rb["ari"]) < 0.05 and rb["nmi"] < l2["nmi"] and rb["p1"] < l2["p1"]
    assert "l1" in res["levels"]
    # permutation baseline preserves k
    pred = [f"c{i % 4}" for i in range(50)]
    base = random_partition_baseline(pred, [f"g{i % 5}" for i in range(50)], n_draws=5, seed=0, mode="uniform")
    assert base["mode"] == "uniform" and base["n_draws"] == 5
