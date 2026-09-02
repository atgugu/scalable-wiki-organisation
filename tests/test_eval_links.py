"""Link metrics: P@k / R@k / AP hand cases, toy-graph hyperlink agreement with
oracle and random baselines, same-class precision with chance, diversity,
category mix and link validity."""
import numpy as np
import pytest

from wikiorg.eval.links import (average_precision_at_k, category_mix, hyperlink_metrics, intra_list_diversity,
                                link_validity, precision_at_k, recall_at_k, same_class_precision)
from wikiorg.eval.state import Category, Page, WikiState, gold_from_rows


def test_ranked_list_hand_cases():
    ranked, rel = ["a", "b", "c", "d", "e"], {"a", "c", "x"}
    assert precision_at_k(ranked, rel, 5) == pytest.approx(2 / 5)
    assert recall_at_k(ranked, rel, 5) == pytest.approx(2 / 3)
    assert average_precision_at_k(ranked, rel, 5) == pytest.approx((1 + 2 / 3) / 3)
    assert average_precision_at_k(["c", "a"], {"a", "c"}, 5) == pytest.approx(1.0)
    assert np.isnan(recall_at_k(ranked, set(), 5))
    # precision ignores order, AP does not
    assert precision_at_k(["b", "d", "e", "c", "a"], rel, 5) == pytest.approx(2 / 5)
    assert average_precision_at_k(["b", "d", "e", "c", "a"], rel, 5) < average_precision_at_k(ranked, rel, 5)


def _ring(n=12):
    rows = [{"id": f"d{i}", "title": f"T{i}", "class": f"g{i % 3}",
             "links": [f"d{(i + 1) % n}"]} for i in range(n)]  # directed ring; undirected union -> degree 2
    return gold_from_rows(rows)


def test_hyperlink_metrics_toy_graph():
    gold = _ring()
    n = 12
    assert all(len(gold.links[f"d{i}"]) == 2 for i in range(n))
    pages = {f"d{i}": Page(title=f"T{i}", category="c0", related=[f"d{(i + 1) % n}", f"d{(i + 5) % n}"]) for i in range(n)}
    st = WikiState("A", pages, {"c0": Category(label="all")})
    res = hyperlink_metrics(st, gold, k=5, n_random=50, seed=0)
    assert res["n_evaluated"] == n
    assert res["precision_at_k"] == pytest.approx(1 / 5)
    assert res["recall_at_k"] == pytest.approx(1 / 2)
    assert res["map"] == pytest.approx(0.5)  # one hit at rank 1 -> AP = 1/min(2,5)
    assert res["oracle_precision_at_k"] == pytest.approx(2 / 5)
    assert res["oracle_recall_at_k"] == pytest.approx(1.0)
    assert res["random_precision_closed_form"] == pytest.approx(2 / 11)
    assert abs(res["random_precision_at_k"] - 2 / 11) < 0.08
    assert res["median_gold_degree"] == 2
    perfect = WikiState("A", {f"d{i}": Page(title=f"T{i}", category="c0", related=sorted(gold.links[f"d{i}"])) for i in range(n)}, {})
    assert hyperlink_metrics(perfect, gold, k=5, n_random=0)["recall_at_k"] == pytest.approx(1.0)


def test_same_class_precision_and_chance():
    rows = [{"id": f"d{i}", "title": f"T{i}", "l2": ["a", "a", "a", "b"][i % 4]} for i in range(40)]
    gold = gold_from_rows(rows)
    cls = {f"d{i}": ["a", "a", "a", "b"][i % 4] for i in range(40)}
    pages = {}
    for i in range(40):
        same = [f"d{j}" for j in range(40) if j != i and cls[f"d{j}"] == cls[f"d{i}"]][:3]
        other = [f"d{j}" for j in range(40) if cls[f"d{j}"] != cls[f"d{i}"]][:2]
        pages[f"d{i}"] = Page(title=f"T{i}", category="c", related=same + other)
    pages["d0"] = Page(title="T0", category="c", related=[])  # no links -> 0
    st = WikiState("A", pages, {"c": Category(label="c")})
    res = same_class_precision(st, gold, level="l2", k=5)
    assert res["n_without_links"] == 1
    assert res["precision_at_k"] == pytest.approx((39 * 0.6) / 40)
    assert res["precision_at_k_linked_only"] == pytest.approx(0.6)
    assert res["chance"] == pytest.approx(0.75 ** 2 + 0.25 ** 2)
    assert res["chance_finite_population"] == pytest.approx(0.75 * 29 / 39 + 0.25 * 9 / 39)


def test_diversity_and_category_mix():
    e = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]), "c": np.array([1.0, 0.0])}
    st = WikiState("A", {"a": Page(title="a", category="x", related=["b", "c"]), "b": Page(title="b", category="y", related=["a", "c"]),
                         "c": Page(title="c", category="x", related=["a"])}, {"x": Category(label="x"), "y": Category(label="y")})
    ild = intra_list_diversity(st, e)
    assert ild["per_document"]["a"] == pytest.approx(1.0)  # b orthogonal to c -> 1 - 0
    assert ild["per_document"]["b"] == pytest.approx(0.0)  # a and c identical
    assert ild["n_docs"] == 2 and ild["ild"] == pytest.approx(0.5)
    mix = category_mix(st)
    assert mix["same_category_frac"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert mix["unique_category_frac"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)


def test_link_validity_classes():
    titles = {"d1": "Danube", "d2": "Rhine", "d3": "Angela Merkel"}
    st = WikiState("C", {
        "d1": Page(title="Danube", category="c", related=["d2"], related_raw=["Rhine", "Rhine", "Danube", "Danub", "Nile", "d3"]),
        "d2": Page(title="Rhine", category="c", related=[], related_raw=[]),
        "d3": Page(title="Angela Merkel", category=None, related=[], related_raw=[], extra={"status": "missing"}),
    }, {"c": Category(label="c")})
    v = link_validity(st, corpus_titles=titles)
    assert (v["resolved"], v["near_miss"], v["dangling"], v["self"], v["duplicate"]) == (2, 1, 1, 1, 1)
    assert v["resolved_frac"] == pytest.approx(0.5) and v["n_docs_without_links"] == 2
    # default targets exclude missing pages: "d3" now dangles
    v2 = link_validity(st)
    assert v2["resolved"] == 1 and v2["dangling"] == 2
