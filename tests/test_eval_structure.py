"""Structure metrics hand cases."""
import networkx as nx
import numpy as np
import pytest

from wikiorg.eval.state import Category, Page, WikiState
from wikiorg.eval.structure import (KAMPS_KOOLEN_2009, gini, hierarchy_depth, kamps_koolen_row, normalised_entropy,
                                    reciprocity, structure_metrics)


def test_gini_and_entropy():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)
    assert gini([]) == 0.0
    assert normalised_entropy([5, 5, 5, 5]) == pytest.approx(1.0)
    assert normalised_entropy([10, 0, 0]) != normalised_entropy([10, 0, 0])  # NaN for a single non-empty bucket
    assert 0 < normalised_entropy([8, 1, 1]) < 1


def test_reciprocity_garlaschelli_loffredo():
    G = nx.DiGraph([("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")])
    r = reciprocity(G)
    assert r["reciprocity_raw"] == pytest.approx(1.0) and r["reciprocity_gl"] == pytest.approx(1.0)
    G2 = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    r2 = reciprocity(G2)
    a = 3 / (4 * 3)
    assert r2["reciprocity_raw"] == 0.0 and r2["reciprocity_gl"] == pytest.approx(-a / (1 - a))
    assert r2["density"] == pytest.approx(a)
    assert np.isnan(reciprocity(nx.DiGraph())["reciprocity_gl"])


def _state(parents=False):
    pages = {"a": Page(title="a", category="x", related=["b", "c"]), "b": Page(title="b", category="x", related=["a"]),
             "c": Page(title="c", category="y", related=["a"]), "d": Page(title="d", category="y", related=[]),
             "e": Page(title="e", category=None, related=[])}
    cats = {"x": Category(label="x", parent="root" if parents else None), "y": Category(label="y", parent="root" if parents else None)}
    if parents:
        cats["root"] = Category(label="root")
    return WikiState("A", pages, cats, extra_pages=[{"type": "entity", "words": 50}, {"type": "topic", "words": 20}])


def test_structure_metrics_hand_case():
    st = _state()
    m = structure_metrics(st)
    assert m["n_pages"] == 5 and m["n_categories"] == 2 and m["n_uncategorised"] == 1
    assert m["depth"] == 2 and hierarchy_depth(_state(parents=True)) == 3
    assert m["n_edges"] == 4
    assert m["orphan_rate"] == pytest.approx(2 / 5)  # d and e have no inbound link
    assert m["docs_per_category_gini"] == pytest.approx(0.0) and m["docs_per_category_entropy_norm"] == pytest.approx(1.0)
    assert m["reciprocity_raw"] == pytest.approx(1.0)  # a<->b, a<->c
    assert m["largest_wcc_frac"] == pytest.approx(3 / 5)
    assert 0 < m["pagerank_top5_mass"] <= 1 and 0 <= m["pagerank_gini"] <= 1
    assert m["n_extra_pages"] == 2 and m["extra_pages_by_type"] == {"entity": 1, "topic": 1} and m["extra_pages_words"] == 70
    assert m["median_in_degree"] == 1.0 and m["median_out_degree"] == 1.0


def test_kamps_koolen_reference_row():
    row = kamps_koolen_row()
    assert row["median_in_degree"] == 4 and row["median_out_degree"] == 12
    assert KAMPS_KOOLEN_2009["source"].startswith("Kamps")
    assert row["orphan_rate"] is None and "Kamps" in row["arm"]
