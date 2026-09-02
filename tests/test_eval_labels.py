"""Label metrics with an injected embedder (no model download in tests)."""
import numpy as np
import pytest

from wikiorg.eval.labels import (HashingEmbedder, cluster_label_table, de_camel, evaluate_labels, label_similarity,
                                 npmi_coherence, token_jaccard, tokenize)
from wikiorg.eval.state import Category, Page, WikiState, gold_from_rows


def test_de_camel_and_tokens():
    assert de_camel("EducationalInstitution") == "educational institution"
    assert de_camel("USPresident") == "us president"
    assert de_camel("MeanOfTransportation") == "mean of transportation"
    assert de_camel("Non-ProfitOrganisation") == "non profit organisation"
    assert de_camel("Plant") == "plant" and de_camel("") == ""
    assert tokenize("New York's 2nd bridge") == ["new", "york", "2nd", "bridge"]
    assert token_jaccard("sports team", "team sports club") == pytest.approx(2 / 3)
    assert token_jaccard("", "") == 0.0


def _gold_and_state(uncategorised=0):
    rows = [{"id": f"d{i}", "title": f"T{i}", "l1": "x", "l2": ["Athlete", "Politician", "WrittenWork"][i % 3]} for i in range(30)]
    gold = gold_from_rows(rows)
    pages = {f"d{i}": Page(title=f"T{i}", category=(f"c{i % 3}" if i >= uncategorised else None)) for i in range(30)}
    cats = {"c0": Category(label="athlete", terms=["athlete", "team"]), "c1": Category(label="politician", terms=["politician"]),
            "c2": Category(label="written work", terms=["book", "novel"])}
    return gold, WikiState("A", pages, cats)


def test_label_similarity_perfect_labels():
    gold, st = _gold_and_state()
    res = label_similarity(st, gold, level="l2", embed_fn=HashingEmbedder(), seed=0)
    assert res["n_clusters"] == 3 and res["n_classes"] == 3
    assert res["weighted_mean"] == pytest.approx(1.0) and res["unweighted_mean"] == pytest.approx(1.0)
    assert res["jaccard_weighted"] == pytest.approx(1.0)
    assert res["floor_random_unweighted"] < 0.5 and res["floor_all_other_unweighted"] < 0.5
    assert all(res["per_document"][f"d{i}"] == pytest.approx(1.0) for i in range(30))
    table = cluster_label_table(st, gold, "l2")
    assert {r.majority_class for r in table} == {"Athlete", "Politician", "WrittenWork"}
    assert all(r.gold_name == de_camel(r.majority_class) for r in table)


def test_uncategorised_documents_score_zero():
    gold, st = _gold_and_state(uncategorised=6)
    res = label_similarity(st, gold, level="l2", embed_fn=HashingEmbedder())
    assert res["n_uncategorised"] == 6
    assert res["weighted_mean"] == pytest.approx(24 / 30)
    assert res["weighted_mean_categorised"] == pytest.approx(1.0)
    assert res["per_document"]["d0"] == 0.0


def test_hashing_embedder_is_unit_norm_and_deterministic():
    e = HashingEmbedder()
    v = e(["alpha beta", "alpha beta", "gamma"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[0], v[1]) and abs(v[0] @ v[2]) < 1e-9


def test_npmi_hand_case():
    texts = ["alpha beta"] * 5 + ["gamma delta"] * 5  # each document is one window
    res = npmi_coherence({"c": ["alpha", "beta", "gamma"]}, texts, window=10)
    assert res["n_windows"] == 10
    assert res["per_category"]["c"] == pytest.approx((1.0 - 1.0 - 1.0) / 3)
    full = npmi_coherence({"c": ["alpha", "beta"]}, texts, window=10)
    assert full["per_category"]["c"] == pytest.approx(1.0)
    # a multi-word term is present when all its tokens are in the window
    two = npmi_coherence({"c": ["alpha beta", "gamma"]}, texts, window=10)
    assert two["per_category"]["c"] == pytest.approx(-1.0)
    # sliding windows: 'x' and 'y' 20 tokens apart never share a 10-token window
    far = npmi_coherence({"c": ["xx", "yy"]}, ["xx " + "zz " * 20 + "yy"], window=10)
    assert far["per_category"]["c"] == -1.0
    assert np.isnan(npmi_coherence({}, texts)["mean"])


def test_evaluate_labels_block():
    gold, st = _gold_and_state()
    texts = {f"d{i}": "athlete team athlete team politician book novel" for i in range(30)}
    res = evaluate_labels(st, gold, level="l2", texts=texts, embed_fn=HashingEmbedder())
    assert res["npmi"] is not None and res["npmi"]["n_categories"] == 3
    assert -1.0 <= res["npmi"]["mean"] <= 1.0
    st_no_terms = WikiState("B", st.pages, {c: Category(label=v.label) for c, v in st.categories.items()})
    assert evaluate_labels(st_no_terms, gold, level="l2", texts=texts, embed_fn=HashingEmbedder())["npmi"] is None
