import numpy as np

from wikiorg.eval.stability import pairwise_agreement, stability_table
from wikiorg.eval.state import Category, Page, WikiState


def _state(assign, arm="A"):
    return WikiState(arm, {i: Page(title=i, category=c) for i, c in assign.items()}, {c: Category(label=c) for c in set(assign.values()) if c})


def test_pairwise_agreement():
    ids = [f"d{i}" for i in range(60)]
    base = {i: f"c{n % 3}" for n, i in enumerate(ids)}
    relabel = {i: f"x{n % 3}" for n, i in enumerate(ids)}  # same partition, other ids
    rng = np.random.default_rng(0)
    rand = {i: f"c{rng.integers(3)}" for i in ids}
    res = pairwise_agreement({"s1": _state(base), "s2": _state(relabel), "s3": _state(rand)})
    assert res["n_runs"] == 3 and res["n_pairs"] == 3 and res["n_docs"] == 60
    p12 = next(p for p in res["pairs"] if {p["a"], p["b"]} == {"s1", "s2"})
    assert p12["ari"] == 1.0 and p12["nmi"] == 1.0
    assert res["ari_min"] < 0.3 and res["ari_max"] == 1.0 and res["ari_sd"] > 0
    assert res["k_per_run"]["s1"] == 3
    # dict partitions and a restricted id set
    res2 = pairwise_agreement([base, relabel], ids=ids[:30])
    assert res2["n_docs"] == 30 and res2["nmi_mean"] == 1.0
    rows = stability_table({"A/dbpedia/300": {"s1": _state(base), "s2": _state(rand)}})
    assert rows[0]["group"] == "A/dbpedia/300" and rows[0]["n_pairs"] == 1 and "s1=3" in rows[0]["k_per_run"]
