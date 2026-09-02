"""End-to-end synthetic check: two fake WikiStates with known agreement through
the full metric suite, the pre-registered decision, the ledger, the tables and
the figures.  Numbers must come out where the construction puts them."""
import numpy as np
import pandas as pd
import pytest

from wikiorg.eval import figures, report
from wikiorg.eval.bootstrap import decide
from wikiorg.eval.cost import arm_a_cost, arm_b_cost, cost_ratio, load_prices
from wikiorg.eval.labels import HashingEmbedder
from wikiorg.eval.stability import stability_table
from wikiorg.eval.state import Category, Page, WikiState, gold_from_rows
from wikiorg.eval.suite import compare_arms, evaluate_run, floors_from_metrics, strip_bulky

CLASSES = ["Athlete", "Politician", "River", "Company", "Plant", "Album"]
N = 240


def make_corpus(seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"id": f"d{i}", "title": f"Doc {i}", "l1": ["Agent", "Agent", "Place", "Organisation", "Species", "Work"][i % 6], "l2": CLASSES[i % 6]} for i in range(N)]
    vocab = {c: [c.lower(), c.lower() + "s", f"{c.lower()} article"] for c in CLASSES}
    texts = {f"d{i}": " ".join(rng.choice(vocab[CLASSES[i % 6]] + ["the", "of", "and"], size=120)) for i in range(N)}
    return gold_from_rows(rows), texts


def make_state(arm, noise, rng, label_kind="gold", link_noise=0.2, missing=0):
    assign = {f"d{i}": (i % 6 if rng.random() > noise else int(rng.integers(6))) for i in range(N)}
    pages = {}
    for i in range(N):
        pid = f"d{i}"
        if i < missing:
            pages[pid] = Page(title=f"Doc {i}", category=None, extra={"status": "missing"})
            continue
        same = [f"d{j}" for j in range(N) if j != i and assign[f"d{j}"] == assign[pid]]
        other = [f"d{j}" for j in range(N) if assign[f"d{j}"] != assign[pid]]
        n_other = int(rng.random() < link_noise)
        links = [str(x) for x in rng.choice(same, size=5 - n_other, replace=False)] + [str(x) for x in rng.choice(other, size=n_other, replace=False)]
        pages[pid] = Page(title=f"Doc {i}", category=f"c{assign[pid]}", related=links, related_raw=[f"Doc {l[1:]}" for l in links])
    cats = {}
    for c in range(6):
        name = CLASSES[c].lower()
        if label_kind == "gold":
            cats[f"c{c}"] = Category(label=name, terms=[name, name + "s", "article"])
        else:
            cats[f"c{c}"] = Category(label=f"{name} pages", terms=[])
    return WikiState(arm, pages, cats)


@pytest.fixture(scope="module")
def world():
    gold, texts = make_corpus()
    rng = np.random.default_rng(1)
    A = make_state("A", 0.05, rng, "gold", link_noise=0.15)
    B = make_state("B", 0.20, rng, "names", link_noise=0.30, missing=4)
    emb = HashingEmbedder()
    mA = evaluate_run(A, gold, texts=texts, embed_fn=emb, n_resamples=200, n_random=20)
    mB = evaluate_run(B, gold, texts=texts, embed_fn=emb, n_resamples=200, n_random=20)
    return gold, texts, A, B, emb, mA, mB


def test_metrics_land_where_constructed(world):
    gold, texts, A, B, emb, mA, mB = world
    pa, pb = mA["primary"], mB["primary"]
    assert pa["k_produced"] == 6 and pb["k_produced"] == 6
    assert pa["nmi_l2"] > pb["nmi_l2"] > 0.4 and pa["p1_l2"] > pb["p1_l2"]
    assert pa["label_sim"] == pytest.approx(1.0)  # labels equal the gold names
    assert 0.5 < pb["label_sim"] < 1.0  # "athlete pages" shares a token
    assert pa["same_l2_p_at_5"] > pb["same_l2_p_at_5"] > mB["links"]["same_class"]["l2"]["chance"]
    assert mB["categories"]["n_uncategorised"] == 4 and mB["links"]["validity"]["dangling"] >= 0
    assert mA["labels"]["npmi"]["mean"] > mA["labels"]["npmi"]["per_category"].get("__none__", -1)
    assert mA["labels"]["npmi"]["mean"] > 0  # co-occurring class vocabulary
    assert mB["labels"]["npmi"] is None
    assert 0 <= mA["structure"]["orphan_rate"] <= 1 and mA["structure"]["n_pages"] == N
    assert mA["structure"]["depth"] == 2 and mA["structure"]["largest_wcc_frac"] > 0.5
    ci = mA["ci"]
    assert ci["nmi_l2"]["ci_low"] <= pa["nmi_l2"] <= ci["nmi_l2"]["ci_high"]
    assert ci["same_l2_p_at_5"]["ci_low"] <= pa["same_l2_p_at_5"] <= ci["same_l2_p_at_5"]["ci_high"]
    l2 = mA["categories"]["levels"]["l2"]
    assert l2["random_baseline"]["nmi"] < 0.2 < l2["nmi"]


def test_decision_supported_and_falsified(world, tmp_path):
    gold, texts, A, B, emb, mA, mB = world
    cmp = compare_arms(A, B, gold, embed_fn=emb, n_resamples=300)
    assert cmp["n_paired"] == N and cmp["n_uncategorised_b"] == 4 and cmp["pairing"] == "all"
    for e in ("E1", "E2", "E3"):
        assert cmp[e]["non_inferior"] and not cmp[e]["falsified"] and cmp[e]["delta"] == 0.10
    prices = load_prices()
    cost_a = arm_a_cost({"gpu_seconds": 60, "cpu_seconds": 120}, prices)
    calls = [{"kind": "ingest", "status": "ok", "doc_id": f"d{i}", "ingest_index": i, "wiki_pages_before": {"total": 4 * i + 3}, "model": "claude-sonnet-5-20260401",
              "tokens": {"input": 3000 + 40 * i, "output": 900, "cache_read": 30000 + 200 * i, "cache_creation": 2000}, "num_turns": 25, "wall_ms": 90000}
             for i in range(N)]
    cost_b = arm_b_cost(calls, prices, n_resamples=100)
    ratio = cost_ratio(cost_a["total_usd"], cost_b["total_usd"])
    assert ratio > 20 and cost_b["marginal_fit"]["ci_excludes_zero"]
    dec = decide({e: cmp[e] for e in ("E1", "E2", "E3")}, ratio, floors_from_metrics(mA))
    assert dec["verdict"] == "supported" and dec["floors_met"] and dec["cost"]["met"]
    # a random arm A is falsified on E1 and fails the NMI floor
    rng = np.random.default_rng(9)
    A_bad = make_state("A", 0.95, rng, "gold")
    cmp_bad = compare_arms(A_bad, B, gold, embed_fn=emb, n_resamples=200)
    assert cmp_bad["E1"]["falsified"]
    m_bad = evaluate_run(A_bad, gold, embed_fn=emb, with_ci=False, n_random=0)
    dec_bad = decide({e: cmp_bad[e] for e in ("E1", "E2", "E3")}, ratio, floors_from_metrics(m_bad))
    assert dec_bad["verdict"] == "falsified" and not dec_bad["floors"]["nmi_vs_l2_at_oracle_k"]["met"]
    # sensitivity pairing restricted to documents both arms categorised
    assert compare_arms(A, B, gold, embed_fn=emb, n_resamples=50, pairing="ok_both")["n_paired"] == N - 4
    # ledger -> tables -> figures
    ledger = tmp_path / "ledger.jsonl"
    rng = np.random.default_rng(2)
    states = {}
    for seed in range(3):
        st = make_state("A", 0.05 + 0.02 * seed, rng, "gold")
        states[f"s{seed}"] = st
        m = evaluate_run(st, gold, texts=texts, embed_fn=emb, n_resamples=100, n_random=10)
        report.append_ledger(report.make_row(arm="A", corpus="dbpedia", n_docs=N, replicate=seed, config={"k": 6}, k_variant="k_matched", k=6,
                                             metrics=strip_bulky(m), cost=cost_a, seeds={"kmeans": seed}), ledger)
    report.append_ledger(report.make_row(arm="B", corpus="dbpedia", n_docs=N, replicate="order-1", config={"schema": "v1"}, k_variant="native", k=6,
                                         metrics=strip_bulky(mB), cost=cost_b, model_id="claude-sonnet-5-20260401", cli_version="2.1.258", claude_md_sha256="x"), ledger)
    report.append_ledger(report.make_row(kind="decision", arm="A-vs-B", corpus="dbpedia", n_docs=N, replicate="headline", metrics=strip_bulky(dec)), ledger)
    report.append_ledger(report.make_row(kind="stability", arm="A", corpus="dbpedia", n_docs=N, replicate="3seeds", metrics={"rows": stability_table({"A/dbpedia": states})}), ledger)
    rows = report.read_ledger(ledger)
    assert len(rows) == 6 and report.headline_replicate(rows, "dbpedia", N) is not None
    written = report.write_tables(rows, tmp_path / "tables", ledger)
    t1 = pd.read_csv(written["category_metrics"])
    assert t1[t1["arm"] == "A"]["n_replicates"].iloc[0] == 3
    stab = pd.read_csv(written["stability"])
    assert stab["ari_mean"].iloc[0] > 0.7
    dec_csv = pd.read_csv(written["decision"])
    assert dec_csv["verdict"].iloc[0] == "supported"
    figs = figures.make_all(tmp_path / "tables", tmp_path / "figs")
    assert "F2_cost_quality" in figs and "F4_cost_per_ingest" in figs
