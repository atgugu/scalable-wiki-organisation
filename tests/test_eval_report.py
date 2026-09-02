"""Ledger semantics and table writers."""
import json

import numpy as np
import pandas as pd
import pytest

from wikiorg.eval import report
from wikiorg.eval.report import (LedgerConflict, append_ledger, config_hash, eval_code_hash, flatten, headline_replicate,
                                 make_row, read_ledger, verify_ledger, write_tables)


def _metrics(nmi, arm="A"):
    return {"primary_level": "l2", "n_docs": 300,
            "categories": {"k_produced": 6, "n_uncategorised": 0, "levels": {"l2": {"nmi": nmi, "ari": nmi - 0.1, "ami": nmi, "p1": 0.9, "purity": 0.95,
                                                                                    "inverse_purity": 0.9, "random_baseline": {"nmi": 0.1, "ari": 0.0, "p1": 0.4}},
                                                                             "l1": {"nmi": nmi - 0.05, "ari": 0.5, "p1": 0.8}}},
            "labels": {"weighted_mean": 0.6, "unweighted_mean": 0.55, "floor_random_weighted": 0.2, "jaccard_weighted": 0.3, "npmi": {"mean": 0.1} if arm == "A" else None},
            "links": {"same_class": {"l2": {"precision_at_k": 0.8, "chance": 0.2}, "l1": {"precision_at_k": 0.9, "chance": 0.3}},
                      "category_mix": {"same_category_frac": 0.7, "unique_category_frac": 0.4, "mean_links_per_doc": 5},
                      "validity": {"resolved_frac": 1.0, "near_miss_frac": 0.0, "dangling_frac": 0.0}},
            "structure": {"n_pages": 300, "n_categories": 6, "depth": 2, "orphan_rate": 0.1, "reciprocity_gl": 0.3, "density": 0.02},
            "ci": {"nmi_l2": {"ci_low": nmi - 0.03, "ci_high": nmi + 0.03}, "label_sim": {"ci_low": 0.55, "ci_high": 0.65}}}


def test_ledger_append_duplicate_conflict_supersede(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    row = make_row(arm="A", corpus="dbpedia", n_docs=300, replicate=0, config={"k": 6}, k_variant="k_matched", k=6,
                   metrics=_metrics(0.8), cost={"total_usd": 0.01}, seeds={"kmeans": 0})
    for f in ("git_hash", "git_dirty", "config_hash", "eval_code_hash", "package_versions", "timestamp", "row_id", "python", "host"):
        assert f in row
    assert row["config_hash"] == config_hash({"k": 6}) and len(row["eval_code_hash"]) == 64
    assert row["package_versions"]["numpy"] and row["package_versions"]["sklearn"]
    assert append_ledger(row, ledger)["status"] == "appended"
    assert append_ledger(row, ledger)["status"] == "duplicate"
    assert len(ledger.read_text().splitlines()) == 1  # append-only, no duplicate line
    changed = make_row(arm="A", corpus="dbpedia", n_docs=300, replicate=0, config={"k": 6}, k_variant="k_matched", k=6,
                       metrics=_metrics(0.85), cost={"total_usd": 0.01})
    with pytest.raises(LedgerConflict):
        append_ledger(changed, ledger)
    assert append_ledger(changed, ledger, allow_supersede=True)["status"] == "superseded"
    lines = ledger.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["supersedes"] == row["row_id"]
    latest = read_ledger(ledger)
    assert len(latest) == 1 and latest[0]["metrics"]["categories"]["levels"]["l2"]["nmi"] == 0.85
    assert len(read_ledger(ledger, latest_only=False)) == 2
    other = make_row(arm="A", corpus="dbpedia", n_docs=300, replicate=1, config={"k": 6}, k_variant="k_matched", k=6, metrics=_metrics(0.7), cost={})
    append_ledger(other, ledger)
    assert len(read_ledger(ledger)) == 2
    v = verify_ledger(ledger)
    assert v["ok"] or all(p["missing"] == "git_hash" for p in v["problems"])  # no commits yet in a fresh repo
    b = make_row(arm="B", corpus="dbpedia", n_docs=300, replicate="order-1", config={"s": 1}, metrics=_metrics(0.7, "B"), cost={"total_usd": 20})
    append_ledger(b, ledger)
    assert any(p["missing"] == "model_id" for p in verify_ledger(ledger)["problems"])


def test_tables_and_headline(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    for seed, nmi in enumerate([0.80, 0.86, 0.83, 0.90, 0.78]):
        append_ledger(make_row(arm="A", corpus="dbpedia", n_docs=300, replicate=seed, config={"k": 6}, k_variant="k_matched", k=6,
                               metrics=_metrics(nmi), cost={"total_usd": 0.012, "gpu_seconds": 90}), ledger)
    append_ledger(make_row(arm="A", corpus="dbpedia", n_docs=300, replicate=0, config={"k": 6, "mmr_lambda": 1.0}, k_variant="k_matched", k=6,
                           variant="mmr_1.0", metrics=_metrics(0.8), cost={"total_usd": 0.012}), ledger)
    for k in (3, 6, 12):
        append_ledger(make_row(arm="A", corpus="dbpedia", n_docs=300, replicate=0, config={"k": k}, k_variant="k_grid", k=k, metrics=_metrics(0.8), cost={}), ledger)
    append_ledger(make_row(arm="B", corpus="dbpedia", n_docs=300, replicate="order-1", config={"s": 1}, k_variant="native", k=7, metrics=_metrics(0.7, "B"),
                           cost={"total_usd": 20.0, "uncached_usd": 60.0, "num_turns": 6000, "n_calls": 300,
                                 "marginal_fit": {"slope": 1e-4, "intercept": 0.02, "slope_ci_low": 0.5e-4, "slope_ci_high": 1.5e-4},
                                 "per_ingest": [{"ingest_index": i, "wiki_pages_before": 4 * i, "usd": 0.02 + 1e-4 * 4 * i, "tokens": 30000} for i in range(50)]},
                           model_id="claude-sonnet-5-x", cli_version="2.1.258", claude_md_sha256="abc"), ledger)
    append_ledger(make_row(kind="human", arm="study", corpus="dbpedia", n_docs=300, replicate="v1",
                           metrics={"rows": [{"variant": "all_raters", "arm": "A", "mean": 3.1, "ci_low": 2.8, "ci_high": 3.4},
                                             {"variant": "all_raters", "arm": "B", "mean": 4.0, "ci_low": 3.7, "ci_high": 4.3}]}), ledger)
    append_ledger(make_row(kind="stability", arm="A", corpus="dbpedia", n_docs=300, replicate="5seeds",
                           metrics={"rows": [{"group": "A/dbpedia/300", "ari_mean": 0.9, "ari_sd": 0.02, "nmi_mean": 0.92, "nmi_sd": 0.01}]}), ledger)
    append_ledger(make_row(kind="decision", arm="A-vs-B", corpus="dbpedia", n_docs=300, replicate="headline", metrics={"verdict": "supported"}), ledger)
    rows = read_ledger(ledger)
    assert headline_replicate(rows, "dbpedia", 300)["replicate"] == 2  # median of the five NMIs (0.83)
    written = write_tables(rows, tmp_path / "tables", ledger)
    for name in ("category_metrics", "category_metrics_runs", "link_metrics", "label_metrics", "structure_metrics", "cost_quality",
                 "ablation_mmr", "k_sensitivity", "cost_per_ingest", "human_eval", "stability", "decision"):
        assert name in written and written[name].exists(), name
    t1 = pd.read_csv(written["category_metrics"])
    a = t1[(t1["arm"] == "A") & (t1["k_variant"] == "k_matched") & (t1["variant"].isna())].iloc[0]
    assert a["n_replicates"] == 5 and a["nmi_l2"] == pytest.approx(np.mean([0.80, 0.86, 0.83, 0.90, 0.78]))
    assert a["nmi_l2_sd"] == pytest.approx(np.std([0.80, 0.86, 0.83, 0.90, 0.78], ddof=1))
    assert "nmi_l2_random" in t1 and "nmi_l2_ci_low" in t1
    t3 = pd.read_csv(written["label_metrics_runs"])
    assert t3[t3["arm"] == "B"]["human_mean"].iloc[0] == 4.0
    t4 = pd.read_csv(written["structure_metrics"])
    assert "Kamps" in t4["arm"].iloc[-1] and t4["median_in_degree"].iloc[-1] == 4
    cq = pd.read_csv(written["cost_quality"])
    assert cq[cq["arm"] == "B"]["usd"].iloc[0] == 20.0
    ks = pd.read_csv(written["k_sensitivity"])
    assert sorted(ks["k"]) == [3, 6, 12]
    cpi = pd.read_csv(written["cost_per_ingest"])
    assert len(cpi) == 50 and cpi["fit_slope"].iloc[0] == pytest.approx(1e-4)
    mmr = pd.read_csv(written["ablation_mmr"])
    assert mmr["mmr_lambda"].tolist() == [1.0]
    assert json.loads((tmp_path / "tables" / "manifest.json").read_text())["eval_code_hash"] == eval_code_hash()
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
    assert write_tables([], tmp_path / "empty", ledger) == {}
