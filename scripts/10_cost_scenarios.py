#!/usr/bin/env python
"""Conservative deployment scenarios for the revised cost analysis (files on disk only; no model calls).

Two things the earlier scaling table (scripts/09_scaling_table.py) lacked:

1. Both LLM-wiki cost fits with uncertainty. The pre-registered fit frozen at ingest 20 is the
   stop-rule fit and the one cited for the secondary endpoint (docs/decision_log.md); the refit on
   all 62 ingests of the longest run is post-hoc and descriptive. Each fit: OLS of ingest USD on
   wiki pages before the ingest, BCa bootstrap interval for the slope over ingests, R^2, Spearman
   rho, first- and last-quintile means (all from wikiorg.eval.cost.marginal_cost_fit). The frozen
   fits are asserted to reproduce the coefficients stored in the run's decisions.jsonl.
2. The pipeline rebuild priced with a hosted embedding API for the same encoder (Nomic Embed
   v1.5) on the token counts recorded by the embedding cache, added to the full measured compute
   charge. The compute charge already contains the GPU embedding time, so the sum double-counts
   and the resulting ratio is a lower bound. Only the four measured DBPedia tiers are reported;
   the agent column is extrapolated in every row and says so.

Outputs
  results/tables/cost_scenarios.json        every coefficient, interval, token total, price constant
  results/tables/embedding_tokens.csv       per-document token counts behind the hosted-embedding column
  results/tables/cost_fits.{csv,md}         the fits table (replaces supp_costgrowth in the revised text)
  results/tables/scaling_hosted.{csv,md}    the scenario table (replaces scaling in the revised text)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wikiorg.eval.cost import _ols, load_prices, marginal_cost_fit  # noqa: E402

EXTRACT = REPO / "results" / "extract"
TABLES = REPO / "results" / "tables"
GEN = REPO / "thesis" / "build" / "generated"
SIDECAR_DIR = REPO / "runs" / "embeddings" / "nomic-embed-text-v1.5_L2048"
TIERS = (100, 300, 1000, 2000)
MAX_TOKENS = 2048
FROZEN_AT = 20
WIKIS = [("B_dbpedia_N300_order1", "N62", "DBPedia, order 1"),
         ("B_dbpedia_N100_order2", "N30", "DBPedia, order 2"),
         ("B_wikics_N100_order1", "N32", "WikiCS")]
HOSTED_MODEL = "nomic-embed-text-v1.5"


def calls_path(rid: str, suffix: str) -> Path:
    """The archived call log; the sanitized evidence bundle is the fallback in the release repository."""
    p = EXTRACT / f"{rid}_{suffix}.calls.jsonl"
    return p if p.exists() else REPO / "evidence" / "ingests" / f"{rid}_{suffix}.jsonl"


TOKENS_CSV = TABLES / "embedding_tokens.csv"


def token_counts_per_document() -> dict[str, dict]:
    """Per-document token counts of the DBPedia tiers (nested 100 < 300 < 1000 < 2000; ``first_tier`` is
    the smallest tier containing the document). Read from the embedding-cache sidecars when they are
    present (local archive) and written to results/tables/embedding_tokens.csv for the release, which
    carries no run directories; otherwise read back from that CSV."""
    sidecar_files = sorted(SIDECAR_DIR.glob("dbpedia-*.json"))
    if sidecar_files:
        sidecars = [json.load(open(p)) for p in sidecar_files]
        per_doc: dict[str, dict] = {}
        for n in TIERS:
            for doc_id in sorted(p.stem for p in (REPO / "data" / "dbpedia" / str(n) / "docs").glob("*.md")):
                if doc_id in per_doc:
                    continue
                for sc in sidecars:
                    if doc_id in sc["n_tokens"]:
                        per_doc[doc_id] = {"first_tier": n, "n_tokens": int(sc["n_tokens"][doc_id]),
                                           "truncated": int(bool(sc["truncated"][doc_id]))}
                        break
                else:
                    raise KeyError(f"{doc_id} not in any sidecar")
        with open(TOKENS_CSV, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["corpus", "first_tier", "doc_id", "n_tokens", "truncated"])
            for doc_id, v in sorted(per_doc.items(), key=lambda kv: (kv[1]["first_tier"], kv[0])):
                w.writerow(["dbpedia", v["first_tier"], doc_id, v["n_tokens"], v["truncated"]])
        return per_doc
    per_doc = {}
    for r in csv.DictReader(open(TOKENS_CSV)):
        per_doc[r["doc_id"]] = {"first_tier": int(r["first_tier"]), "n_tokens": int(r["n_tokens"]), "truncated": int(r["truncated"])}
    return per_doc


def load_rows(rid: str, suffix: str) -> list[dict]:
    p = calls_path(rid, suffix)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def ingests(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("kind") == "ingest" and r.get("status") == "ok"]


def fit_block(rows: list[dict], prices: dict, label: str, kind: str) -> dict:
    f = marginal_cost_fit(rows, prices)          # OLS + BCa over ingests, 2,000 resamples, seed 0
    pages = np.array([r["wiki_pages_before"]["total"] for r in rows], float)
    idx = np.arange(1, len(rows) + 1, dtype=float)
    g, g0, g_r2 = _ols(idx, pages)
    return {
        "run": label, "fit": kind, "n": f["n"],
        "intercept_usd": f["intercept"], "slope_usd_per_page": f["slope"],
        "slope_usd_per_100_pages": f["slope_per_100_pages"],
        "slope_ci_low_per_100": f["slope_ci_low_per_100"], "slope_ci_high_per_100": f["slope_ci_high_per_100"],
        "ci_method": f["ci_method"], "ci_excludes_zero": f["ci_excludes_zero"],
        "r2": f["r2"], "spearman_rho": f["spearman_rho"],
        "first_quintile_mean_usd": f["y_first_quintile_mean"], "last_quintile_mean_usd": f["y_last_quintile_mean"],
        "pages_min": f["x_min"], "pages_max": f["x_max"],
        "pages_per_ingest": g, "pages_intercept": g0, "pages_r2": g_r2,
        "resampling": ("BCa bootstrap of the OLS slope over ingests, 2,000 resamples, seed 0; ingests are "
                       "treated as exchangeable and serial dependence is not modelled"),
    }


def main() -> int:
    prices = load_prices()
    hosted = prices["hosted_embedding"]
    price_nomic = float(hosted[HOSTED_MODEL]["usd_per_mtok"])

    # ---- 1. the fits -------------------------------------------------------------------------
    fits, checks = [], []
    for rid, suffix, label in WIKIS:
        rows = load_rows(rid, suffix)
        ing = ingests(rows)
        run_total = float(sum(r.get("cost_usd") or 0 for r in rows))       # all rows, lint included (as Table 4)
        last_pages = int(ing[-1]["wiki_pages_before"]["total"])
        first = [r for r in ing if int(r.get("ingest_index") or 0) <= FROZEN_AT]
        fb = fit_block(first, prices, label, f"frozen at {FROZEN_AT}")
        dec_path = EXTRACT / f"{rid}.decisions.jsonl"
        if dec_path.exists():                      # the stop-rule record written at ingest 20 (local archive)
            dec = json.loads(next(l for l in dec_path.read_text().splitlines() if l.strip()))
            diffs = {"slope": abs(fb["slope_usd_per_page"] - dec["cost_slope_usd_per_page"]),
                     "intercept": abs(fb["intercept_usd"] - dec["cost_intercept_usd"]),
                     "pages_per_ingest": abs(fb["pages_per_ingest"] - dec["pages_slope_per_ingest"])}
            assert all(v < 1e-6 for v in diffs.values()), (rid, diffs)
            checks.append({"run": rid, "frozen_fit_reproduced_to": max(diffs.values()), "n_done": dec["n_done"]})
        else:
            checks.append({"run": rid, "frozen_fit_reproduced_to": None, "note": "decisions.jsonl not available; fit not cross-checked"})
        src = str(calls_path(rid, suffix).relative_to(REPO))
        fb.update({"last_pages": last_pages, "total_usd": run_total, "ingests_total": len(ing), "source": src})
        fits.append(fb)
        if len(ing) > FROZEN_AT and rid == "B_dbpedia_N300_order1":
            fa = fit_block(ing, prices, label, f"post-hoc, all {len(ing)}")
            fa.update({"last_pages": last_pages, "total_usd": run_total, "ingests_total": len(ing), "source": src})
            fits.insert(1, fa)
    frozen = fits[0]
    refit = fits[1]
    assert refit["fit"].startswith("post-hoc")

    # ---- 2. token totals: embedding-cache sidecars locally, results/tables/embedding_tokens.csv in the release
    per_doc = token_counts_per_document()
    tokens = {}
    for n in TIERS:
        c = np.array([v["n_tokens"] for v in per_doc.values() if v["first_tier"] <= n])
        trunc = sum(v["truncated"] for v in per_doc.values() if v["first_tier"] <= n)
        assert len(c) == n, (n, len(c))
        assert trunc == 0 and c.max() <= MAX_TOKENS, (n, trunc, int(c.max()))
        tokens[n] = {"docs": n, "tokens": int(c.sum()), "mean_per_doc": float(c.mean()), "max_per_doc": int(c.max()),
                     "truncated_docs": int(trunc), "cap_tokens": n * MAX_TOKENS}

    # ---- 3. measured pipeline rebuild (mean over five oracle, lambda 0.7 seeds) ----------------
    compute = {}
    for r in csv.DictReader(open(TABLES / "supp_armA.csv")):
        if r["corpus"] == "dbpedia" and int(r["N"]) in TIERS:
            v = float(r["usd"])
            assert compute.setdefault(int(r["N"]), v) == v, ("supp_armA duplicate rows differ", r)
    assert set(compute) == set(TIERS)

    # ---- 4. scenarios ---------------------------------------------------------------------------
    a, b, g = refit["intercept_usd"], refit["slope_usd_per_page"], refit["pages_per_ingest"]
    b_lo, b_hi = refit["slope_ci_low_per_100"] / 100, refit["slope_ci_high_per_100"] / 100
    a20, b20, g20 = frozen["intercept_usd"], frozen["slope_usd_per_page"], frozen["pages_per_ingest"]
    scen = []
    for n in TIERS:
        pages = g * n
        nxt = a + b * pages
        emb = tokens[n]["tokens"] / 1e6 * price_nomic
        comp = compute[n]
        rebuild = emb + comp
        cap_rebuild = tokens[n]["cap_tokens"] / 1e6 * price_nomic + comp
        row = {"documents": n, "index_pages": pages, "beyond_fit": pages > refit["pages_max"],
               "agent_next_ingest_usd": nxt,
               "agent_next_ingest_slope_ci_usd": [a + b_lo * pages, a + b_hi * pages],
               "agent_next_ingest_frozen_fit_usd": a20 + b20 * g20 * n,
               "hosted_embedding_usd": emb, "structural_compute_usd": comp, "hosted_rebuild_usd": rebuild,
               "ratio": nxt / rebuild, "ratio_compute_only": nxt / comp,
               "ratio_slope_ci": [(a + b_lo * pages) / rebuild, (a + b_hi * pages) / rebuild],
               "cap_bound_rebuild_usd": cap_rebuild, "cap_bound_ratio": nxt / cap_rebuild,
               "sensitivity": {m: {"hosted_rebuild_usd": tokens[n]["tokens"] / 1e6 * float(v["usd_per_mtok"]) + comp,
                                   "ratio": nxt / (tokens[n]["tokens"] / 1e6 * float(v["usd_per_mtok"]) + comp)}
                               for m, v in hosted.items() if isinstance(v, dict)}}
        scen.append(row)

    # ---- 5. Arm C, the structure-only LLM comparator (measured, for prose) --------------------
    armc = {}
    for p in sorted((REPO / "runs" / "C").glob("*/*/*/manifest.json")):
        m = json.load(open(p))
        tot = (m.get("totals") or {}).get("cost_usd")
        if tot is not None:
            armc[str(p.relative_to(REPO).parent)] = {"model": m.get("model"), "n": m.get("n") or m.get("n_docs"), "cost_usd": tot}

    # ---- 6. write -----------------------------------------------------------------------------
    out = {"generated_by": "scripts/10_cost_scenarios.py", "prices": {"hosted_embedding": hosted, "path": prices["_path"]},
           "frozen_fit_checks": checks, "fits": fits, "tokens": tokens, "structural_compute_usd": compute,
           "scenarios": scen, "arm_c": armc,
           "notes": ["agent next ingest = intercept + slope x (pages per ingest x documents), from the post-hoc 62-ingest fit "
                     "(the shallower slope of the two fits); every scenario row lies beyond the 469-page fit domain",
                     "slope interval rows hold the intercept at its estimate",
                     "hosted embedding = recorded token count x hosted price; no DBPedia document reaches the 2,048-token cap, so "
                     "the local and hosted inputs are identical",
                     "structural compute = measured cloud-equivalent full rebuild (mean over five seeds), which already contains the "
                     "GPU embedding time; adding the hosted charge double-counts it, so the ratio is a lower bound"]}
    (TABLES / "cost_scenarios.json").write_text(json.dumps(out, indent=2) + "\n")

    def table(stem, cols, rows, caption):
        with open(TABLES / f"{stem}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows({c: r[c] for c in cols} for r in rows)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        lines += ["", f"Table: {caption} {{#tbl_{stem}}}"]
        (TABLES / f"{stem}.md").write_text("\n".join(lines) + "\n")
        GEN.mkdir(parents=True, exist_ok=True)
        (GEN / f"tbl_{stem}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines[:2 + len(rows)]))

    fit_rows = [{"run": f["run"], "fit": f["fit"], "intercept_usd": round(f["intercept_usd"], 3),
                 "slope_usd_per_100_pages": round(f["slope_usd_per_100_pages"], 3),
                 "slope_ci_per_100": (f"[{f['slope_ci_low_per_100']:.3f}, {f['slope_ci_high_per_100']:.3f}]").replace("-", "\u2212"),
                 "r2": round(f["r2"], 2), "last_pages": f["last_pages"], "total_usd": round(f["total_usd"], 2)} for f in fits]
    table("cost_fits", ["run", "fit", "intercept_usd", "slope_usd_per_100_pages", "slope_ci_per_100", "r2", "last_pages", "total_usd"],
          fit_rows, "LLM-wiki ingest cost against wiki pages before the ingest: the pre-registered fits frozen at ingest 20 and the "
          "post-hoc refit on all 62 ingests of the first order. Slope in USD per 100 pages with a BCa bootstrap interval over ingests.")
    scen_rows = [{"Documents": f"{r['documents']:,}", "Agent, next ingest": f"${r['agent_next_ingest_usd']:,.2f}",
                  "Hosted embedding": f"${r['hosted_embedding_usd']:.4f}", "Structural compute": f"${r['structural_compute_usd']:.5f}",
                  "Conservative hosted rebuild": f"${r['hosted_rebuild_usd']:.4f}", "Lower-bound ratio": f"{r['ratio']:.0f}"} for r in scen]
    table("scaling_hosted", ["Documents", "Agent, next ingest", "Hosted embedding", "Structural compute", "Conservative hosted rebuild", "Lower-bound ratio"],
          scen_rows, "Conservative deployment scenarios at the measured DBPedia tiers: the extrapolated next agent ingest against a "
          "pipeline rebuild whose embeddings are bought from the hosted Nomic Embed v1.5 API at the recorded token counts.")
    print("\nfits:")
    for f in fits:
        print(f"  {f['run']:18s} {f['fit']:16s} n={f['n']:2d} a=${f['intercept_usd']:.4f} b=${f['slope_usd_per_100_pages']:.4f}/100p "
              f"CI [{f['slope_ci_low_per_100']:.4f}, {f['slope_ci_high_per_100']:.4f}] ({f['ci_method']}) R2={f['r2']:.3f} "
              f"rho={f['spearman_rho']:.3f} g={f['pages_per_ingest']:.3f} quintiles ${f['first_quintile_mean_usd']:.3f}->${f['last_quintile_mean_usd']:.3f}")
    for r in scen:
        print(f"  N={r['documents']:5d} next=${r['agent_next_ingest_usd']:.3f} ci=[{r['agent_next_ingest_slope_ci_usd'][0]:.3f},{r['agent_next_ingest_slope_ci_usd'][1]:.3f}] "
              f"frozen=${r['agent_next_ingest_frozen_fit_usd']:.3f} emb=${r['hosted_embedding_usd']:.5f} comp=${r['structural_compute_usd']:.5f} "
              f"ratio={r['ratio']:.1f} ci_ratio=[{r['ratio_slope_ci'][0]:.1f},{r['ratio_slope_ci'][1]:.1f}] cap_ratio={r['cap_bound_ratio']:.1f} "
              f"compute_only={r['ratio_compute_only']:.0f} openai={r['sensitivity']['text-embedding-3-small']['ratio']:.0f} voyage={r['sensitivity']['voyage-4']['ratio']:.0f}")
    print("  tokens:", {n: (t["tokens"], round(t["mean_per_doc"], 1), t["max_per_doc"]) for n, t in tokens.items()})
    print("  arm C:", armc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
