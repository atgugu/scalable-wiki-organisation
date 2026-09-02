#!/usr/bin/env python
"""Pre-registration v2, primary experiment R: run-to-run reproducibility of the organizing layer.

Given several WikiStates built from the SAME documents in the SAME order, compute for every pair:
  R1  partition agreement:   ARI and NMI of the two topic assignments over the shared documents
  R2  name recurrence:       fraction of topics whose (normalised) name recurs in the other run under a
                             best one-to-one matching, and the mean judge-embedding similarity of matched names
  R3  link agreement:        mean per-document Jaccard of the related-link sets; fraction of documents with
                             an identical link set
  descriptive:               topic / entity / concept page counts and their slug Jaccard (LLM wikis only)
and the pre-registered decision (configs/prereg_v2.yaml): mean cross-seed pipeline ARI minus mean LLM-wiki
ARI >= 0.20 with a document-subsampling 95% CI excluding 0, and byte-identical same-seed pipeline runs.

Usage:
  python scripts/02_reproducibility.py \
      --group pipeline_cross_seed runs/A/dbpedia/20/s{1,2,3,4,5}_oracle_lam0.7 \
      --group pipeline_same_seed  runs/A/dbpedia/20/s1_oracle_lam0.7 runs/A_identity/dbpedia/20/s1_oracle_lam0.7 \
      --group llm_wiki            runs/B/dbpedia/20/B_dbpedia_N300_order1@154d680 runs/B/dbpedia/20/R_dbpedia_N20_order1_r2 ... \
      --gold data/dbpedia/20/gold.jsonl --out results/reproducibility
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

from wikiorg.eval.state import coerce_state  # noqa: E402

N_RESAMPLES = 2000
SUBSAMPLE_FRAC = 0.8
SEED = 0


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
def load_state(run_dir: Path):
    ws = run_dir / "wiki_state.json"
    if not ws.exists():
        raise FileNotFoundError(f"{run_dir}: no wiki_state.json (for an LLM wiki, run `wikiorg extract` first)")
    return coerce_state(json.loads(ws.read_text(encoding="utf-8")))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gold_ids(path: Path) -> list[str]:
    return [json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------------------------
# per-run views
# ----------------------------------------------------------------------------------------------
def partition(state, ids: list[str]) -> list[str]:
    """Category id per document; uncategorised documents become singletons (as in the v1 metrics)."""
    out = []
    for i in ids:
        page = state.pages.get(i)
        cat = page.category if page is not None else None
        out.append(str(cat) if cat is not None else f"__singleton__{i}")
    return out


def labels_by_category(state) -> dict[str, str]:
    return {cid: str(c.label or cid) for cid, c in state.categories.items()}


def links(state, ids: list[str]) -> dict[str, set[str]]:
    return {i: set(state.pages[i].related or []) if i in state.pages else set() for i in ids}


_WS = re.compile(r"[^a-z0-9 ]+")


def norm_name(s: str) -> str:
    s = _WS.sub(" ", s.lower().replace("-", " ").replace("_", " ")).strip()
    s = re.sub(r"\s+", " ", s)
    # crude plural folding so "politicians" == "politician"
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in s.split())


# ----------------------------------------------------------------------------------------------
# pairwise metrics
# ----------------------------------------------------------------------------------------------
def r1_partition(a, b, ids):
    pa, pb = partition(a, ids), partition(b, ids)
    return {"ari": float(adjusted_rand_score(pa, pb)), "nmi": float(normalized_mutual_info_score(pa, pb))}


def _judge_embed(texts: list[str]):
    """mpnet judge embeddings (different family from the pipeline's nomic); CPU by default."""
    import os
    from sentence_transformers import SentenceTransformer
    dev = os.environ.get("WIKIORG_JUDGE_DEVICE", "cpu")
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=dev)
    return m.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def r2_names(a, b, embed_cache: dict) -> dict:
    la, lb = list(labels_by_category(a).values()), list(labels_by_category(b).values())
    if not la or not lb:
        return {"exact_recurrence": 0.0, "matched_similarity": 0.0, "n_a": len(la), "n_b": len(lb)}
    na, nb = [norm_name(x) for x in la], [norm_name(x) for x in lb]
    # judge similarity matrix
    for x in la + lb:
        if x not in embed_cache:
            embed_cache[x] = None
    missing = [x for x, v in embed_cache.items() if v is None]
    if missing:
        for x, e in zip(missing, _judge_embed(missing)):
            embed_cache[x] = e
    S = np.array([[float(embed_cache[x] @ embed_cache[y]) for y in lb] for x in la])
    # best one-to-one matching on judge similarity (Hungarian)
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-S)
    exact = sum(1 for i, j in zip(ri, ci) if na[i] == nb[j])
    sims = [S[i, j] for i, j in zip(ri, ci)]
    denom = max(len(la), len(lb))
    return {"exact_recurrence": exact / denom, "matched_similarity": float(np.mean(sims)) if sims else 0.0,
            "n_a": len(la), "n_b": len(lb), "matched_pairs": [(la[i], lb[j], round(float(S[i, j]), 3)) for i, j in zip(ri, ci)]}


def r3_links(a, b, ids):
    La, Lb = links(a, ids), links(b, ids)
    jac, ident = [], 0
    for i in ids:
        u = La[i] | Lb[i]
        j = (len(La[i] & Lb[i]) / len(u)) if u else 1.0
        jac.append(j)
        ident += int(La[i] == Lb[i])
    with_any = sum(1 for i in ids if La[i] or Lb[i]) / len(ids)
    return {"mean_jaccard": float(np.mean(jac)), "identical_fraction": ident / len(ids),
            "with_links_fraction": with_any, "per_doc": dict(zip(ids, jac))}


def page_inventory(state) -> dict:
    extra = getattr(state, "extra_pages", None) or []
    slugs = set()
    kinds = {}
    for p in extra:
        slug = p if isinstance(p, str) else (p.get("id") or p.get("slug") or p.get("path") or "")
        kind = "" if isinstance(p, str) else str(p.get("type") or p.get("kind") or "")
        if slug:
            slugs.add(str(slug))
            kinds[kind] = kinds.get(kind, 0) + 1
    return {"n_topics": len(state.categories), "n_extra_pages": len(slugs), "extra_kinds": kinds, "slugs": slugs}


# ----------------------------------------------------------------------------------------------
# bootstrap of the decision quantity
# ----------------------------------------------------------------------------------------------
def subsample_ci(states_pipe, states_llm, ids, n_resamples=N_RESAMPLES, frac=SUBSAMPLE_FRAC, seed=SEED):
    """95% CI of mean_pairwise_ARI(pipeline) - mean_pairwise_ARI(llm) by 80% document subsampling
    without replacement (duplicated documents would inflate partition agreement)."""
    rng = np.random.default_rng(seed)
    b = max(2, int(round(frac * len(ids))))
    pp = [partition(s, ids) for s in states_pipe]
    pl = [partition(s, ids) for s in states_llm]
    diffs = []
    for _ in range(n_resamples):
        idx = rng.choice(len(ids), size=b, replace=False)
        def mean_ari(parts):
            vals = [adjusted_rand_score([p[i] for i in idx], [q[i] for i in idx]) for p, q in itertools.combinations(parts, 2)]
            return float(np.mean(vals)) if vals else float("nan")
        diffs.append(mean_ari(pp) - mean_ari(pl))
    diffs = np.array(diffs, dtype=float)
    if not np.isfinite(diffs).any():
        return {"diff_mean": None, "ci_low": None, "ci_high": None, "n_resamples": n_resamples, "subsample": b}
    return {"diff_mean": float(np.nanmean(diffs)), "ci_low": float(np.nanpercentile(diffs, 2.5)),
            "ci_high": float(np.nanpercentile(diffs, 97.5)), "n_resamples": n_resamples, "subsample": b}


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", action="append", metavar=("NAME", "RUN_DIR"), required=True,
                    help="group name followed by run dirs (repeatable): pipeline_same_seed | pipeline_cross_seed | llm_wiki | ...")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default=str(REPO / "results" / "reproducibility"))
    ap.add_argument("--prereg", default=str(REPO / "configs" / "prereg_v2.yaml"))
    ap.add_argument("--ledger", default=str(REPO / "results" / "ledger.jsonl"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args(argv)

    ids = gold_ids(Path(args.gold))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, object]] = {}
    for g in args.group:
        name, dirs = g[0], [Path(d) for d in g[1:]]
        groups[name] = {Path(d).as_posix(): load_state(Path(d)) for d in dirs}

    embed_cache: dict = {}
    report = {"ids": ids, "n_docs": len(ids), "groups": {}}
    lines = ["# Reproducibility of the organizing layer (prereg v2, experiment R)", "",
             f"{len(ids)} documents, same order for every run.", ""]
    for name, runs in groups.items():
        keys = list(runs)
        pairs = []
        for (ka, a), (kb, b) in itertools.combinations(runs.items(), 2):
            p = {"a": ka, "b": kb, "R1": r1_partition(a, b, ids), "R2": r2_names(a, b, embed_cache), "R3": r3_links(a, b, ids)}
            ia, ib = page_inventory(a), page_inventory(b)
            u = ia["slugs"] | ib["slugs"]
            p["extra_slug_jaccard"] = (len(ia["slugs"] & ib["slugs"]) / len(u)) if u else None
            pairs.append(p)
        inv = {k: {kk: vv for kk, vv in page_inventory(s).items() if kk != "slugs"} for k, s in runs.items()}
        ident = None
        if len(keys) == 2:
            ident = sha256_file(Path(keys[0]) / "wiki_state.json") == sha256_file(Path(keys[1]) / "wiki_state.json")
        summary = {}
        for metric, path in (("ari", ("R1", "ari")), ("nmi", ("R1", "nmi")), ("name_exact", ("R2", "exact_recurrence")),
                             ("name_sim", ("R2", "matched_similarity")), ("link_jaccard", ("R3", "mean_jaccard")),
                             ("link_identical", ("R3", "identical_fraction")),
                             ("with_links", ("R3", "with_links_fraction"))):
            vals = [p[path[0]][path[1]] for p in pairs]
            summary[metric] = {"mean": float(np.mean(vals)) if vals else None, "min": float(np.min(vals)) if vals else None,
                               "max": float(np.max(vals)) if vals else None, "n_pairs": len(vals)}
        report["groups"][name] = {"runs": keys, "n_runs": len(keys), "pairs": [
            {k: (v if k != "R2" else {kk: vv for kk, vv in v.items() if kk != "matched_pairs"}) for k, v in p.items() if k != "R3"}
            | {"R3": {k: v for k, v in p["R3"].items() if k != "per_doc"}} for p in pairs],
            "summary": summary, "inventory": inv, "byte_identical": ident}
        lines.append(f"## {name}  ({len(keys)} runs, {len(pairs)} pairs)" + (f"  byte-identical: {ident}" if ident is not None else ""))
        lines.append("| metric | mean | min | max |")
        lines.append("|---|---|---|---|")
        for m, s in summary.items():
            if s["mean"] is not None:
                lines.append(f"| {m} | {s['mean']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")
        lines.append("")
        for k, v in inv.items():
            lines.append(f"- `{k}`: {v['n_topics']} topics, {v['n_extra_pages']} extra pages {v['extra_kinds'] or ''}")
        lines.append("")

    # decision (prereg v2)
    dec = {}
    enough = all(len(groups.get(g, {})) >= 2 for g in ("pipeline_cross_seed", "llm_wiki"))
    if not enough:
        lines += [f"## Pre-registered decision ({Path(args.prereg).stem})", "", "- not computed: needs >= 2 runs in both `pipeline_cross_seed` and `llm_wiki`", ""]
    if enough:
        ci = subsample_ci(list(groups["pipeline_cross_seed"].values()), list(groups["llm_wiki"].values()), ids)
        pipe = report["groups"]["pipeline_cross_seed"]["summary"]["ari"]["mean"]
        llm = report["groups"]["llm_wiki"]["summary"]["ari"]["mean"]
        same_ok = report["groups"].get("pipeline_same_seed", {}).get("byte_identical")
        supported = (pipe - llm >= 0.20) and (ci["ci_low"] > 0) and bool(same_ok)
        falsified = (llm >= 0.85) or (pipe < 0.70)
        dec = {"pipeline_cross_seed_ari": pipe, "llm_wiki_ari": llm, "difference": pipe - llm, "difference_ci": ci,
               "pipeline_same_seed_byte_identical": same_ok,
               "verdict": "falsified" if falsified else ("supported" if supported else "inconclusive"),
               "rule": "supported iff pipeline - llm >= 0.20 with CI excluding 0 AND same-seed byte-identical; "
                       "falsified iff llm ARI >= 0.85 or pipeline cross-seed ARI < 0.70"}
        lines += [f"## Pre-registered decision ({Path(args.prereg).stem})", "",
                  f"- pipeline cross-seed mean ARI: **{pipe:.3f}**", f"- LLM-wiki mean pairwise ARI: **{llm:.3f}**",
                  f"- difference: **{pipe - llm:+.3f}**, 95% subsampling CI [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]",
                  f"- same-seed pipeline runs byte-identical: **{same_ok}**", f"- **verdict: {dec['verdict']}**", ""]
    report["decision"] = dec

    (out_dir / "reproducibility.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not args.no_ledger:
        try:
            from wikiorg.eval.report import append_ledger, make_row
            row = make_row(kind="reproducibility", arm="R", corpus="dbpedia", n_docs=len(ids), replicate="prereg_v2_R",
                           metrics={"groups": {g: v["summary"] for g, v in report["groups"].items()}, "decision": dec})
            print(append_ledger(row, args.ledger, allow_supersede=True))
        except Exception as exc:  # noqa: BLE001
            print(f"(ledger append skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
