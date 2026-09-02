"""Run-to-run stability: pairwise ARI / NMI between repeated partitions.

Arm A is repeated over 5 seeds, arm B over two ingest orders and arm C over two
runs; the mean +/- sd of the pairwise agreement is the *noise floor* against
which every between-arm difference is read (pre-registration, threats).  The
same functions also serve k-sensitivity (agreement between k variants).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn import metrics as skm

from .categories import _as_int_codes
from .state import WikiState, coerce_state


def _partition(obj: Any, ids: Sequence[str] | None) -> dict[str, str]:
    if isinstance(obj, Mapping) and obj and all(isinstance(v, str) for v in obj.values()) and "pages" not in obj:
        part = dict(obj)
    else:
        st = coerce_state(obj)
        part = st.partition(ids if ids is not None else st.ids(), singleton_uncategorised=True)
    return part


def pairwise_agreement(runs: Mapping[str, Any] | Sequence[Any], ids: Sequence[str] | None = None) -> dict[str, Any]:
    """ARI/NMI between every pair of runs over the documents common to all.

    ``runs`` maps a run name to a WikiState (or to a ``{doc_id: cluster}``
    dict).  Returns mean, sd (ddof=1), min, max, the list of pairs and the
    number of common documents.
    """
    if not isinstance(runs, Mapping):
        runs = {str(i): r for i, r in enumerate(runs)}
    parts = {name: _partition(r, ids) for name, r in runs.items()}
    common: set[str] | None = None
    for p in parts.values():
        common = set(p) if common is None else common & set(p)
    common_ids = sorted(common or [])
    if ids is not None:
        common_ids = [i for i in ids if i in set(common_ids)]
    pairs = []
    for a, b in combinations(parts, 2):
        la = _as_int_codes([parts[a][i] for i in common_ids])
        lb = _as_int_codes([parts[b][i] for i in common_ids])
        if len(common_ids) == 0:
            ari = nmi = float("nan")
        else:
            ari = float(skm.adjusted_rand_score(la, lb))
            nmi = float(skm.normalized_mutual_info_score(la, lb))
        pairs.append({"a": a, "b": b, "ari": ari, "nmi": nmi})
    out: dict[str, Any] = {"n_runs": len(parts), "n_pairs": len(pairs), "n_docs": len(common_ids), "pairs": pairs}
    for m in ("ari", "nmi"):
        vals = np.array([p[m] for p in pairs], dtype=float)
        out[f"{m}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
        out[f"{m}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        out[f"{m}_min"] = float(np.nanmin(vals)) if len(vals) else float("nan")
        out[f"{m}_max"] = float(np.nanmax(vals)) if len(vals) else float("nan")
    ks = {name: len(set(p.values())) for name, p in parts.items()}
    out["k_per_run"] = ks
    return out


def stability_table(groups: Mapping[str, Mapping[str, Any]], ids_by_group: Mapping[str, Sequence[str]] | None = None) -> list[dict[str, Any]]:
    """One row per group (e.g. ``"A/dbpedia/300"``) for T6."""
    rows = []
    for gname, runs in groups.items():
        ids = ids_by_group.get(gname) if ids_by_group else None
        res = pairwise_agreement(runs, ids)
        rows.append({"group": gname, "n_runs": res["n_runs"], "n_pairs": res["n_pairs"], "n_docs": res["n_docs"],
                     "ari_mean": res["ari_mean"], "ari_sd": res["ari_sd"], "ari_min": res["ari_min"],
                     "nmi_mean": res["nmi_mean"], "nmi_sd": res["nmi_sd"], "nmi_min": res["nmi_min"],
                     "k_per_run": ";".join(f"{k}={v}" for k, v in res["k_per_run"].items())})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """``wikiorg stability --glob 'runs/A/dbpedia/300/s*_k_matched_lam0.7' --group A/dbpedia/300``."""
    import argparse
    import glob as _glob
    from pathlib import Path

    from .report import LEDGER_PATH, append_ledger, load_corpus, make_row, resolve_corpus_dir
    from .state import WikiState

    p = argparse.ArgumentParser(prog="wikiorg stability", description="pairwise ARI/NMI between repeated runs (noise floor)")
    p.add_argument("--run", action="append", default=[], help="run directory (repeatable)")
    p.add_argument("--glob", action="append", default=[], help="glob of run directories (repeatable)")
    p.add_argument("--group", help="label of the group (default: arm/corpus/N of the first run)")
    p.add_argument("--corpus-dir")
    p.add_argument("--ledger", default=str(LEDGER_PATH))
    p.add_argument("--no-append", action="store_true")
    argv = list(argv or [])
    while argv and argv[0] == "--":
        argv.pop(0)
    args = p.parse_args(argv)
    runs = [Path(r) for r in args.run] + [Path(x) for g in args.glob for x in sorted(_glob.glob(g))]
    runs = [r for r in runs if (r / "wiki_state.json").exists()]
    if len(runs) < 2:
        print("need at least two run directories with wiki_state.json", file=sys.stderr)
        return 2
    import json
    manifest = json.loads((runs[0] / "manifest.json").read_text()) if (runs[0] / "manifest.json").exists() else {}
    corpus_dir = resolve_corpus_dir(manifest, args.corpus_dir)
    gold, _ = load_corpus(corpus_dir)
    states = {r.name: WikiState.load(r / "wiki_state.json") for r in runs}
    arm = str(manifest.get("arm") or next(iter(states.values())).arm)
    corpus = str(manifest.get("corpus") or corpus_dir.parent.name)
    n_docs = len(gold.ids())
    group = args.group or f"{arm}/{corpus}/{n_docs}"
    res = pairwise_agreement(states, ids=gold.ids())
    rows = stability_table({group: states}, {group: gold.ids()})
    print(f"{group}: {res['n_runs']} runs, {res['n_pairs']} pairs over {res['n_docs']} docs: "
          f"ARI {res['ari_mean']:.3f} +/- {res['ari_sd']:.3f}, NMI {res['nmi_mean']:.3f} +/- {res['nmi_sd']:.3f}")
    if not args.no_append:
        row = make_row(kind="stability", arm=arm, corpus=corpus, n_docs=n_docs, replicate=group,
                       metrics={"rows": rows, "pairs": res["pairs"], "k_per_run": res["k_per_run"], "runs": [str(r) for r in runs]})
        print(append_ledger(row, args.ledger, allow_supersede=True))
    return 0


import sys  # noqa: E402
