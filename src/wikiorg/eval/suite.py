"""Run-level evaluation and the arm-A-vs-B comparison feeding ``decide``.

``evaluate_run`` computes every metric block for one WikiState against the
gold (categories, labels, links, structure) plus CIs for the headline numbers;
``compare_arms`` builds the three pre-registered endpoints from two states on
the same gold, paired over documents.

Pairing rule for E2/E3 (recorded in the output): the evaluation universe is
the gold document set; a document the arm produced no page or no category for
scores 0 on label similarity and same-class precision, exactly as it forms a
singleton cluster in E1.  ``pairing="ok_both"`` restricts to documents both
arms categorised (reported as a sensitivity variant, never the headline).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .bootstrap import (N_RESAMPLES, mean_ci, non_inferiority, non_inferiority_partition, prereg_thresholds,
                        subsample_metric_ci)
from .categories import evaluate_categories, predicted_labels
from .labels import evaluate_labels, label_similarity
from .links import evaluate_links, same_class_precision
from .state import Gold, WikiState, coerce_state, common_ids
from .structure import structure_metrics

BULKY_KEYS = ("per_document", "per_cluster", "flagged_calls", "broken_links", "not_in_index", "no_inbound", "dist")


def primary_level(gold: Gold) -> str:
    if "l2" in gold.levels:
        return "l2"
    if "class" in gold.levels:
        return "class"
    return gold.levels[0]


def strip_bulky(obj: Any, keys: Sequence[str] = BULKY_KEYS) -> Any:
    """Drop per-document dictionaries etc. before a metrics dict enters the ledger."""
    if isinstance(obj, Mapping):
        return {k: strip_bulky(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [strip_bulky(v, keys) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def evaluate_run(state: WikiState | Any, gold: Gold, *, texts: Mapping[str, str] | None = None,
                 embed_fn: Callable | None = None, judge_embeddings: Mapping[str, np.ndarray] | None = None,
                 k: int = 5, seed: int = 0, n_random: int = 100, with_ci: bool = True,
                 n_resamples: int = N_RESAMPLES, corpus_titles: Mapping[str, str] | None = None,
                 levels: Sequence[str] | None = None) -> dict[str, Any]:
    """Every metric block for one run.  ``metrics["primary"]`` is the flat
    summary the tables use; ``metrics["ci"]`` holds the headline CIs."""
    st = coerce_state(state)
    lvl = primary_level(gold)
    ids = common_ids(st, gold, lvl)
    m: dict[str, Any] = {
        "primary_level": lvl,
        "n_docs": len(ids),
        "categories": evaluate_categories(st, gold, levels=levels if levels is not None else [l for l in gold.levels if l != "l3"],
                                          n_random=n_random, seed=seed),
        "labels": evaluate_labels(st, gold, level=lvl, texts=texts, embed_fn=embed_fn, seed=seed),
        "links": evaluate_links(st, gold, k=k, embeddings=judge_embeddings, corpus_titles=corpus_titles,
                                seed=seed, n_random=n_random),
        "structure": structure_metrics(st),
    }
    cat = m["categories"]["levels"][lvl]
    prim = {"k_produced": m["categories"]["k_produced"], f"nmi_{lvl}": cat["nmi"], f"ari_{lvl}": cat["ari"],
            f"ami_{lvl}": cat["ami"], f"p1_{lvl}": cat["p1"], "label_sim": m["labels"]["weighted_mean"],
            "label_sim_unweighted": m["labels"]["unweighted_mean"],
            f"same_{lvl}_p_at_{k}": m["links"]["same_class"][lvl]["precision_at_k"]}
    if "l1" in m["categories"]["levels"]:
        prim["nmi_l1"] = m["categories"]["levels"]["l1"]["nmi"]
        prim["same_l1_p_at_%d" % k] = m["links"]["same_class"]["l1"]["precision_at_k"]
    if "hyperlinks" in m["links"]:
        h = m["links"]["hyperlinks"]
        prim.update({f"hyperlink_p_at_{k}": h["precision_at_k"], f"hyperlink_r_at_{k}": h["recall_at_k"],
                     "hyperlink_map": h["map"]})
    if m["labels"].get("npmi"):
        prim["npmi"] = m["labels"]["npmi"]["mean"]
    m["primary"] = prim
    if with_ci and len(ids) >= 3:
        pred = predicted_labels(st, ids)
        g = gold.classes(lvl, ids)
        ci: dict[str, Any] = {}
        for name in ("nmi", "p1", "ari"):
            r = subsample_metric_ci(pred, g, name, n_resamples=n_resamples, seed=seed)
            ci[f"{name}_{lvl}"] = r.to_dict()
        ls = m["labels"]["per_document"]
        ci["label_sim"] = mean_ci([ls[i] for i in ids], n_resamples=n_resamples, seed=seed).to_dict()
        sc = m["links"]["same_class"][lvl]["per_document"]
        ci[f"same_{lvl}_p_at_{k}"] = mean_ci([sc[i] for i in ids], n_resamples=n_resamples, seed=seed).to_dict()
        if "hyperlinks" in m["links"]:
            pd_ = m["links"]["hyperlinks"]["per_document"]["precision_at_k"]
            if pd_:
                ci[f"hyperlink_p_at_{k}"] = mean_ci(list(pd_.values()), n_resamples=n_resamples, seed=seed).to_dict()
        m["ci"] = ci
    return m


def compare_arms(state_a: WikiState | Any, state_b: WikiState | Any, gold: Gold, *, embed_fn: Callable | None = None,
                 k: int = 5, seed: int = 0, n_resamples: int = N_RESAMPLES, deltas: Mapping[str, float] | None = None,
                 pairing: str = "all", alternative: str = "two-sided") -> dict[str, Any]:
    """E1 (NMI vs l2, 80 % subsampling), E2 (label similarity, paired BCa),
    E3 (same-l2 P@k, paired BCa) for A - B on the same gold."""
    sa, sb = coerce_state(state_a), coerce_state(state_b)
    lvl = primary_level(gold)
    deltas = dict(deltas) if deltas is not None else prereg_thresholds()["deltas"]
    ids = common_ids(sa, gold, lvl)
    if pairing == "ok_both":
        ids = [i for i in ids if sa.pages.get(i) and sa.pages[i].category is not None
               and sb.pages.get(i) and sb.pages[i].category is not None]
    elif pairing != "all":
        raise ValueError(pairing)
    g = gold.classes(lvl, ids)
    pa, pb = predicted_labels(sa, ids), predicted_labels(sb, ids)
    e1 = non_inferiority_partition(pa, pb, g, deltas.get("E1", 0.10), metric="nmi", n_resamples=n_resamples,
                                   seed=seed, alternative=alternative)
    la = label_similarity(sa, gold, level=lvl, embed_fn=embed_fn, ids=ids, seed=seed)
    lb = label_similarity(sb, gold, level=lvl, embed_fn=embed_fn, ids=ids, seed=seed)
    e2 = non_inferiority([la["per_document"][i] for i in ids], [lb["per_document"][i] for i in ids],
                         deltas.get("E2", 0.10), metric="label_sim", n_resamples=n_resamples, seed=seed,
                         alternative=alternative)
    ca = same_class_precision(sa, gold, level=lvl, k=k, ids=ids)
    cb = same_class_precision(sb, gold, level=lvl, k=k, ids=ids)
    e3 = non_inferiority([ca["per_document"][i] for i in ids], [cb["per_document"][i] for i in ids],
                         deltas.get("E3", 0.10), metric=f"same_{lvl}_p_at_{k}", n_resamples=n_resamples, seed=seed,
                         alternative=alternative)
    n_missing_a = sum(1 for i in ids if not (sa.pages.get(i) and sa.pages[i].category is not None))
    n_missing_b = sum(1 for i in ids if not (sb.pages.get(i) and sb.pages[i].category is not None))
    return {"level": lvl, "k": k, "pairing": pairing, "n_paired": len(ids), "n_uncategorised_a": n_missing_a,
            "n_uncategorised_b": n_missing_b, "arm_a": sa.arm, "arm_b": sb.arm, "deltas": deltas,
            "E1": e1, "E2": e2, "E3": e3,
            "summary": {"E1": {"estimate_a": e1.get("estimate_a"), "estimate_b": e1.get("estimate_b")},
                        "E2": {"mean_a": la["weighted_mean"], "mean_b": lb["weighted_mean"]},
                        "E3": {"mean_a": ca["precision_at_k"], "mean_b": cb["precision_at_k"], "chance": ca["chance"]}}}


def floors_from_metrics(metrics_oracle_k: Mapping[str, Any]) -> dict[str, float]:
    """The two absolute floors from an arm-A run at oracle k (N = 300)."""
    lvl = metrics_oracle_k.get("primary_level", "l2")
    cat = metrics_oracle_k["categories"]["levels"][lvl]
    return {"nmi_vs_l2_at_oracle_k": float(cat["nmi"]), "p1_at_oracle_k": float(cat["p1"])}
