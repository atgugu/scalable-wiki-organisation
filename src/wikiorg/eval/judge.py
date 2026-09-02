"""LLM-as-judge agreement with the human raters (secondary, never a falsifier).

This module makes **no LLM calls**.  The harness (workstream W2) asks the judge
model (Sonnet, same rubric as ``human.INSTRUCTIONS``, blinded items) to rate
the same 45 items and stores a CSV with columns ``item_id, rating`` (optional
``rater`` / ``model`` / ``run``).  ``judge_agreement`` then reports the judge's
per-arm means, its Spearman correlation with the human item means, Cliff's
delta per arm pair on the judge scores, and Krippendorff's alpha with the judge
added as an extra rater (and, separately, judge vs. the human consensus).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bootstrap import cliffs_delta
from .human import krippendorff_alpha, read_csv, reliability_matrix


def read_judge_csv(path: str | Path, rater: str = "judge") -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        raw = str(row.get("rating", row.get("score", "")) or "").strip()
        try:
            val = float(raw)
        except ValueError:
            continue
        rows.append({"rater": str(row.get("rater") or rater), "item_id": row["item_id"], "rating": val,
                     "model": row.get("model", ""), "run": row.get("run", "")})
    return rows


def judge_agreement(judge: Sequence[Mapping[str, Any]], human: Sequence[Mapping[str, Any]],
                    key: Mapping[str, Mapping[str, Any]], exclude_raters: Sequence[str] = ()) -> dict[str, Any]:
    from scipy import stats

    human = [r for r in human if r["rater"] not in set(exclude_raters)]
    Mh, hr, items = reliability_matrix(human)
    human_mean = {it: float(np.nanmean(Mh[:, i])) for i, it in enumerate(items) if not np.isnan(Mh[:, i]).all()}
    judge_runs: dict[str, dict[str, float]] = defaultdict(dict)
    for r in judge:
        judge_runs[str(r["rater"])][r["item_id"]] = float(r["rating"])
    judge_mean: dict[str, float] = {}
    for it in {i for run in judge_runs.values() for i in run}:
        vals = [run[it] for run in judge_runs.values() if it in run]
        judge_mean[it] = float(np.mean(vals))
    common = sorted(set(human_mean) & set(judge_mean))
    out: dict[str, Any] = {"n_common_items": len(common), "n_judge_runs": len(judge_runs),
                           "excluded_raters": list(exclude_raters)}
    if len(common) >= 3:
        h = np.array([human_mean[i] for i in common])
        j = np.array([judge_mean[i] for i in common])
        out["spearman_judge_vs_human_mean"] = float(stats.spearmanr(h, j).statistic)
        out["pearson_judge_vs_human_mean"] = float(stats.pearsonr(h, j).statistic)
        out["mean_abs_diff"] = float(np.mean(np.abs(h - j)))
        out["judge_minus_human_mean"] = float(np.mean(j - h))
        M2 = np.vstack([np.array([human_mean[i] for i in common]), j])
        out["alpha_judge_vs_human_consensus"] = krippendorff_alpha(M2, "interval")
        judge_rows = [{"rater": "judge", "item_id": it, "rating": judge_mean[it]} for it in common]
        Mall, _, _ = reliability_matrix(list(human) + judge_rows, item_ids=common)
        out["alpha_humans_plus_judge"] = krippendorff_alpha(Mall, "interval")
        out["alpha_humans_only"] = krippendorff_alpha(Mh[:, [items.index(i) for i in common]], "interval")
    by_arm: dict[str, list[float]] = defaultdict(list)
    by_arm_h: dict[str, list[float]] = defaultdict(list)
    for it, v in judge_mean.items():
        arm = key.get(it, {}).get("arm")
        if arm is not None:
            by_arm[str(arm)].append(v)
            if it in human_mean:
                by_arm_h[str(arm)].append(human_mean[it])
    out["arms"] = {a: {"judge_mean": float(np.mean(v)), "n_items": len(v),
                       "human_mean": float(np.mean(by_arm_h[a])) if by_arm_h.get(a) else float("nan")}
                   for a, v in sorted(by_arm.items())}
    arms = sorted(by_arm)
    out["cliffs_delta_judge"] = [{"a": a, "b": b, "cliffs_delta": cliffs_delta(by_arm[a], by_arm[b])}
                                 for i, a in enumerate(arms) for b in arms[i + 1:]]
    return out
