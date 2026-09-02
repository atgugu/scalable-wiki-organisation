"""Human label-rating study: sheet construction and scoring.

Design (pre-registered, secondary/descriptive): 45 items = 10 gold ``l2``
classes x arms A / A+ / B / C (the produced category whose majority class is
that gold class: its label and three member titles) + 5 *gold-name controls*
(the de-CamelCased gold class name with three member titles).  Three blinded
raters (the student and two peers) score each label on a 5-point anchored
scale; arms are hidden behind random letter codes and every rater gets their
own random item order.  Raters cannot be blind to arm A's term-list format,
and the student rates their own system, so every statistic is reported twice:
with all raters and **without the student's ratings**.

Scoring: per-arm mean of item means with BCa bootstrap CIs over items;
Krippendorff's alpha (Krippendorff 2004; ``krippendorff`` package, interval
level) and pairwise Spearman correlations between raters; Cliff's delta
between arms on item means.  LLM-judge agreement lives in ``judge.py``.
"""
from __future__ import annotations

import csv
import io
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bootstrap import cliffs_delta, mean_ci
from .categories import majority_class
from .labels import de_camel
from .state import Gold, WikiState, coerce_state, common_ids

ARMS = ("A", "A+", "B", "C")
CONTROL_ARM = "GOLD"
SCALE: dict[int, str] = {
    1: "wrong or meaningless: does not describe these articles",
    2: "loosely related: touches the theme but a reader would not group the articles under it",
    3: "partly right: describes some of the articles or is too broad / too narrow",
    4: "good: a reader would accept it as the category of these articles",
    5: "as good as an encyclopedia category name for these articles",
}
INSTRUCTIONS = (
    "You will see {n} category labels. Each label was proposed for a group of encyclopedia articles; "
    "three article titles from the group are shown. Rate how well the label names the group as an "
    "encyclopedia category, on a 1-5 scale:\n" +
    "\n".join(f"  {k}: {v}" for k, v in SCALE.items()) +
    "\nRate the label only (not the articles). Some labels are lists of keywords; rate whether the "
    "keywords name the group. Write the number in the 'rating' column. Do not discuss items with other raters."
)


def select_classes(gold: Gold, level: str = "l2", n_classes: int = 10, seed: int = 0,
                   strategy: str = "largest", min_members: int = 3, ids: Sequence[str] | None = None) -> list[str]:
    counts = gold.class_counts(level, ids)
    eligible = [c for c, n in counts.items() if n >= min_members]
    if strategy == "largest":
        chosen = sorted(eligible, key=lambda c: (-counts[c], c))[:n_classes]
    elif strategy == "random":
        rng = random.Random(seed)
        chosen = sorted(eligible)
        rng.shuffle(chosen)
        chosen = chosen[:n_classes]
    else:
        raise ValueError(strategy)
    return chosen


def _cluster_for_class(state: WikiState, gold_labels: Mapping[str, str], cls: str, ids: set[str]) -> tuple[str | None, list[str], bool]:
    """Produced category best representing ``cls``: majority-class match with
    the most members of the class; else the category holding most of the class."""
    best_major, best_any = None, None
    for cid, members in state.members().items():
        members = [m for m in members if m in ids]
        if not members:
            continue
        maj, n_maj, _ = majority_class(members, gold_labels)
        n_cls = sum(1 for m in members if gold_labels.get(m) == cls)
        if n_cls == 0:
            continue
        if maj == cls and (best_major is None or (n_cls, len(members)) > best_major[1:]):
            best_major = (cid, n_cls, len(members))
        if best_any is None or (n_cls, -len(members)) > best_any[1:]:
            best_any = (cid, n_cls, -len(members))
    if best_major is not None:
        return best_major[0], [m for m in state.members()[best_major[0]] if m in ids], True
    if best_any is not None:
        return best_any[0], [m for m in state.members()[best_any[0]] if m in ids], False
    return None, [], False


def build_items(states: Mapping[str, Any], gold: Gold, level: str = "l2", n_classes: int = 10, n_controls: int = 5,
                titles_per_item: int = 3, seed: int = 0, class_strategy: str = "largest",
                arms: Sequence[str] = ARMS) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return ``(items, arm_codes)``.  Each item carries the public fields
    (``item_id``, ``label``, ``title_1..3``) and the private key fields
    (``arm``, ``arm_code``, ``gold_class``, ``cid``, ``majority_matches``)."""
    rng = random.Random(seed)
    sts = {arm: coerce_state(states[arm]) for arm in arms if arm in states}
    ids = set(common_ids(sts[next(iter(sts))], gold, level)) if sts else set(gold.ids())
    gl = gold.labels[level]
    classes = select_classes(gold, level, n_classes, seed, class_strategy, ids=sorted(ids))
    letters = list("KQRVZXHMJT")
    rng.shuffle(letters)
    arm_codes = {arm: letters[i] for i, arm in enumerate(list(sts) + [CONTROL_ARM])}
    items: list[dict[str, Any]] = []

    def pick_titles(members: Sequence[str], cls: str) -> list[str]:
        in_cls = [m for m in members if gl.get(m) == cls]
        pool = in_cls if len(in_cls) >= titles_per_item else list(members)
        pool = sorted(pool)
        rng.shuffle(pool)
        chosen = pool[:titles_per_item]
        return [gold.titles.get(m) or sts_title(m) for m in chosen]

    def sts_title(m: str) -> str:
        for st in sts.values():
            t = st.title_of(m)
            if t:
                return t
        return m

    for cls in classes:
        for arm, st in sts.items():
            cid, members, matches = _cluster_for_class(st, gl, cls, ids)
            label = st.categories[cid].label if cid in st.categories else ""
            if not label and cid in st.categories and st.categories[cid].terms:
                label = ", ".join(st.categories[cid].terms[:3])
            titles = pick_titles(members, cls) if members else []
            titles += [""] * (titles_per_item - len(titles))
            items.append({"arm": arm, "arm_code": arm_codes[arm], "gold_class": cls, "cid": cid,
                          "majority_matches": matches, "label": label,
                          **{f"title_{i + 1}": t for i, t in enumerate(titles)}})
    control_classes = list(classes)
    rng.shuffle(control_classes)
    for cls in control_classes[:n_controls]:
        members = [m for m in sorted(ids) if gl.get(m) == cls]
        titles = pick_titles(members, cls)
        titles += [""] * (titles_per_item - len(titles))
        items.append({"arm": CONTROL_ARM, "arm_code": arm_codes[CONTROL_ARM], "gold_class": cls, "cid": None,
                      "majority_matches": True, "label": de_camel(cls),
                      **{f"title_{i + 1}": t for i, t in enumerate(titles)}})
    rng.shuffle(items)
    for n, it in enumerate(items, start=1):
        it["item_id"] = f"i{n:02d}"
    return items, arm_codes


PUBLIC_COLUMNS = ("rater", "position", "item_id", "label", "title_1", "title_2", "title_3", "rating")
KEY_COLUMNS = ("item_id", "arm", "arm_code", "gold_class", "cid", "majority_matches", "label")


def rating_sheets(items: Sequence[Mapping[str, Any]], raters: Sequence[str], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    """Per-rater rows in a rater-specific random order, rating column blank."""
    sheets: dict[str, list[dict[str, Any]]] = {}
    for r_i, rater in enumerate(raters):
        rng = random.Random(f"{seed}:{rater}:{r_i}")
        order = list(items)
        rng.shuffle(order)
        sheets[rater] = [{"rater": rater, "position": pos + 1, "item_id": it["item_id"], "label": it["label"],
                          "title_1": it.get("title_1", ""), "title_2": it.get("title_2", ""),
                          "title_3": it.get("title_3", ""), "rating": ""} for pos, it in enumerate(order)]
    return sheets


def write_csv(rows: Iterable[Mapping[str, Any]], path: str | Path, columns: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
    return path


def write_study(items: Sequence[Mapping[str, Any]], arm_codes: Mapping[str, str], raters: Sequence[str],
                out_dir: str | Path, seed: int = 0) -> dict[str, Path]:
    """Write ``sheets/<rater>.csv`` (blinded), ``key.csv`` (private) and
    ``instructions.txt``; returns the paths."""
    out_dir = Path(out_dir)
    paths: dict[str, Path] = {}
    for rater, rows in rating_sheets(items, raters, seed).items():
        paths[f"sheet:{rater}"] = write_csv(rows, out_dir / "sheets" / f"{rater}.csv", PUBLIC_COLUMNS)
    paths["key"] = write_csv(items, out_dir / "key.csv", KEY_COLUMNS)
    codes_path = out_dir / "arm_codes.csv"
    write_csv([{"arm": a, "arm_code": c} for a, c in arm_codes.items()], codes_path, ("arm", "arm_code"))
    paths["arm_codes"] = codes_path
    ins = out_dir / "instructions.txt"
    ins.write_text(INSTRUCTIONS.format(n=len(items)) + "\n", encoding="utf-8")
    paths["instructions"] = ins
    return paths


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_ratings(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Returned sheets -> ``[{rater, item_id, rating}]`` (blank/invalid ratings dropped)."""
    out = []
    for p in paths:
        for row in read_csv(p):
            raw = str(row.get("rating", "") or "").strip()
            try:
                val = float(raw)
            except ValueError:
                continue
            if not 1 <= val <= 5:
                continue
            out.append({"rater": str(row.get("rater") or Path(p).stem), "item_id": row["item_id"], "rating": val})
    return out


def read_key(path: str | Path) -> dict[str, dict[str, Any]]:
    key = {}
    for row in read_csv(path):
        row = dict(row)
        row["majority_matches"] = str(row.get("majority_matches", "True")).lower() in ("true", "1", "yes")
        key[row["item_id"]] = row
    return key


def reliability_matrix(ratings: Sequence[Mapping[str, Any]], raters: Sequence[str] | None = None,
                       item_ids: Sequence[str] | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    raters = list(raters) if raters is not None else sorted({r["rater"] for r in ratings})
    item_ids = list(item_ids) if item_ids is not None else sorted({r["item_id"] for r in ratings})
    ri = {r: i for i, r in enumerate(raters)}
    ii = {it: i for i, it in enumerate(item_ids)}
    M = np.full((len(raters), len(item_ids)), np.nan)
    for r in ratings:
        if r["rater"] in ri and r["item_id"] in ii:
            M[ri[r["rater"]], ii[r["item_id"]]] = float(r["rating"])
    return M, raters, item_ids


def krippendorff_alpha(M: np.ndarray, level: str = "interval") -> float:
    if M.shape[0] < 2 or np.isnan(M).all():
        return float("nan")
    try:
        import krippendorff  # type: ignore
        return float(krippendorff.alpha(reliability_data=M, level_of_measurement=level))
    except Exception:
        return float("nan")


def pairwise_spearman(M: np.ndarray, raters: Sequence[str]) -> list[dict[str, Any]]:
    from scipy import stats
    out = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            mask = ~np.isnan(M[i]) & ~np.isnan(M[j])
            if mask.sum() < 3:
                rho, n = float("nan"), int(mask.sum())
            else:
                rho, n = float(stats.spearmanr(M[i, mask], M[j, mask]).statistic), int(mask.sum())
            out.append({"a": raters[i], "b": raters[j], "spearman": rho, "n": n})
    return out


def score(ratings: Sequence[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]], exclude_raters: Sequence[str] = (),
          n_resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per-arm means with BCa CIs over items, Krippendorff's alpha, pairwise
    Spearman, Cliff's delta between arms (on item means)."""
    ratings = [r for r in ratings if r["rater"] not in set(exclude_raters)]
    M, raters, item_ids = reliability_matrix(ratings)
    item_mean = {it: float(np.nanmean(M[:, i])) for i, it in enumerate(item_ids) if not np.isnan(M[:, i]).all()}
    by_arm: dict[str, list[float]] = defaultdict(list)
    for it, val in item_mean.items():
        arm = key.get(it, {}).get("arm")
        if arm is not None:
            by_arm[str(arm)].append(val)
    arms_out: dict[str, Any] = {}
    for arm, vals in sorted(by_arm.items()):
        ci = mean_ci(vals, n_resamples=n_resamples, seed=seed)
        arms_out[arm] = {"mean": float(np.mean(vals)), "ci_low": ci.low, "ci_high": ci.high, "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                         "n_items": len(vals), "ci_method": ci.method}
    deltas = []
    arms_sorted = sorted(by_arm)
    for i in range(len(arms_sorted)):
        for j in range(i + 1, len(arms_sorted)):
            a, b = arms_sorted[i], arms_sorted[j]
            deltas.append({"a": a, "b": b, "cliffs_delta": cliffs_delta(by_arm[a], by_arm[b]),
                           "mean_diff": float(np.mean(by_arm[a]) - np.mean(by_arm[b]))})
    per_rater = {r: {"mean": float(np.nanmean(M[i])), "n": int((~np.isnan(M[i])).sum())} for i, r in enumerate(raters)}
    return {"n_raters": len(raters), "raters": raters, "n_items": len(item_ids), "n_ratings": len(ratings),
            "excluded_raters": list(exclude_raters), "arms": arms_out, "cliffs_delta": deltas,
            "krippendorff_alpha_interval": krippendorff_alpha(M, "interval"),
            "krippendorff_alpha_ordinal": krippendorff_alpha(M, "ordinal"),
            "spearman": pairwise_spearman(M, raters), "per_rater": per_rater, "item_means": item_mean}


def score_with_and_without(ratings: Sequence[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]],
                           student_rater: str | None, **kw) -> dict[str, Any]:
    """The pre-registered pair of reports: all raters, and without the student."""
    out = {"all_raters": score(ratings, key, **kw)}
    if student_rater is not None:
        out["without_student"] = score(ratings, key, exclude_raters=(student_rater,), **kw)
        out["student_rater"] = student_rater
    return out


def human_table_rows(scored: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten ``score_with_and_without`` output into T5 rows."""
    rows = []
    for variant in ("all_raters", "without_student"):
        s = scored.get(variant)
        if not s:
            continue
        for arm, v in s["arms"].items():
            rows.append({"variant": variant, "arm": arm, "mean": v["mean"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                         "sd": v["sd"], "n_items": v["n_items"], "n_raters": s["n_raters"],
                         "krippendorff_alpha": s["krippendorff_alpha_interval"],
                         "spearman_mean": float(np.nanmean([p["spearman"] for p in s["spearman"]])) if s["spearman"] else float("nan")})
    return rows
