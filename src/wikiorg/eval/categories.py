"""Category (partition) quality: NMI, ARI, AMI, V-measure family, purity, inverse
purity and their harmonic mean P1, with a random-partition baseline at the same k.

References
----------
* NMI: Strehl & Ghosh (2002), *Cluster ensembles*, JMLR 3.  We use scikit-learn's
  ``normalized_mutual_info_score`` with its default ``average_method="arithmetic"``
  (Vinh, Epps & Bailey 2010), which coincides with the V-measure.
* ARI: Hubert & Arabie (1985), *Comparing partitions*, J. Classification 2.
* AMI: Vinh, Epps & Bailey (2010), JMLR 11 (chance-adjusted NMI).
* Homogeneity / completeness / V-measure: Rosenberg & Hirschberg (2007), EMNLP.
* Purity and inverse purity: Zhao & Karypis (2001); Amigó et al. (2009),
  *A comparison of extrinsic clustering evaluation metrics based on formal
  constraints*, Information Retrieval 12.  P1 = harmonic mean of purity and
  inverse purity, the scale used by Pham et al. (2024), *TopicGPT*, NAACL, for
  the BERTopic/TopicGPT comparison the pre-registration cites.

Conventions
-----------
* Documents with no produced category are singleton clusters (pre-registered).
* The random baseline **permutes the produced labels** over the documents, so
  k and the cluster-size distribution (singletons included) are preserved; the
  reported value is the mean over ``n_draws`` permutations.  This shows the
  chance level of the unadjusted scores (NMI, purity, P1); ARI/AMI are
  chance-corrected and sit near zero by construction.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn import metrics as skm
from sklearn.metrics.cluster import contingency_matrix

from .state import Gold, WikiState, coerce_state, common_ids

METRIC_KEYS = ("nmi", "ari", "ami", "homogeneity", "completeness", "v_measure",
               "purity", "inverse_purity", "p1")


def _as_int_codes(labels: Sequence[Any]) -> np.ndarray:
    codes, _ = _factorise(labels)
    return codes


def _factorise(labels: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
    uniq: dict[Any, int] = {}
    codes = np.empty(len(labels), dtype=np.int64)
    for i, lab in enumerate(labels):
        codes[i] = uniq.setdefault(lab, len(uniq))
    return codes, list(uniq)


def contingency(pred: Sequence[Any], gold: Sequence[Any]) -> np.ndarray:
    """Contingency table with produced clusters as rows and gold classes as columns."""
    p = _as_int_codes(pred)
    g = _as_int_codes(gold)
    return np.asarray(contingency_matrix(g, p)).T  # rows = clusters, cols = classes


def purity(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    """Purity = (1/N) sum_k max_j n_kj: each cluster credits its majority class."""
    C = contingency(pred, gold)
    return float(C.max(axis=1).sum() / C.sum())


def inverse_purity(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    """Inverse purity = (1/N) sum_j max_k n_kj: each class credits the cluster
    that contains most of it (a recall-like counterpart of purity)."""
    C = contingency(pred, gold)
    return float(C.max(axis=0).sum() / C.sum())


def harmonic_purity(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    """P1 = harmonic mean of purity and inverse purity (Pham et al. 2024)."""
    p, ip = purity(pred, gold), inverse_purity(pred, gold)
    return 0.0 if p + ip == 0 else float(2 * p * ip / (p + ip))


def partition_metrics(pred: Sequence[Any], gold: Sequence[Any]) -> dict[str, float]:
    """All partition scores of ``pred`` against ``gold`` (aligned sequences)."""
    if len(pred) != len(gold):
        raise ValueError("pred and gold must be aligned sequences of equal length")
    if len(pred) == 0:
        return {k: float("nan") for k in METRIC_KEYS} | {"n": 0, "k_pred": 0, "k_gold": 0}
    p = _as_int_codes(pred)
    g = _as_int_codes(gold)
    h, c, v = skm.homogeneity_completeness_v_measure(g, p)
    out = {
        "nmi": float(skm.normalized_mutual_info_score(g, p)),
        "ari": float(skm.adjusted_rand_score(g, p)),
        "ami": float(skm.adjusted_mutual_info_score(g, p)),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
        "purity": purity(pred, gold),
        "inverse_purity": inverse_purity(pred, gold),
        "p1": harmonic_purity(pred, gold),
        "n": int(len(pred)),
        "k_pred": int(len(set(pred))),
        "k_gold": int(len(set(gold))),
    }
    return out


def random_partition_baseline(pred: Sequence[Any], gold: Sequence[Any], n_draws: int = 100,
                              seed: int = 0, mode: str = "permute") -> dict[str, float]:
    """Chance level of every score at the same k.

    ``mode="permute"`` (default) shuffles the produced labels across documents,
    preserving k and all cluster sizes; ``mode="uniform"`` assigns each document
    to one of k clusters uniformly at random.  Returns the mean and sd over draws
    (keys ``<metric>`` and ``<metric>_sd``).
    """
    rng = np.random.default_rng(seed)
    pred = list(pred)
    k = len(set(pred))
    acc: dict[str, list[float]] = {m: [] for m in METRIC_KEYS}
    for _ in range(n_draws):
        if mode == "permute":
            draw = [pred[i] for i in rng.permutation(len(pred))]
        elif mode == "uniform":
            draw = rng.integers(0, k, size=len(pred)).tolist()
        else:
            raise ValueError(mode)
        m = partition_metrics(draw, gold)
        for key in METRIC_KEYS:
            acc[key].append(m[key])
    out: dict[str, float] = {}
    for key, vals in acc.items():
        out[key] = float(np.mean(vals))
        out[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    out["n_draws"] = n_draws
    out["mode"] = mode  # type: ignore[assignment]
    return out


def predicted_labels(state: WikiState | Any, ids: Iterable[str]) -> list[str]:
    """Cluster id per document in ``ids`` (singletons for uncategorised docs)."""
    st = coerce_state(state)
    part = st.partition(ids, singleton_uncategorised=True)
    return [part[i] for i in ids]


def evaluate_categories(state: WikiState | Any, gold: Gold, levels: Sequence[str] | None = None,
                        ids: Sequence[str] | None = None, n_random: int = 100,
                        seed: int = 0) -> dict[str, Any]:
    """Score a WikiState against every gold level (``l1``, ``l2``; WikiCS ``class``).

    Returns ``{"k_produced", "n_uncategorised", "n_docs", "levels": {level: {...,
    "random_baseline": {...}}}}``.  The evaluation universe is the gold id set
    (documents the arm never wrote a page for become singletons).
    """
    st = coerce_state(state)
    levels = list(levels) if levels is not None else gold.levels
    out: dict[str, Any] = {
        "k_produced": st.k_produced(),
        "n_uncategorised": 0,
        "n_docs": 0,
        "levels": {},
    }
    for lvl in levels:
        lvl_ids = list(ids) if ids is not None else common_ids(st, gold, lvl)
        lvl_ids = [i for i in lvl_ids if i in gold.labels[lvl]]
        pred = predicted_labels(st, lvl_ids)
        g = gold.classes(lvl, lvl_ids)
        res = partition_metrics(pred, g)
        res["n_singletons"] = sum(1 for p in pred if p.startswith("__singleton__"))
        if n_random > 0:
            res["random_baseline"] = random_partition_baseline(pred, g, n_draws=n_random, seed=seed)
        out["levels"][lvl] = res
        out["n_docs"] = max(out["n_docs"], len(lvl_ids))
        out["n_uncategorised"] = max(out["n_uncategorised"], res["n_singletons"])
    return out


def majority_class(members: Iterable[str], gold_labels: dict[str, str]) -> tuple[str | None, int, int]:
    """(majority class, its count, number of labelled members) for a cluster."""
    cnt = Counter(gold_labels[m] for m in members if m in gold_labels)
    if not cnt:
        return None, 0, 0
    cls, n = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
    return cls, n, sum(cnt.values())


def per_document_metric_values(pred: Sequence[Any], gold: Sequence[Any]) -> dict[str, np.ndarray]:
    """Per-document contributions that average to purity / inverse purity.

    purity = mean_i [gold_i == majority class of cluster(i)]; inverse purity =
    mean_i [cluster(i) == the cluster holding most of class(gold_i)].  These are
    the per-document units for paired resampling of purity-type scores.
    """
    pred = list(pred)
    gold = list(gold)
    C = contingency(pred, gold)
    pc, pl = _factorise(pred)
    gc, gl = _factorise(gold)
    maj_of_cluster = C.argmax(axis=1)
    best_cluster_of_class = C.argmax(axis=0)
    pur = np.array([1.0 if maj_of_cluster[pc[i]] == gc[i] else 0.0 for i in range(len(pred))])
    ipur = np.array([1.0 if best_cluster_of_class[gc[i]] == pc[i] else 0.0 for i in range(len(pred))])
    return {"purity": pur, "inverse_purity": ipur}
