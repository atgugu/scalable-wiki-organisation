"""Uncertainty, effect sizes, multiplicity and the pre-registered decision rule.

Confidence intervals
--------------------
* Per-document metrics (P@5, same-class precision, per-document label
  similarity, purity contributions): BCa bootstrap (Efron 1987; Efron &
  Tibshirani 1993; DiCiccio & Efron 1996) via ``scipy.stats.bootstrap`` with
  2000 resamples, **paired over documents** (the same resampled document set is
  scored for both arms, so the CI of A - B accounts for the shared documents).
* Partition metrics (NMI / ARI / P1): a bootstrap resample duplicates
  documents, and a duplicated document is trivially "clustered together" with
  its copy, which inflates agreement scores.  We therefore draw **80 % of the
  documents without replacement** (2000 draws) and score both arms on the same
  subset.  The primary interval is the subsampling interval of Politis, Romano &
  Wolf (1999, *Subsampling*, Thm. 2.2.1): quantiles of sqrt(b) (theta_b -
  theta_n) rescaled by sqrt(b/n) around the full-sample statistic; the raw
  percentile interval of the subsample distribution (wider by about
  1/sqrt(0.8)) is reported alongside as ``ci_low_raw`` / ``ci_high_raw``.
* A degenerate BCa (constant bootstrap distribution, e.g. every document
  scored identically) falls back to the percentile interval and is flagged.

Non-inferiority (``configs/prereg.yaml``)
---------------------------------------
``non_inferior`` iff the lower bound of the 95 % CI of (A - B) exceeds
-delta.  The CI is the customary *two-sided* 95 % interval, so the one-sided
test runs at alpha = 0.025 (ICH E9; the standard reading of "lower limit of
the 95 % CI" in non-inferiority trials); ``alternative="greater"`` gives the
one-sided 95 % reading instead and the result records which was used.
``falsified`` iff the *upper* bound is below -delta.  The one-sided p-value for
H0: A - B <= -delta is obtained by inverting the BCa interval (the confidence
level at which the lower bound equals -delta; for subsampling, the fraction of
the pivotal distribution at or below -delta) and the three primary endpoints
are corrected with Holm (1979) step-down at the family-wise one-sided alpha.

Effect sizes: absolute difference with CI, Cohen's d (1988; pooled sd) and
d_z (paired), Cliff's delta (1993).

``decide`` applies the full rule: supported iff E1-E3 all non-inferior after
Holm AND cost ratio USD(B)/USD(A) >= 20 AND the absolute floors (NMI vs l2 >=
0.55 and P1 >= 0.60 at oracle k, N = 300) hold; falsified iff any endpoint's
upper bound < -delta or a floor fails; otherwise inconclusive.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri
from sklearn import metrics as skm

from .categories import harmonic_purity

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREREG = REPO_ROOT / "configs" / "prereg.yaml"
N_RESAMPLES = 2000
SUBSAMPLE_FRAC = 0.8

PARTITION_METRICS: dict[str, Callable[[Sequence[Any], Sequence[Any]], float]] = {
    "nmi": lambda pred, gold: float(skm.normalized_mutual_info_score(list(gold), list(pred))),
    "ari": lambda pred, gold: float(skm.adjusted_rand_score(list(gold), list(pred))),
    "ami": lambda pred, gold: float(skm.adjusted_mutual_info_score(list(gold), list(pred))),
    "p1": lambda pred, gold: harmonic_purity(list(pred), list(gold)),
}


@dataclass
class CIResult:
    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n: int
    n_resamples: int
    seed: int
    alternative: str = "two-sided"
    degenerate: bool = False
    dist: np.ndarray | None = field(default=None, repr=False)
    z0: float | None = None
    accel: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"estimate": self.estimate, "ci_low": self.low, "ci_high": self.high, "confidence": self.confidence,
             "method": self.method, "n": self.n, "n_resamples": self.n_resamples, "seed": self.seed,
             "alternative": self.alternative, "degenerate": self.degenerate}
        d.update(self.extra)
        return d


def _mean_stat(*samples, axis=-1):
    return np.mean(samples[0], axis=axis)


def _mean_diff_stat(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def _jackknife_accel(samples: Sequence[np.ndarray], statistic: Callable) -> float:
    """BCa acceleration from the leave-one-out jackknife (same formula as scipy)."""
    n = len(samples[0])
    if n < 3:
        return 0.0
    theta = np.empty(n)
    for i in range(n):
        keep = np.r_[0:i, i + 1:n]
        theta[i] = float(statistic(*[s[keep] for s in samples], axis=-1))
    dot = theta.mean()
    num = ((dot - theta) ** 3).sum()
    den = ((dot - theta) ** 2).sum()
    return float(num / (6 * den ** 1.5)) if den > 0 else 0.0


def bootstrap_ci(samples: Sequence[np.ndarray], statistic: Callable, n_resamples: int = N_RESAMPLES,
                 seed: int = 0, confidence: float = 0.95, alternative: str = "two-sided",
                 paired: bool = True) -> CIResult:
    """BCa CI of ``statistic(*samples)`` with scipy; percentile fallback when degenerate."""
    samples = [np.asarray(s, dtype=float) for s in samples]
    n = len(samples[0])
    est = float(statistic(*samples, axis=-1))
    if n < 2 or all(np.ptp(s) == 0 for s in samples):
        return CIResult(est, est, est, confidence, "constant", n, n_resamples, seed, alternative, degenerate=True,
                        dist=np.full(n_resamples, est), z0=0.0, accel=0.0)
    rng = np.random.default_rng(seed)
    method = "BCa"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = stats.bootstrap(tuple(samples), statistic, paired=paired, vectorized=True, n_resamples=n_resamples,
                                  confidence_level=confidence, alternative=alternative, method="BCa", rng=rng)
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            dist = np.asarray(res.bootstrap_distribution, dtype=float)
        except Exception:
            lo = hi = float("nan")
            dist = None
        finite = (np.isfinite(lo) or alternative == "less") and (np.isfinite(hi) or alternative == "greater")
        degenerate = not finite
        if degenerate:
            res = stats.bootstrap(tuple(samples), statistic, paired=paired, vectorized=True, n_resamples=n_resamples,
                                  confidence_level=confidence, alternative=alternative, method="percentile",
                                  rng=np.random.default_rng(seed))
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            dist = np.asarray(res.bootstrap_distribution, dtype=float)
            method = "percentile (BCa degenerate)"
    frac_below = float(np.mean(dist < est)) if dist is not None else 0.5
    z0 = float(ndtri(min(max(frac_below, 1e-12), 1 - 1e-12)))
    accel = _jackknife_accel(samples, statistic) if not degenerate else 0.0
    return CIResult(est, lo, hi, confidence, method, n, n_resamples, seed, alternative, degenerate,
                    dist=dist, z0=z0, accel=accel)


def mean_ci(values: Sequence[float], **kw) -> CIResult:
    return bootstrap_ci([np.asarray(values, dtype=float)], _mean_stat, **kw)


def paired_mean_diff_ci(a: Sequence[float], b: Sequence[float], **kw) -> CIResult:
    """CI of mean(a) - mean(b), paired over documents (ordinary resampling)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci([a, b], _mean_diff_stat, paired=True, **kw)


def bca_bound(res: CIResult, alpha: float, side: str = "lower") -> float:
    """BCa quantile of the stored bootstrap distribution at one-sided level alpha."""
    if res.dist is None:
        return float("nan")
    if res.method.startswith("constant"):
        return res.estimate
    z0, a = res.z0 or 0.0, res.accel or 0.0
    if res.method.startswith("percentile"):
        q = alpha if side == "lower" else 1 - alpha
        return float(np.percentile(res.dist, 100 * q))
    z = ndtri(alpha) if side == "lower" else ndtri(1 - alpha)
    num = z0 + z
    q = float(ndtr(z0 + num / (1 - a * num)))
    return float(np.percentile(res.dist, 100 * min(max(q, 0.0), 1.0)))


def pvalue_greater(res: CIResult, threshold: float) -> float:
    """One-sided p-value of H0: theta <= threshold, by inverting the interval.

    For BCa this is the level alpha at which the one-sided (1 - alpha) lower
    bound equals ``threshold``; for percentile/pivotal distributions it is the
    fraction of the distribution at or below ``threshold``.
    """
    if res.dist is None:
        return float("nan")
    if res.method.startswith("constant"):
        return 0.0 if res.estimate > threshold else 1.0
    q = float(np.mean(res.dist <= threshold))
    if res.method.startswith("percentile") or res.method.startswith("subsampling"):
        return q
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    z0, a = res.z0 or 0.0, res.accel or 0.0
    w = float(ndtri(q)) - z0
    denom = 1 + a * w
    if denom <= 0:
        return 0.0 if w < 0 else 1.0
    return float(ndtr(w / denom - z0))


# --------------------------------------------------------------------------
# Subsampling for partition metrics
# --------------------------------------------------------------------------
def subsample_diff_ci(pred_a: Sequence[Any], pred_b: Sequence[Any], gold: Sequence[Any],
                      metric: str | Callable[[Sequence[Any], Sequence[Any]], float] = "nmi",
                      frac: float = SUBSAMPLE_FRAC, n_resamples: int = N_RESAMPLES, seed: int = 0,
                      confidence: float = 0.95, alternative: str = "two-sided") -> CIResult:
    """CI of metric(A) - metric(B) by paired 80 % subsampling without replacement."""
    fn = PARTITION_METRICS[metric] if isinstance(metric, str) else metric
    pa, pb, g = list(pred_a), list(pred_b), list(gold)
    n = len(g)
    if not (len(pa) == len(pb) == n):
        raise ValueError("pred_a, pred_b and gold must be aligned")
    b = max(2, int(round(frac * n)))
    theta_n = fn(pa, g) - fn(pb, g)
    rng = np.random.default_rng(seed)
    thetas = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.choice(n, size=b, replace=False)
        sa = [pa[i] for i in idx]
        sb = [pb[i] for i in idx]
        sg = [g[i] for i in idx]
        thetas[r] = fn(sa, sg) - fn(sb, sg)
    scale = math.sqrt(b / n)
    pivot = theta_n - scale * (thetas - theta_n)  # Politis-Romano-Wolf rescaled, basic form
    alpha = 1 - confidence
    if alternative == "two-sided":
        lo, hi = np.quantile(pivot, alpha / 2), np.quantile(pivot, 1 - alpha / 2)
        lo_raw, hi_raw = np.quantile(thetas, alpha / 2), np.quantile(thetas, 1 - alpha / 2)
    elif alternative == "greater":
        lo, hi = np.quantile(pivot, alpha), float("inf")
        lo_raw, hi_raw = np.quantile(thetas, alpha), float("inf")
    elif alternative == "less":
        lo, hi = float("-inf"), np.quantile(pivot, 1 - alpha)
        lo_raw, hi_raw = float("-inf"), np.quantile(thetas, 1 - alpha)
    else:
        raise ValueError(alternative)
    degenerate = bool(np.ptp(thetas) == 0)
    return CIResult(float(theta_n), float(lo), float(hi), confidence,
                    "subsampling 80% w/o replacement (Politis-Romano-Wolf rescaled)", n, n_resamples, seed,
                    alternative, degenerate, dist=pivot, z0=0.0, accel=0.0,
                    extra={"ci_low_raw": float(lo_raw), "ci_high_raw": float(hi_raw), "subsample_size": b,
                           "subsample_frac": frac, "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
                           "estimate_a": fn(pa, g), "estimate_b": fn(pb, g)})


def subsample_metric_ci(pred: Sequence[Any], gold: Sequence[Any], metric: str | Callable = "nmi",
                        frac: float = SUBSAMPLE_FRAC, n_resamples: int = N_RESAMPLES, seed: int = 0,
                        confidence: float = 0.95) -> CIResult:
    """CI of a single partition metric by 80 % subsampling (for floors and tables)."""
    fn = PARTITION_METRICS[metric] if isinstance(metric, str) else metric
    p, g = list(pred), list(gold)
    n = len(g)
    b = max(2, int(round(frac * n)))
    theta_n = fn(p, g)
    rng = np.random.default_rng(seed)
    thetas = np.array([fn([p[i] for i in idx], [g[i] for i in idx])
                       for idx in (rng.choice(n, size=b, replace=False) for _ in range(n_resamples))])
    pivot = theta_n - math.sqrt(b / n) * (thetas - theta_n)
    alpha = 1 - confidence
    return CIResult(float(theta_n), float(np.quantile(pivot, alpha / 2)), float(np.quantile(pivot, 1 - alpha / 2)),
                    confidence, "subsampling 80% w/o replacement (Politis-Romano-Wolf rescaled)", n, n_resamples, seed,
                    "two-sided", bool(np.ptp(thetas) == 0), dist=pivot, z0=0.0, accel=0.0,
                    extra={"ci_low_raw": float(np.quantile(thetas, alpha / 2)),
                           "ci_high_raw": float(np.quantile(thetas, 1 - alpha / 2)), "subsample_size": b})


# --------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------
def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    s2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return float((a.mean() - b.mean()) / math.sqrt(s2)) if s2 > 0 else (0.0 if a.mean() == b.mean() else float("inf") * np.sign(a.mean() - b.mean()))


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired standardised mean difference: mean(a - b) / sd(a - b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return float("nan")
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else (0.0 if d.mean() == 0 else float("inf") * np.sign(d.mean()))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta = P(a > b) - P(a < b) over all pairs (Cliff 1993)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = lt = 0
    step = 4096
    for i in range(0, len(a), step):
        chunk = a[i:i + step, None]
        gt += int((chunk > b[None, :]).sum())
        lt += int((chunk < b[None, :]).sum())
    return float((gt - lt) / (len(a) * len(b)))


def effect_sizes(a: Sequence[float], b: Sequence[float], paired: bool = True) -> dict[str, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = {"mean_a": float(a.mean()) if len(a) else float("nan"), "mean_b": float(b.mean()) if len(b) else float("nan"),
           "mean_diff": float(a.mean() - b.mean()) if len(a) and len(b) else float("nan"),
           "cohen_d": cohens_d(a, b), "cliffs_delta": cliffs_delta(a, b)}
    if paired and len(a) == len(b):
        out["cohen_dz"] = cohens_dz(a, b)
    return out


# --------------------------------------------------------------------------
# Multiplicity and non-inferiority
# --------------------------------------------------------------------------
def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def non_inferiority_from_ci(res: CIResult, delta: float, metric: str = "", alpha: float | None = None) -> dict[str, Any]:
    """Apply the pre-registered rule to a CI of (A - B)."""
    d = res.to_dict()
    p = pvalue_greater(res, -delta)
    d.update({"metric": metric, "delta": delta, "margin": -delta,
              "non_inferior": bool(res.low > -delta),
              "falsified": bool(res.high < -delta),
              "p_one_sided": p,
              "ci_reading": ("lower bound of the two-sided 95% CI (one-sided alpha 0.025)" if res.alternative == "two-sided"
                             else f"one-sided {res.alternative} {int(res.confidence * 100)}% bound")})
    if alpha is not None:
        d["alpha_one_sided"] = alpha
        d["non_inferior_at_alpha"] = bool(p < alpha)
    d["_ci"] = res
    return d


def non_inferiority(a: Sequence[float], b: Sequence[float], delta: float, metric: str = "", n_resamples: int = N_RESAMPLES,
                    seed: int = 0, confidence: float = 0.95, alternative: str = "two-sided") -> dict[str, Any]:
    """Per-document endpoint: paired BCa CI of mean(a) - mean(b); non-inferior
    iff the lower bound exceeds -delta (``configs/prereg.yaml``)."""
    res = paired_mean_diff_ci(a, b, n_resamples=n_resamples, seed=seed, confidence=confidence, alternative=alternative)
    out = non_inferiority_from_ci(res, delta, metric)
    out["effect"] = effect_sizes(a, b, paired=True)
    out["n_paired"] = len(a)
    return out


def non_inferiority_partition(pred_a: Sequence[Any], pred_b: Sequence[Any], gold: Sequence[Any], delta: float,
                              metric: str = "nmi", n_resamples: int = N_RESAMPLES, seed: int = 0,
                              confidence: float = 0.95, alternative: str = "two-sided", frac: float = SUBSAMPLE_FRAC) -> dict[str, Any]:
    """Partition endpoint (E1): 80 % subsampling CI of metric(A) - metric(B)."""
    res = subsample_diff_ci(pred_a, pred_b, gold, metric=metric, frac=frac, n_resamples=n_resamples, seed=seed,
                            confidence=confidence, alternative=alternative)
    out = non_inferiority_from_ci(res, delta, metric)
    out["n_paired"] = len(gold)
    return out


def holm_non_inferiority(endpoints: Mapping[str, Mapping[str, Any]], alpha: float = 0.025) -> dict[str, dict[str, Any]]:
    """Holm step-down over the endpoints' one-sided p-values at family-wise alpha.

    Adds ``p_holm``, ``non_inferior_holm`` and ``ci_low_holm`` (the lower bound at
    the Holm-adjusted per-endpoint level, computed from the stored bootstrap
    distribution) to a copy of each endpoint dict.
    """
    names = list(endpoints)
    pvals = [float(endpoints[n]["p_one_sided"]) for n in names]
    adj = holm(pvals)
    m = len(names)
    order = sorted(range(m), key=lambda i: pvals[i])
    out: dict[str, dict[str, Any]] = {}
    stopped = False
    for rank, i in enumerate(order):
        name = names[i]
        e = dict(endpoints[name])
        level = alpha / (m - rank)
        e["p_holm"] = adj[i]
        e["holm_level"] = level
        passed = (not stopped) and pvals[i] < level
        if not passed:
            stopped = True
        e["non_inferior_holm"] = bool(passed)
        res = endpoints[name].get("_ci")
        e["ci_low_holm"] = bca_bound(res, level, "lower") if isinstance(res, CIResult) else float("nan")
        out[name] = e
    return {n: out[n] for n in names}


def load_prereg(path: str | Path | None = None) -> dict[str, Any]:
    import yaml  # type: ignore
    with open(Path(path) if path else DEFAULT_PREREG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prereg_thresholds(prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Deltas, cost ratio, floors and alpha as frozen in ``configs/prereg.yaml``."""
    prereg = prereg or load_prereg()
    pe = prereg.get("primary_endpoints", {})
    deltas = {k: float(v["delta"]) for k, v in pe.items() if isinstance(v, Mapping) and "delta" in v}
    # Prefer the explicit structured keys (added 2026-09-02 to remove ambiguity in the free-text
    # ``test`` field, where a regex scan can match the wrong "alpha N.NN" occurrence depending on
    # prose wording). Fall back to regex-scanning ``test`` only for older prereg files that predate
    # the structured keys.
    if "alpha_family_one_sided" in pe:
        alpha_family = float(pe["alpha_family_one_sided"])
        alpha_one_sided = float(pe.get("alpha_per_endpoint_one_sided", alpha_family / 2))
    else:
        m = re.search(r"alpha\s*([0-9.]+)", str(pe.get("test", "")))
        alpha_family = float(m.group(1)) if m else 0.05
        alpha_one_sided = alpha_family / 2
    m = re.search(r"1\s*/\s*(\d+)", str(prereg.get("cost_endpoint", {}).get("statement", "1/20")))
    cost_ratio_required = float(m.group(1)) if m else 20.0
    floors_cfg = prereg.get("absolute_floors", {})
    floors = {k: float(v) for k, v in floors_cfg.items() if isinstance(v, (int, float)) and k != "at_N"}
    return {"deltas": deltas, "alpha_family": alpha_family, "alpha_one_sided": alpha_one_sided,
            "cost_ratio_required": cost_ratio_required, "floors": floors, "floors_at_N": floors_cfg.get("at_N"),
            "ci_confidence": 0.95}


def decide(endpoints: Mapping[str, Mapping[str, Any]], cost_ratio: float, floors: Mapping[str, float],
           prereg: Mapping[str, Any] | None = None, alpha_one_sided: float | None = None) -> dict[str, Any]:
    """The pre-registered decision.

    ``endpoints``: ``{"E1": result, "E2": result, "E3": result}`` from
    :func:`non_inferiority` / :func:`non_inferiority_partition` (their
    ``delta`` fields must match the pre-registration; they are checked).
    ``cost_ratio`` = USD(B, list price with caching) / USD(A, cloud-equivalent).
    ``floors``: measured ``{"nmi_vs_l2_at_oracle_k": x, "p1_at_oracle_k": y}``.
    """
    th = prereg_thresholds(prereg)
    alpha = th["alpha_one_sided"] if alpha_one_sided is None else alpha_one_sided
    checked = {}
    for name, e in endpoints.items():
        exp = th["deltas"].get(name)
        if exp is not None and not math.isclose(float(e["delta"]), exp, abs_tol=1e-12):
            raise ValueError(f"{name}: delta {e['delta']} differs from the pre-registered {exp}")
        checked[name] = e
    adjusted = holm_non_inferiority(checked, alpha=alpha)
    all_ni = all(e["non_inferior_holm"] for e in adjusted.values()) and len(adjusted) == len(th["deltas"])
    all_ni_unadjusted = all(e["non_inferior"] for e in adjusted.values())
    any_falsified = any(e["falsified"] for e in adjusted.values())
    floor_checks = {}
    for k, req in th["floors"].items():
        val = floors.get(k)
        floor_checks[k] = {"required": req, "value": val, "met": (val is not None and val >= req)}
    floors_met = all(v["met"] for v in floor_checks.values()) and bool(floor_checks)
    floor_failed = any(v["value"] is not None and not v["met"] for v in floor_checks.values())
    cost_ok = bool(cost_ratio >= th["cost_ratio_required"])
    if any_falsified or floor_failed:
        verdict = "falsified"
    elif all_ni and cost_ok and floors_met:
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "all_non_inferior_holm": bool(all_ni),
        "all_non_inferior_unadjusted": bool(all_ni_unadjusted),
        "any_falsified": bool(any_falsified),
        "cost": {"ratio": float(cost_ratio), "required": th["cost_ratio_required"], "met": cost_ok},
        "floors": floor_checks, "floors_met": bool(floors_met), "floor_failed": bool(floor_failed),
        "alpha_one_sided": alpha, "alpha_family_two_sided": th["alpha_family"],
        "endpoints": {n: {k: v for k, v in e.items() if k != "_ci"} for n, e in adjusted.items()},
        "missing_endpoints": [n for n in th["deltas"] if n not in adjusted],
        "rule": "supported iff E1-E3 non-inferior (Holm) AND cost ratio >= required AND floors met; "
                "falsified iff any upper bound < -delta or a floor fails",
    }
