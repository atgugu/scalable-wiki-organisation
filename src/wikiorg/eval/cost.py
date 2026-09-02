"""Cost accounting: dated price table, recomputation from ``modelUsage`` token
counts, reconciliation against the CLI's ``total_cost_usd``, per-run
aggregates, arm A's cloud-equivalent dollars, and the arm-B *marginal cost per
ingest vs wiki size* fit -- the direct test of the gist's claim that "the cost
of maintenance is near zero" (Karpathy 2026).

Price table: ``configs/prices.yaml`` (USD per million tokens, fetched
2026-09-02 from platform.claude.com).  ``cached`` cost is the list price with
prompt caching as billed (cache reads at the cache-read price, cache creation at
the 5-minute write price); ``uncached`` re-prices every input-side token at the
plain input price, i.e. what the same calls would cost without prompt caching.
The pre-registered cost endpoint uses the cached (actual list) figure.

Token sources: ``modelUsage`` from the CLI result (includes subagents; the
top-level ``usage`` undercounts) with per-model entries
``{inputTokens, outputTokens, cacheReadInputTokens, cacheCreationInputTokens,
costUSD}``.  Key spellings from the harness' ``calls.jsonl`` contract
(``tokens{input, output, cache_read, cache_creation}``) are accepted too.

Marginal cost: OLS of per-ingest USD on the wiki size before the ingest
(``wiki_pages_before``), slope with a BCa bootstrap CI over ingests
(``scipy.stats.bootstrap``); a CI excluding zero means maintenance cost grows
with the wiki.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRICES = REPO_ROOT / "configs" / "prices.yaml"

_TOKEN_ALIASES = {
    "input": ("inputTokens", "input_tokens", "input"),
    "output": ("outputTokens", "output_tokens", "output"),
    "cache_read": ("cacheReadInputTokens", "cache_read_input_tokens", "cache_read"),
    "cache_creation": ("cacheCreationInputTokens", "cache_creation_input_tokens", "cache_creation", "cache_write"),
}


def load_prices(path: str | Path | None = None) -> dict[str, Any]:
    import yaml  # type: ignore
    p = Path(path) if path else DEFAULT_PRICES
    with open(p, encoding="utf-8") as fh:
        prices = yaml.safe_load(fh)
    prices.setdefault("models", {})
    prices.setdefault("compute", {"gpu_usd_per_hour": 0.35, "cpu_usd_per_hour": 0.05})
    prices["_path"] = str(p)
    return prices


def _family(model_id: str) -> tuple[str | None, str | None]:
    m = re.search(r"(sonnet|haiku|opus)[-_ ]?(\d+(?:[-.]\d+)?)?", model_id.lower())
    if not m:
        return None, None
    ver = (m.group(2) or "").replace("-", ".")
    return m.group(1), ver.split(".")[0] if ver else None


def match_model(model_id: str, prices: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    """Find the price entry for a full model id (exact, longest prefix, or family+major version)."""
    models = prices["models"]
    if model_id in models:
        return model_id, models[model_id]
    low = model_id.lower()
    prefixes = [k for k in models if low.startswith(k.lower())]
    if prefixes:
        k = max(prefixes, key=len)
        return k, models[k]
    fam, major = _family(model_id)
    for k in models:
        kf, km = _family(k)
        if fam and kf == fam and (major is None or km is None or km == major):
            return k, models[k]
    raise KeyError(f"no price entry for model {model_id!r} in {prices.get('_path', 'prices')}")


def usage_tokens(usage: Mapping[str, Any]) -> dict[str, int]:
    """Normalise a per-model usage dict to ``{input, output, cache_read, cache_creation}``."""
    out: dict[str, int] = {}
    for key, aliases in _TOKEN_ALIASES.items():
        val = 0
        for a in aliases:
            v = usage.get(a)
            # "cache_creation" is both an alias for the lump count and, in the CLI's snake_case
            # top-level usage, the name of a per-tier sub-dict -- only accept numeric values here.
            if v is not None and not isinstance(v, Mapping):
                val = v
                break
        out[key] = int(val or 0)
    cc = usage.get("cacheCreation") or usage.get("cache_creation_detail")
    if not isinstance(cc, Mapping):
        # The CLI's top-level ``usage`` emits snake_case: {"cache_creation": {"ephemeral_1h_input_tokens", ...}}
        raw_cc = usage.get("cache_creation")
        if isinstance(raw_cc, Mapping):
            cc = raw_cc
    if isinstance(cc, Mapping):
        n5 = int(cc.get("ephemeral5mInputTokens", cc.get("ephemeral_5m_input_tokens", 0)) or 0)
        n1 = int(cc.get("ephemeral1hInputTokens", cc.get("ephemeral_1h_input_tokens", 0)) or 0)
        out["cache_creation_5m"], out["cache_creation_1h"] = n5, n1
        if out["cache_creation"] == 0:
            out["cache_creation"] = n5 + n1
    for k in ("cache_creation_5m", "cache_creation_1h"):
        if k in usage and usage[k] is not None:
            out[k] = int(usage[k] or 0)
    return out


DEFAULT_CACHE_WRITE_TIER = "1h"
"""Every real CLI result observed so far (W0 smoke, W2 validation, the live arm-B run) reports
``usage.cache_creation.ephemeral_1h_input_tokens == cache_creation_input_tokens`` and zero
5-minute writes, i.e. cache writes are billed at the 1-hour tier ($4.00/MTok on Sonnet 5), not
the 5-minute tier ($2.50). Assuming "5m" under-counted arm B's cost by ~14.6% and broke the
pre-registered <=1% reconciliation against the CLI's own total_cost_usd."""


def _write_price(entry: Mapping[str, float], tier: str) -> float:
    return entry.get(f"cache_write_{tier}", entry.get("cache_write_5m", entry["input"] * 1.25))


def cost_of(tokens: Mapping[str, int], entry: Mapping[str, float],
            cache_write_tier: str = DEFAULT_CACHE_WRITE_TIER) -> dict[str, float]:
    """USD for one model's tokens at the given price entry (per million tokens).

    If ``tokens`` carries an explicit per-tier split (``cache_creation_1h`` / ``cache_creation_5m``)
    each tier is priced at its own rate; otherwise the lump ``cache_creation`` count is priced at
    ``cache_write_tier``."""
    m = 1e6
    n1 = int(tokens.get("cache_creation_1h", 0) or 0)
    n5 = int(tokens.get("cache_creation_5m", 0) or 0)
    if n1 or n5:
        cache_write_usd = (n1 * _write_price(entry, "1h") + n5 * _write_price(entry, "5m")) / m
    else:
        cache_write_usd = tokens.get("cache_creation", 0) * _write_price(entry, cache_write_tier) / m
    parts = {
        "input_usd": tokens.get("input", 0) * entry["input"] / m,
        "output_usd": tokens.get("output", 0) * entry["output"] / m,
        "cache_read_usd": tokens.get("cache_read", 0) * entry.get("cache_read", entry["input"] * 0.1) / m,
        "cache_write_usd": cache_write_usd,
    }
    parts["total_usd"] = sum(parts.values())
    parts["uncached_usd"] = ((tokens.get("input", 0) + tokens.get("cache_read", 0) + tokens.get("cache_creation", 0))
                             * entry["input"] / m + parts["output_usd"])
    return parts


def cost_from_model_usage(model_usage: Mapping[str, Mapping[str, Any]], prices: Mapping[str, Any],
                          cache_write_tier: str = DEFAULT_CACHE_WRITE_TIER,
                          split: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Recompute USD from a ``modelUsage`` dict; also sums the CLI's own ``costUSD``."""
    per_model: dict[str, Any] = {}
    totals = Counter()
    reported = 0.0
    any_reported = False
    for model_id, usage in model_usage.items():
        key, entry = match_model(model_id, prices)
        toks = usage_tokens(usage)
        # modelUsage lumps cache writes into cacheCreationInputTokens; the per-tier split lives on
        # the row (tokens.cache_creation_1h/5m or usage_raw). Apply it when there is a single model.
        if split and len(model_usage) == 1 and not (toks.get("cache_creation_1h") or toks.get("cache_creation_5m")):
            toks = {**toks, "cache_creation_1h": int(split.get("1h", 0) or 0), "cache_creation_5m": int(split.get("5m", 0) or 0)}
        c = cost_of(toks, entry, cache_write_tier)
        per_model[model_id] = {"price_key": key, "tokens": toks, **c}
        for k, v in toks.items():
            totals[k] += v
        totals["total_usd"] += c["total_usd"]
        totals["uncached_usd"] += c["uncached_usd"]
        if usage.get("costUSD") is not None or usage.get("cost_usd") is not None:
            any_reported = True
            reported += float(usage.get("costUSD", usage.get("cost_usd")) or 0.0)
    return {"per_model": per_model, "tokens": {k: int(totals[k]) for k in _TOKEN_ALIASES},
            "total_usd": float(totals["total_usd"]), "uncached_usd": float(totals["uncached_usd"]),
            "reported_usd": reported if any_reported else None}


def reconcile(computed: float, reported: float | None, tol: float = 0.01) -> dict[str, Any]:
    """Compare recomputed USD with the CLI's ``total_cost_usd`` (flag > ``tol`` relative)."""
    if reported is None:
        return {"computed_usd": computed, "reported_usd": None, "abs_diff": None, "rel_diff": None,
                "within_tolerance": None, "flag": False}
    diff = computed - reported
    denom = max(abs(reported), 1e-9)
    rel = diff / denom
    within = abs(rel) <= tol or abs(diff) < 1e-6
    return {"computed_usd": computed, "reported_usd": reported, "abs_diff": diff, "rel_diff": rel,
            "within_tolerance": bool(within), "flag": not within, "tolerance": tol}


def row_cache_tier(row: Mapping[str, Any]) -> tuple[str, dict[str, int] | None]:
    """(default tier, per-tier split) for a ``calls.jsonl`` row.

    Split comes from ``tokens.cache_creation_1h/5m`` or ``usage_raw.cache_creation``; the tier from
    ``row['cache_write_tier']``, else the dominant tier of the split, else :data:`DEFAULT_CACHE_WRITE_TIER`."""
    toks = row.get("tokens") or {}
    split: dict[str, int] | None = None
    if isinstance(toks, Mapping) and ("cache_creation_1h" in toks or "cache_creation_5m" in toks):
        split = {"1h": int(toks.get("cache_creation_1h") or 0), "5m": int(toks.get("cache_creation_5m") or 0)}
    else:
        ur = row.get("usage_raw") or {}
        cc = ur.get("cache_creation") if isinstance(ur, Mapping) else None
        if isinstance(cc, Mapping):
            split = {"1h": int(cc.get("ephemeral_1h_input_tokens") or 0), "5m": int(cc.get("ephemeral_5m_input_tokens") or 0)}
    tier = row.get("cache_write_tier")
    if tier not in ("1h", "5m"):
        if split and (split["1h"] or split["5m"]):
            tier = "1h" if split["1h"] >= split["5m"] else "5m"
        else:
            tier = DEFAULT_CACHE_WRITE_TIER
    return tier, split


def call_cost(row: Mapping[str, Any], prices: Mapping[str, Any], cache_write_tier: str | None = None) -> dict[str, Any]:
    """Recomputed cost of one ``calls.jsonl`` row (``model_usage`` preferred over ``tokens``+``model``).

    ``cache_write_tier`` overrides the row's own tier when given; otherwise the tier and the exact
    per-tier write split are taken from the row (see :func:`row_cache_tier`)."""
    row_tier, split = row_cache_tier(row)
    tier = cache_write_tier or row_tier
    mu = row.get("model_usage") or row.get("modelUsage")
    if mu:
        res = cost_from_model_usage(mu, prices, tier, split=split)
    else:
        toks = usage_tokens(row.get("tokens") or row.get("usage") or {})
        model = row.get("model") or "claude-sonnet-5"
        key, entry = match_model(model, prices)
        c = cost_of(toks, entry, tier)
        res = {"per_model": {model: {"price_key": key, "tokens": toks, **c}}, "tokens": toks,
               "total_usd": c["total_usd"], "uncached_usd": c["uncached_usd"], "reported_usd": None}
    reported = row.get("cost_usd", row.get("total_cost_usd"))
    res["reported_usd"] = float(reported) if reported is not None else res.get("reported_usd")
    res["cache_write_tier"] = tier
    res["reconciliation"] = reconcile(res["total_usd"], res["reported_usd"])
    return res


def _wiki_size(row: Mapping[str, Any]) -> float | None:
    v = row.get("wiki_pages_before")
    if isinstance(v, Mapping):
        if "total" in v:
            return float(v["total"])
        nums = [x for x in v.values() if isinstance(x, (int, float))]
        return float(sum(nums)) if nums else None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def aggregate_calls(rows: Iterable[Mapping[str, Any]], prices: Mapping[str, Any] | None = None,
                    tol: float = 0.01, ok_statuses: Sequence[str] = ("ok", "success")) -> dict[str, Any]:
    """Totals for a run: calls, turns, tokens, USD (cached / uncached / reported),
    reconciliation flags, wall-clock and API time, breakdown by ``kind`` and ``status``."""
    prices = prices or load_prices()
    rows = list(rows)
    tokens = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_status: Counter = Counter()
    total = uncached = reported = 0.0
    turns = wall_ms = api_ms = dur_ms = 0
    flagged: list[dict[str, Any]] = []
    tool_calls: Counter = Counter()
    n_reported = 0
    for r in rows:
        c = call_cost(r, prices)
        kind = str(r.get("kind", "call"))
        status = str(r.get("status", "ok"))
        by_status[status] += 1
        for k, v in c["tokens"].items():
            tokens[k] += v
            by_kind[kind][k] += v
        total += c["total_usd"]
        uncached += c["uncached_usd"]
        by_kind[kind]["total_usd"] += c["total_usd"]
        by_kind[kind]["n_calls"] += 1
        if c["reported_usd"] is not None:
            reported += c["reported_usd"]
            n_reported += 1
            if c["reconciliation"]["flag"] and abs(c["reconciliation"]["rel_diff"] or 0) > tol:
                flagged.append({"doc_id": r.get("doc_id"), "ingest_index": r.get("ingest_index"),
                                **{k: c["reconciliation"][k] for k in ("computed_usd", "reported_usd", "rel_diff")}})
        turns += int(r.get("num_turns", 0) or 0)
        wall_ms += int(r.get("wall_ms", 0) or 0)
        api_ms += int(r.get("duration_api_ms", 0) or 0)
        dur_ms += int(r.get("duration_ms", 0) or 0)
        tc = r.get("tool_calls")
        if isinstance(tc, Mapping):
            for k, v in tc.items():
                try:
                    tool_calls[str(k)] += int(v)
                except (TypeError, ValueError):
                    pass
    out: dict[str, Any] = {
        "n_calls": len(rows), "n_ok": sum(v for k, v in by_status.items() if k in ok_statuses),
        "by_status": dict(by_status), "tokens": {k: int(tokens[k]) for k in _TOKEN_ALIASES},
        "total_usd": total, "uncached_usd": uncached,
        "reported_usd": reported if n_reported else None,
        "reconciliation": reconcile(total, reported if n_reported else None, tol),
        "n_calls_flagged": len(flagged), "flagged_calls": flagged,
        "num_turns": turns, "wall_ms": wall_ms, "duration_api_ms": api_ms, "duration_ms": dur_ms,
        "wall_hours": wall_ms / 3.6e6, "tool_calls": dict(tool_calls),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "price_table": prices.get("_path"), "price_table_date": prices.get("fetched"),
    }
    ingests = [r for r in rows if str(r.get("kind", "")) == "ingest"]
    if ingests:
        per = np.array([call_cost(r, prices)["total_usd"] for r in ingests])
        out["ingest_usd_mean"] = float(per.mean())
        out["ingest_usd_median"] = float(np.median(per))
        out["n_ingests"] = len(ingests)
    return out


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(slope, intercept, r2) by least squares."""
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(beta[0]), float(beta[1]), r2


def marginal_cost_fit(rows: Iterable[Mapping[str, Any]], prices: Mapping[str, Any] | None = None,
                      n_resamples: int = 2000, seed: int = 0, confidence: float = 0.95,
                      y_key: str = "usd", ok_statuses: Sequence[str] = ("ok", "success")) -> dict[str, Any]:
    """OLS slope of per-ingest cost (USD, or ``y_key="tokens"`` for total tokens)
    on wiki size before the ingest, with a BCa bootstrap CI over ingests.

    ``x`` is ``wiki_pages_before`` (``total`` of the page-count dict) and falls
    back to ``ingest_index`` when absent (``x_source`` says which).  Only
    successful ingests enter the fit.  ``slope_per_100_pages`` is the extra USD
    an ingest costs for every 100 additional wiki pages.
    """
    from scipy import stats

    prices = prices or load_prices()
    rows = [r for r in rows if str(r.get("kind", "ingest")) == "ingest" and str(r.get("status", "ok")) in ok_statuses]
    xs, ys, x_source = [], [], "wiki_pages_before"
    for r in rows:
        c = call_cost(r, prices)
        x = _wiki_size(r)
        if x is None:
            x_source = "ingest_index"
            x = float(r.get("ingest_index", len(xs)))
        xs.append(x)
        ys.append(c["total_usd"] if y_key == "usd" else float(sum(c["tokens"].values())))
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(x)
    out: dict[str, Any] = {"n": n, "x_source": x_source, "y": y_key, "method": "OLS + BCa bootstrap over ingests",
                           "n_resamples": n_resamples, "confidence": confidence, "seed": seed}
    if n < 3 or np.ptp(x) == 0:
        out.update({"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                    "slope_ci_low": float("nan"), "slope_ci_high": float("nan"), "slope_per_100_pages": float("nan"),
                    "mean_y": float(y.mean()) if n else float("nan")})
        return out
    slope, intercept, r2 = _ols(x, y)

    def stat(xx, yy, axis=-1):
        xx, yy = np.asarray(xx), np.asarray(yy)
        xm = xx.mean(axis=axis, keepdims=True)
        ym = yy.mean(axis=axis, keepdims=True)
        num = ((xx - xm) * (yy - ym)).sum(axis=axis)
        den = ((xx - xm) ** 2).sum(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            return num / den

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = stats.bootstrap((x, y), stat, paired=True, vectorized=True, n_resamples=n_resamples,
                                  confidence_level=confidence, method="BCa", rng=np.random.default_rng(seed))
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            method = "BCa"
        except Exception:
            lo = hi = float("nan")
            method = "failed"
        if not (np.isfinite(lo) and np.isfinite(hi)):
            res = stats.bootstrap((x, y), stat, paired=True, vectorized=True, n_resamples=n_resamples,
                                  confidence_level=confidence, method="percentile", rng=np.random.default_rng(seed))
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            method = "percentile (BCa degenerate)"
    out.update({"slope": slope, "intercept": intercept, "r2": r2, "slope_ci_low": lo, "slope_ci_high": hi,
                "ci_method": method, "slope_per_100_pages": slope * 100, "slope_ci_low_per_100": lo * 100,
                "slope_ci_high_per_100": hi * 100, "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "mean_y": float(y.mean()), "x_min": float(x.min()), "x_max": float(x.max()),
                "y_first_quintile_mean": float(y[x <= np.quantile(x, 0.2)].mean()),
                "y_last_quintile_mean": float(y[x >= np.quantile(x, 0.8)].mean()),
                "spearman_rho": float(stats.spearmanr(x, y).statistic)})
    return out


def per_ingest_points(rows: Iterable[Mapping[str, Any]], prices: Mapping[str, Any] | None = None,
                      ok_statuses: Sequence[str] = ("ok", "success")) -> list[dict[str, Any]]:
    """One point per successful ingest (the F4 source): wiki size before, USD
    recomputed (cached and uncached), tokens, turns, wall-clock."""
    prices = prices or load_prices()
    pts = []
    for r in rows:
        if str(r.get("kind", "ingest")) != "ingest" or str(r.get("status", "ok")) not in ok_statuses:
            continue
        c = call_cost(r, prices)
        pts.append({"ingest_index": r.get("ingest_index"), "doc_id": r.get("doc_id"),
                    "wiki_pages_before": _wiki_size(r), "usd": c["total_usd"], "usd_uncached": c["uncached_usd"],
                    "usd_reported": c["reported_usd"], "tokens": int(sum(c["tokens"].values())),
                    "output_tokens": c["tokens"]["output"], "num_turns": r.get("num_turns"), "wall_ms": r.get("wall_ms"),
                    "files_touched": (len(r["files_touched"]) if isinstance(r.get("files_touched"), (list, dict)) else r.get("files_touched"))})
    return pts


def arm_b_cost(rows: Iterable[Mapping[str, Any]], prices: Mapping[str, Any] | None = None, n_resamples: int = 2000,
               seed: int = 0) -> dict[str, Any]:
    """Cost block for an LLM arm run: aggregates + marginal-cost fit + per-ingest points
    (all from the same recomputed USD, so the figure and the fit cannot disagree)."""
    prices = prices or load_prices()
    rows = list(rows)
    out = aggregate_calls(rows, prices)
    out["marginal_fit"] = marginal_cost_fit(rows, prices, n_resamples=n_resamples, seed=seed)
    out["marginal_fit_tokens"] = marginal_cost_fit(rows, prices, n_resamples=n_resamples, seed=seed, y_key="tokens")
    out["per_ingest"] = per_ingest_points(rows, prices)
    return out


def cloud_equivalent_usd(gpu_seconds: float, cpu_seconds: float, prices: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Arm A's dollar-equivalent from raw seconds at the price table's rental rates."""
    prices = prices or load_prices()
    comp = prices.get("compute", {})
    gpu_rate = float(comp.get("gpu_usd_per_hour", 0.35))
    cpu_rate = float(comp.get("cpu_usd_per_hour", 0.05))
    gpu_usd = gpu_seconds / 3600 * gpu_rate
    cpu_usd = cpu_seconds / 3600 * cpu_rate
    return {"gpu_seconds": float(gpu_seconds), "cpu_seconds": float(cpu_seconds), "gpu_usd_per_hour": gpu_rate,
            "cpu_usd_per_hour": cpu_rate, "gpu_usd": gpu_usd, "cpu_usd": cpu_usd, "total_usd": gpu_usd + cpu_usd}


def arm_a_cost(timing: Mapping[str, Any], prices: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Cost block for arm A from a ``timing.json`` (tolerant: top-level
    ``gpu_seconds``/``cpu_seconds``/``wall_seconds`` or a ``stages`` list/dict
    with those fields; ``*_ms`` variants accepted)."""
    def secs(d: Mapping[str, Any], key: str) -> float:
        if key + "_seconds" in d:
            return float(d[key + "_seconds"] or 0)
        if key + "_s" in d:
            return float(d[key + "_s"] or 0)
        if key + "_ms" in d:
            return float(d[key + "_ms"] or 0) / 1000
        return 0.0

    if "gpu_stage_wall_s" in timing or "cpu_stage_wall_s" in timing:
        # wikiorg.timing.Timing.summary(): GPU stages billed on wall-clock at the GPU rate, others at the CPU rate
        gpu = float(timing.get("gpu_stage_wall_s") or 0.0)
        cpu = float(timing.get("cpu_stage_wall_s") or 0.0)
        wall = float(timing.get("total_wall_s") or (gpu + cpu))
    else:
        gpu = secs(timing, "gpu")
        cpu = secs(timing, "cpu")
        wall = secs(timing, "wall")
        stages = timing.get("stages")
        if stages:
            items = stages.values() if isinstance(stages, Mapping) else stages
            for s in items:
                if not isinstance(s, Mapping):
                    continue
                if "gpu" in s and "wall_s" in s:  # Timing-style stage: {wall_s, gpu: bool, ...}
                    ws = float(s.get("wall_s", 0.0) or 0.0)
                    if s.get("gpu"):
                        gpu += ws
                    else:
                        cpu += ws
                    wall += ws
                else:  # explicit per-stage seconds
                    gpu += secs(s, "gpu")
                    cpu += secs(s, "cpu")
                    wall += secs(s, "wall")
        if cpu == 0 and wall > 0:
            cpu = max(wall - gpu, 0.0)  # CPU time defaults to non-GPU wall-clock
    out = cloud_equivalent_usd(gpu, cpu, prices)
    out["wall_seconds"] = wall
    out["wall_hours"] = wall / 3600
    out["gpu_event_seconds"] = timing.get("gpu_event_s")
    out["peak_vram_mib"] = timing.get("peak_vram_mib")
    out["usd_reported"] = timing.get("usd_cloud_equivalent")
    out["reconciliation"] = reconcile(out["total_usd"], timing.get("usd_cloud_equivalent"))
    out["n_llm_calls"] = 0
    out["n_calls"] = 0
    out["num_turns"] = 0
    out["tokens"] = {k: 0 for k in _TOKEN_ALIASES}
    out["price_table_date"] = (prices or load_prices()).get("fetched")
    return out


def cost_ratio(usd_a: float, usd_b: float) -> float:
    """USD(B) / USD(A); infinite when A costs nothing."""
    if usd_a <= 0:
        return float("inf") if usd_b > 0 else float("nan")
    return usd_b / usd_a
