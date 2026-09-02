"""Cost accounting: prices, recomputation, reconciliation, marginal-cost fit."""
import numpy as np
import pytest

from wikiorg.eval.cost import (aggregate_calls, arm_a_cost, arm_b_cost, call_cost, cloud_equivalent_usd, cost_from_model_usage,
                               cost_of, cost_ratio, load_prices, marginal_cost_fit, match_model, per_ingest_points,
                               reconcile, usage_tokens)

PRICES = load_prices()
SONNET = "claude-sonnet-5-20260401"


def test_price_table_and_matching():
    assert PRICES["fetched"] == "2026-09-02"
    assert match_model(SONNET, PRICES)[0] == "claude-sonnet-5"
    assert match_model("claude-sonnet-5", PRICES)[0] == "claude-sonnet-5"
    assert match_model("us.anthropic.claude-haiku-4-5-20251001-v1:0", PRICES)[0] == "claude-haiku-4-5"
    with pytest.raises(KeyError):
        match_model("gpt-5", PRICES)


def test_cost_of_hand_case():
    """The default cache-write tier is 1h ($4.00/MTok on Sonnet 5): every real CLI result observed
    bills cache writes at the 1-hour tier. Assuming 5m ($2.50) under-counted arm B by ~14.6%."""
    entry = PRICES["models"]["claude-sonnet-5"]
    toks = {"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_creation": 1_000_000}
    c = cost_of(toks, entry)
    assert (c["input_usd"], c["output_usd"], c["cache_read_usd"], c["cache_write_usd"]) == (2.0, 10.0, 0.2, 4.0)
    assert c["total_usd"] == pytest.approx(16.2) and c["uncached_usd"] == pytest.approx(16.0)
    # explicit tiers are still honoured
    assert cost_of(toks, entry, cache_write_tier="5m")["cache_write_usd"] == 2.5
    assert cost_of(toks, entry, cache_write_tier="1h")["cache_write_usd"] == 4.0
    # an explicit per-tier split is priced per tier, regardless of the default tier
    mixed = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 1_000_000,
             "cache_creation_1h": 750_000, "cache_creation_5m": 250_000}
    assert cost_of(mixed, entry, cache_write_tier="5m")["cache_write_usd"] == pytest.approx(0.75 * 4.0 + 0.25 * 2.5)
    assert usage_tokens({"inputTokens": 5, "cacheReadInputTokens": 7})["cache_read"] == 7
    assert usage_tokens({"inputTokens": 5, "cacheReadInputTokens": 7})["cache_creation"] == 0
    # camelCase (modelUsage) and snake_case (top-level usage) sub-dicts are both understood
    assert usage_tokens({"input_tokens": 1, "cacheCreation": {"ephemeral5mInputTokens": 3, "ephemeral1hInputTokens": 4}})["cache_creation"] == 7
    u = usage_tokens({"input_tokens": 1, "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 4}})
    assert (u["cache_creation"], u["cache_creation_5m"], u["cache_creation_1h"]) == (7, 3, 4)


def test_model_usage_and_reconciliation():
    # reported costUSD computed at the 1h tier: 1000*2 + 500*10 + 20000*0.2 + 3000*4.0 = 23000 uUSD
    mu = {SONNET: {"inputTokens": 1000, "outputTokens": 500, "cacheReadInputTokens": 20000, "cacheCreationInputTokens": 3000, "costUSD": 0.0230}}
    res = cost_from_model_usage(mu, PRICES)
    assert res["total_usd"] == pytest.approx(0.0230) and res["reported_usd"] == pytest.approx(0.0230)
    assert res["tokens"]["cache_read"] == 20000
    assert not reconcile(0.0230, 0.0230)["flag"]
    assert reconcile(0.0250, 0.0230)["flag"] and reconcile(0.0231, 0.0230)["within_tolerance"]
    assert reconcile(1.0, None)["within_tolerance"] is None
    row = {"model_usage": mu, "cost_usd": 0.0300}
    assert call_cost(row, PRICES)["reconciliation"]["flag"]


def test_real_row_shape_reconciles_exactly():
    """Regression for the 14.6% reconciliation miss: a row in the exact shape the harness writes
    (modelUsage lumps cache writes; the per-tier split and cache_write_tier live on the row) must
    reconcile to floating-point precision against the CLI's own costUSD."""
    n_in, n_out, n_read, n_write = 20, 8346, 162738, 17087
    cost_1h = (n_in * 2.0 + n_out * 10.0 + n_read * 0.2 + n_write * 4.0) / 1e6
    row = {
        "model": "claude-sonnet-5", "cache_write_tier": "1h", "cost_usd": cost_1h,
        "tokens": {"input": n_in, "output": n_out, "cache_read": n_read, "cache_creation": n_write,
                   "cache_creation_5m": 0, "cache_creation_1h": n_write},
        "model_usage": {"claude-sonnet-5": {"inputTokens": n_in, "outputTokens": n_out, "cacheReadInputTokens": n_read,
                                            "cacheCreationInputTokens": n_write, "costUSD": cost_1h}},
        "usage_raw": {"cache_creation": {"ephemeral_1h_input_tokens": n_write, "ephemeral_5m_input_tokens": 0}},
    }
    res = call_cost(row, PRICES)
    assert res["cache_write_tier"] == "1h"
    assert res["reconciliation"]["within_tolerance"] and abs(res["reconciliation"]["rel_diff"]) < 1e-12
    # the old assumption (5m) is exactly the bug: it must NOT reconcile within the pre-registered 1%
    res_5m = call_cost({**row, "cache_write_tier": None, "tokens": {k: v for k, v in row["tokens"].items() if "_1h" not in k and "_5m" not in k},
                        "usage_raw": {}}, PRICES, cache_write_tier="5m")
    assert res_5m["reconciliation"]["flag"] and abs(res_5m["reconciliation"]["rel_diff"]) > 0.10
    # when the row gives no tier information at all, the default resolves to 1h and still reconciles
    bare = {"model": "claude-sonnet-5", "cost_usd": cost_1h,
            "model_usage": {"claude-sonnet-5": {"inputTokens": n_in, "outputTokens": n_out, "cacheReadInputTokens": n_read,
                                                "cacheCreationInputTokens": n_write}}}
    assert call_cost(bare, PRICES)["reconciliation"]["within_tolerance"]


def _rows(n=80, slope_tokens=50, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pages = 3 * i + 5
        rows.append({"kind": "ingest", "status": "ok", "doc_id": f"d{i}", "ingest_index": i, "wiki_pages_before": {"total": pages, "sources": i},
                     "model": SONNET, "tokens": {"input": int(2000 + slope_tokens * pages + rng.normal(0, noise)), "output": 800,
                                                  "cache_read": 40000, "cache_creation": 1500},
                     "cost_usd": None, "num_turns": 20, "wall_ms": 60000, "duration_api_ms": 50000, "tool_calls": {"Read": 5, "Edit": 3}})
    rows.append({"kind": "lint", "status": "ok", "doc_id": None, "model": SONNET, "tokens": {"input": 50000, "output": 5000, "cache_read": 0, "cache_creation": 0},
                 "cost_usd": 0.15, "num_turns": 100, "wall_ms": 300000})
    rows.append({"kind": "ingest", "status": "failed", "doc_id": "dX", "ingest_index": n, "model": SONNET, "tokens": {"input": 100, "output": 10}, "num_turns": 1})
    return rows


def test_aggregate_calls():
    rows = _rows()
    agg = aggregate_calls(rows, PRICES)
    assert agg["n_calls"] == 82 and agg["n_ok"] == 81 and agg["by_status"] == {"ok": 81, "failed": 1}
    assert agg["n_ingests"] == 81 and agg["num_turns"] == 80 * 20 + 100 + 1
    assert agg["total_usd"] > 0 and agg["uncached_usd"] > agg["total_usd"]
    assert agg["tool_calls"] == {"Read": 400, "Edit": 240}
    assert "ingest" in agg["by_kind"] and "lint" in agg["by_kind"]
    assert agg["reconciliation"]["reported_usd"] == pytest.approx(0.15)  # only the lint row reported a cost
    assert agg["price_table_date"] == "2026-09-02"


def test_marginal_cost_fit_recovers_slope():
    rows = _rows(slope_tokens=50, noise=200, seed=1)
    fit = marginal_cost_fit(rows, PRICES, n_resamples=300, seed=0)
    expected = 50 * PRICES["models"]["claude-sonnet-5"]["input"] / 1e6  # USD per page
    assert fit["n"] == 80 and fit["x_source"] == "wiki_pages_before"
    assert fit["slope"] == pytest.approx(expected, rel=0.05)
    assert fit["slope_ci_low"] <= fit["slope"] <= fit["slope_ci_high"] and fit["ci_excludes_zero"]
    assert fit["slope_per_100_pages"] == pytest.approx(fit["slope"] * 100)
    flat = marginal_cost_fit(_rows(slope_tokens=0, noise=300, seed=2), PRICES, n_resamples=300, seed=0)
    assert flat["slope_ci_low"] < 0 < flat["slope_ci_high"] and not flat["ci_excludes_zero"]
    tiny = marginal_cost_fit(rows[:2], PRICES, n_resamples=10)
    assert np.isnan(tiny["slope"])
    pts = per_ingest_points(rows, PRICES)
    assert len(pts) == 80 and pts[0]["wiki_pages_before"] == 5 and pts[0]["usd"] > 0
    block = arm_b_cost(rows, PRICES, n_resamples=50)
    assert block["marginal_fit"]["slope"] == pytest.approx(fit["slope"]) and len(block["per_ingest"]) == 80
    assert block["per_ingest"][3]["usd"] == pytest.approx(call_cost(rows[3], PRICES)["total_usd"])


def test_arm_a_cloud_equivalent():
    c = cloud_equivalent_usd(3600, 7200, PRICES)
    assert c["gpu_usd"] == pytest.approx(0.35) and c["cpu_usd"] == pytest.approx(0.10) and c["total_usd"] == pytest.approx(0.45)
    a = arm_a_cost({"stages": [{"name": "embed", "gpu_seconds": 60, "wall_seconds": 70}, {"name": "cluster", "wall_ms": 30000}]}, PRICES)
    assert a["gpu_seconds"] == 60 and a["wall_seconds"] == 100 and a["cpu_seconds"] == 40 and a["n_llm_calls"] == 0
    assert cost_ratio(0.01, 10.0) == pytest.approx(1000.0) and cost_ratio(0.0, 1.0) == np.inf
