"""Unit tests for the Claude CLI harness.  No network, no subprocess, no money.

The captured strings below are the shapes the CLI produced during the W0 smoke
(``configs/cli_facts.json``, 2026-09-02) plus the error texts the classifier has
to keep apart during a run that spans days.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from wikiorg.llm import calls as C
from wikiorg.llm import runner as R

# --------------------------------------------------------------------------------------------
# fixtures: captured CLI output shapes
# --------------------------------------------------------------------------------------------
# Token counts of the shape the W0 smoke ingest produced (cache_read 60623), with the
# dollar figure made exactly consistent with configs/prices.yaml so the 1 % reconciliation
# the pre-registration requires can be asserted arithmetically.
# cache writes are billed at the 1-hour tier ($4.00/MTok on Sonnet 5) -- every real CLI result shows
# ephemeral_1h_input_tokens == cache_creation_input_tokens; 2.5 (the 5-minute rate) was the bug.
SMOKE_USD = (41 * 2.0 + 5000 * 10.0 + 60623 * 0.2 + 12045 * 4.0) / 1e6
MODEL_USAGE = {
    "claude-sonnet-5": {"inputTokens": 41, "outputTokens": 5000, "cacheReadInputTokens": 60623,
                        "cacheCreationInputTokens": 12045, "costUSD": SMOKE_USD,
                        "webSearchRequests": 0, "contextWindow": 200000},
}
RESULT_OK = {
    "type": "result", "subtype": "success", "is_error": False, "duration_ms": 61300,
    "duration_api_ms": 58120, "num_turns": 11, "session_id": "3f2a-...",
    "result": "DONE topic=association-football new_topic=yes files=7",
    "total_cost_usd": SMOKE_USD,
    "usage": {"input_tokens": 41, "output_tokens": 5000, "cache_read_input_tokens": 60623,
              "cache_creation_input_tokens": 12045},
    "modelUsage": MODEL_USAGE, "permission_denials": [],
}
RESULT_BUDGET = {**RESULT_OK, "subtype": "error_max_budget_usd", "is_error": True,
                 "result": "Budget exceeded"}
RESULT_TURNS = {**RESULT_OK, "subtype": "error_max_turns", "is_error": True,
                "result": "Reached maximum turns"}

INIT_EVENT = {"type": "system", "subtype": "init", "session_id": "3f2a-...",
              "model": "claude-sonnet-5", "permissionMode": "acceptEdits", "mcp_servers": [],
              "tools": ["Read", "Write", "Edit", "Glob", "Grep"], "slash_commands": []}


def stream_stdout(result: dict, tools=("Read", "Glob", "Write", "Write", "Edit"), thinking=True) -> str:
    lines = [json.dumps(INIT_EVENT)]
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": "..."})
    for i, t in enumerate(tools):
        content.append({"type": "tool_use", "id": f"tu{i}", "name": t, "input": {}})
        lines.append(json.dumps({"type": "assistant", "message": {"content": [content[-1]]}}))
    lines.insert(1, json.dumps({"type": "assistant", "message": {"content": content[:1]}}))
    lines.append(json.dumps(result))
    return "\n".join(lines) + "\n"


USAGE_LIMIT_TEXTS = [
    "Claude AI usage limit reached|1788451200",
    "You've reached your weekly usage limit. Your limit will reset at 3pm (Europe/Bucharest).",
    "Claude usage limit reached. Try again later.",
    "You have hit your limit for this 5-hour window; it resets at 21:00.",
]
TRANSIENT_TEXTS = [
    '{"type":"result","subtype":"error_during_execution","is_error":true,"api_error_status":529}',
    '{"type":"result","is_error":true,"error":"overloaded_error: Overloaded"}',
    '{"type":"result","is_error":true,"error":"rate_limit_error"}',
    '{"type":"result","is_error":true,"api_error_status":500}',
    "fetch failed: ECONNRESET",
    "",
]
PERMANENT_TEXTS = [
    '{"type":"result","is_error":true,"error":"authentication_error: invalid bearer token"}',
    '{"type":"result","is_error":true,"error":"invalid_request_error: prompt is too long"}',
    '{"type":"result","is_error":true,"api_error_status":400}',
]


# --------------------------------------------------------------------------------------------
# environment and flags
# --------------------------------------------------------------------------------------------
def test_build_env_pins_and_drops():
    base = {"HOME": "/home/x", "PATH": "/bin", "ANTHROPIC_API_KEY": "sk-x", "CLAUDECODE": "1",
            "CLAUDE_CODE_ENTRYPOINT": "cli", "DISABLE_AUTOUPDATER": "0"}
    env = R.build_env(base=base)
    assert env["HOME"] == "/home/x" and env["PATH"] == "/bin"
    for dropped in ("ANTHROPIC_API_KEY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        assert dropped not in env, dropped
    for key, val in R.ENV_PINS.items():
        assert env[key] == val


def test_common_flags_pins_model_and_effort_and_never_falls_back():
    flags = R.common_flags(budget=2.0, max_turns=80)
    assert flags[:4] == ["--model", "claude-sonnet-5", "--effort", "high"]
    for required in ("--setting-sources", "--strict-mcp-config", "--disable-slash-commands",
                     "--no-session-persistence"):
        assert required in flags
    assert "--fallback-model" not in flags
    assert flags[flags.index("--max-budget-usd") + 1] == "2"
    assert flags[flags.index("--max-turns") + 1] == "80"
    assert "--max-budget-usd" not in R.common_flags()


# --------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------
def test_parse_result_json_format():
    got = R.parse_result(json.dumps(RESULT_OK))
    assert got["num_turns"] == 11 and got["total_cost_usd"] == pytest.approx(SMOKE_USD)


def test_parse_result_stream_json_takes_the_last_result_line():
    stdout = stream_stdout(RESULT_OK)
    got = R.parse_result(stdout)
    assert got["subtype"] == "success" and got["num_turns"] == 11


def test_parse_result_ignores_trailing_noise_and_empty():
    assert R.parse_result("") is None
    assert R.parse_result("not json at all") is None
    stdout = stream_stdout(RESULT_OK) + "warning: something on stdout\n"
    assert R.parse_result(stdout)["subtype"] == "success"


def test_parse_result_prefers_the_last_result_line():
    a = {**RESULT_OK, "num_turns": 1}
    stdout = json.dumps(a) + "\n" + json.dumps(RESULT_OK) + "\n"
    assert R.parse_result(stdout)["num_turns"] == 11


def test_tool_calls_and_thinking_from_the_stream():
    ev = R.stream_events(stream_stdout(RESULT_OK))
    assert R.tool_calls_from_events(ev) == {"Read": 1, "Glob": 1, "Write": 2, "Edit": 1}
    assert R.has_thinking(ev) is True
    assert R.has_thinking(R.stream_events(stream_stdout(RESULT_OK, thinking=False))) is False
    assert R.init_event(ev)["tools"] == ["Read", "Write", "Edit", "Glob", "Grep"]


# --------------------------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------------------------
def test_usage_prefers_model_usage_over_usage():
    """``usage`` omits subagent traffic, so ``modelUsage`` wins whenever present."""
    result = {**RESULT_OK, "modelUsage": {
        "claude-sonnet-5": {"inputTokens": 100, "outputTokens": 200, "cacheReadInputTokens": 300,
                            "cacheCreationInputTokens": 400}}}
    tokens, mu = R.usage_from_model_usage(result)
    assert tokens == {"input": 100, "output": 200, "cache_read": 300, "cache_creation": 400}
    assert mu is result["modelUsage"]


def test_usage_sums_every_model_but_asserts_one_sonnet():
    two_models = {"claude-sonnet-5": {"inputTokens": 10, "outputTokens": 1},
                  "claude-haiku-4-5": {"inputTokens": 5, "outputTokens": 2}}
    tokens, _ = R.usage_from_model_usage({"modelUsage": two_models})
    assert tokens["input"] == 15 and tokens["output"] == 3

    with pytest.raises(R.HarnessError):
        R.usage_from_model_usage({"modelUsage": {"claude-sonnet-5": {}, "claude-sonnet-4-5": {}}})
    with pytest.raises(R.HarnessError):
        R.usage_from_model_usage({"modelUsage": {"claude-opus-5": {}}})
    # the guard can be switched off for error paths where the model never answered
    R.usage_from_model_usage({"modelUsage": {"claude-opus-5": {}}}, require_single_sonnet=False)


def test_usage_falls_back_to_usage_when_model_usage_is_absent():
    tokens, mu = R.usage_from_model_usage({"usage": RESULT_OK["usage"]})
    assert tokens["cache_read"] == 60623 and mu == {}


def test_cost_recomputation_matches_model_usage_within_one_percent():
    """The pre-registration reconciles recomputed list price with ``total_cost_usd`` to 1 %."""
    from wikiorg.eval.cost import call_cost, load_prices

    row = {"model_usage": MODEL_USAGE, "cost_usd": RESULT_OK["total_cost_usd"], "model": "claude-sonnet-5"}
    res = call_cost(row, load_prices())
    assert res["tokens"]["cache_read"] == 60623
    assert res["reconciliation"]["within_tolerance"], res["reconciliation"]
    # hand-computed list price: 41*2 + 5000*10 + 60623*0.2 + 12045*2.5 per million tokens
    assert res["total_usd"] == pytest.approx(SMOKE_USD, rel=1e-9)
    # uncached: every input-side token at the plain input price
    assert res["uncached_usd"] == pytest.approx(((41 + 60623 + 12045) * 2.0 + 5000 * 10.0) / 1e6, rel=1e-9)


def test_calls_row_tokens_are_the_keys_eval_cost_expects():
    inv = R.Invocation(cmd=[], returncode=0, stdout=stream_stdout(RESULT_OK), stderr="", wall_ms=61300)
    res = R.parse_call(inv)
    row = C.row_from_call(res, kind="ingest", attempt=0, prompt="p", started_at="a", ended_at="b")
    assert set(row["tokens"]) == {"input", "output", "cache_read", "cache_creation"}
    assert row["done_line"] == "DONE topic=association-football new_topic=yes files=7"
    assert row["topic"] == "association-football" and row["new_topic"] is True and row["files"] == 7
    assert row["num_turns"] == 11 and row["tool_calls"]["Write"] == 2 and row["thinking"] is True


# --------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------
def test_classify_success_and_cli_error_subtypes():
    assert R.classify(RESULT_OK) == "ok"
    assert R.classify(RESULT_BUDGET) == "budget"
    assert R.classify(RESULT_TURNS) == "turns"
    assert R.classify({**RESULT_OK, "subtype": "error_max_structured_output_retries",
                       "is_error": True}) == "permanent"


@pytest.mark.parametrize("text", USAGE_LIMIT_TEXTS)
def test_classify_usage_limit(text):
    assert R.classify(None, stdout=text) == "usage_limit"


@pytest.mark.parametrize("text", TRANSIENT_TEXTS)
def test_classify_transient(text):
    assert R.classify(None, stdout=text) == "transient"


@pytest.mark.parametrize("text", PERMANENT_TEXTS)
def test_classify_permanent(text):
    assert R.classify(None, stdout=text) == "permanent"


def test_classify_timeout_and_signal_beat_the_text():
    inv = R.Invocation(cmd=[], returncode=-9, stdout="", stderr="", wall_ms=1200000, timed_out=True)
    assert R.classify(None, inv) == "timeout"
    killed = R.Invocation(cmd=[], returncode=-2, stdout=json.dumps(RESULT_OK), stderr="", wall_ms=10)
    assert R.classify(None, killed) == "interrupted"


def test_usage_limit_wins_over_a_co_occurring_rate_limit_word():
    text = "rate_limit_error: Claude AI usage limit reached|1788451200"
    assert R.classify(None, stdout=text) == "usage_limit"


# --------------------------------------------------------------------------------------------
# retries
# --------------------------------------------------------------------------------------------
def _fake(status: str) -> R.CallResult:
    inv = R.Invocation(cmd=[], returncode=0, stdout="", stderr="", wall_ms=1)
    return R.CallResult(invocation=inv, result=None, status=status, error=status)


def test_backoff_is_the_pre_registered_schedule():
    rng = random.Random(0)
    for k in range(5):
        s = R.backoff_seconds(k, rng=rng)
        assert 30 * 2 ** k <= s <= 30 * 2 ** k + 30


def test_with_retries_stops_on_terminal_statuses():
    for status in R.TERMINAL_STATUSES:
        n = {"c": 0}

        def call(_k, status=status, n=n):
            n["c"] += 1
            return _fake(status)

        assert with_no_sleep(call).status == status
        assert n["c"] == 1


def with_no_sleep(call, **kw):
    return R.with_retries(call, sleep=lambda _s: None, log=lambda _m: None, **kw)


def test_with_retries_retries_transients_five_times_then_gives_up():
    n = {"c": 0}

    def call(_k):
        n["c"] += 1
        return _fake("transient")

    res = with_no_sleep(call)
    assert res.status == "transient" and n["c"] == 5


def test_with_retries_succeeds_after_transients():
    seq = ["transient", "transient", "ok"]

    def call(k):
        return _fake(seq[k])

    assert with_no_sleep(call).status == "ok"


def test_usage_limit_waits_and_does_not_consume_attempts():
    """A run spanning days hits the weekly cap repeatedly; that must not exhaust retries."""
    seq = ["usage_limit"] * 8 + ["ok"]
    waits = []

    def call(k):
        return _fake(seq[k])

    res = R.with_retries(call, sleep=lambda _s: None, log=lambda _m: None,
                         on_wait=lambda r: waits.append(r) or 0.0)
    assert res.status == "ok" and len(waits) == 8


# --------------------------------------------------------------------------------------------
# usage-limit reset parsing and waiting
# --------------------------------------------------------------------------------------------
def test_parse_reset_time_unix_pipe_form():
    got = R.parse_reset_time("Claude AI usage limit reached|1788451200")
    assert got is not None and int(got.timestamp()) == 1788451200


def test_parse_reset_time_clock_form_rolls_to_tomorrow():
    now = datetime(2026, 9, 2, 16, 0).astimezone()
    got = R.parse_reset_time("Your limit will reset at 3pm.", now=now)
    assert got is not None and got.hour == 15 and got.date() == (now + timedelta(days=1)).date()
    got = R.parse_reset_time("Your limit will reset at 9pm.", now=now)
    assert got.hour == 21 and got.date() == now.date()


def test_parse_reset_time_iso_form_and_failure():
    got = R.parse_reset_time("resets at 2026-09-03T05:30:00Z")
    assert got is not None and got.astimezone().utcoffset() is not None
    assert R.parse_reset_time("no time here") is None


def test_wait_for_usage_reset_logs_the_wait_and_returns_seconds(tmp_path):
    """Waits are logged separately so the evaluation can exclude them from durations."""
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return _fake("usage_limit" if calls["n"] < 3 else "ok")

    waited = R.wait_for_usage_reset(text="Claude usage limit reached, try later",
                                    waits_path=tmp_path / "waits.jsonl", probe=probe,
                                    sleep=lambda _s: None, log=lambda _m: None,
                                    context={"run_id": "r"})
    rows = R.read_jsonl(tmp_path / "waits.jsonl")
    assert waited >= 0 and len(rows) == 1
    assert rows[0]["n_probes"] == 3 and rows[0]["reason"] == "probe" and rows[0]["run_id"] == "r"


def test_probe_cmd_is_cheap_and_toolless():
    cmd = R.probe_cmd()
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--max-budget-usd") + 1] == "0.02"
    assert cmd[cmd.index("--max-turns") + 1] == "1"


# --------------------------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------------------------
def test_call_log_appends_and_reads_back(tmp_path):
    log = C.CallLog(tmp_path, "run1", "B", defaults={"corpus": "dbpedia", "n": 3})
    log.archive_stdout("ingest-0001-a0", stream_stdout(RESULT_OK))
    log.append({"kind": "ingest", "doc_id": "d1", "ingest_index": 1, "status": "ok", "cost_usd": 0.1})
    log.append({"kind": "ingest", "doc_id": "d2", "ingest_index": 2, "status": "failed", "cost_usd": 0.2})
    rows = log.rows()
    assert [r["doc_id"] for r in rows] == ["d1", "d2"]
    assert rows[0]["run_id"] == "run1" and rows[0]["arm"] == "B" and rows[0]["corpus"] == "dbpedia"
    assert list(rows[0])[:4] == ["run_id", "arm", "kind", "corpus"]      # stable column order
    assert len(log.ok_rows("ingest")) == 1
    assert (tmp_path / "stdout" / "ingest-0001-a0.jsonl").exists()
    t = C.totals(rows)
    assert t["n_calls"] == 2 and t["n_ok"] == 1 and t["cost_usd"] == pytest.approx(0.3)


def test_read_jsonl_survives_a_torn_last_line(tmp_path):
    p = tmp_path / "calls.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n{"c":', encoding="utf-8")
    assert R.read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_done_line_parsing_variants():
    assert C.parse_done("DONE lint files=12")["files"] == 12
    assert C.parse_done("DONE lint files=12")["done_strict"] is True
    assert C.parse_done("chatter\n\nDONE lint files=12")["done_strict"] is False
    assert C.parse_done("preamble\nDONE topic=x new_topic=no files=3")["new_topic"] is False
    assert C.parse_done("nothing here")["done_line"] is None


# --------------------------------------------------------------------------------------------
# real captured CLI output (sanitised) from the 3-ingest verification run, 2026-09-02
# --------------------------------------------------------------------------------------------
FIXTURES = Path(__file__).parent / "fixtures" / "llm"


def test_real_stream_json_parses_end_to_end():
    stdout = (FIXTURES / "ingest_stream_json.jsonl").read_text(encoding="utf-8")
    inv = R.Invocation(cmd=[], returncode=0, stdout=stdout, stderr="", wall_ms=63000)
    res = R.parse_call(inv)
    assert res.status == "ok" and res.subtype == "success"
    assert res.num_turns == 19 and res.permission_denials == []
    # the schema asks for exactly one DONE line; this real reply carries a preamble,
    # which the parser tolerates and records as done_strict=False
    assert "DONE topic=" in res.text and not res.text.startswith("DONE")
    parsed = C.parse_done(res.text)
    assert parsed["topic"] == "film-festivals" and parsed["files"] == 9
    assert parsed["new_topic"] is True and parsed["done_strict"] is False
    assert res.tokens["cache_read"] > 0 and res.tokens["output"] > 0
    assert sorted(res.model_usage) == ["claude-sonnet-5"]
    assert res.thinking is True
    assert sum(res.tool_calls.values()) > 0
    assert R.init_event(R.stream_events(stdout))["tools"] == ["Edit", "Glob", "Grep", "Read", "Write"]
    assert R.init_event(R.stream_events(stdout))["mcp_servers"] == []


def test_claude_code_writes_its_prompt_cache_with_the_one_hour_ttl():
    """Measured, not assumed: cache creation is billed at $4.00/MTok, not $2.50.

    ``modelUsage`` carries no TTL, so pricing arm B's cache writes at the 5-minute
    rate understates the LLM baseline by ~15 % and breaks the 1 % reconciliation.
    """
    result = json.loads((FIXTURES / "ingest_result.json").read_text(encoding="utf-8"))
    split = R.cache_creation_split(result)
    assert split["cache_creation_1h"] == result["usage"]["cache_creation_input_tokens"]
    assert split["cache_creation_5m"] == 0
    assert R.cache_write_tier(result) == "1h"
    assert C.CACHE_WRITE_TIER == "1h"

    from wikiorg.eval.cost import cost_from_model_usage, load_prices
    prices = load_prices()
    reported = result["total_cost_usd"]
    at_1h = cost_from_model_usage(result["modelUsage"], prices, "1h")["total_usd"]
    at_5m = cost_from_model_usage(result["modelUsage"], prices, "5m")["total_usd"]
    assert at_1h == pytest.approx(reported, rel=1e-6), "1h reconciles exactly"
    assert abs(at_5m - reported) / reported > 0.05, "5m does not"


def test_real_row_carries_the_tier_and_the_split():
    stdout = (FIXTURES / "ingest_stream_json.jsonl").read_text(encoding="utf-8")
    res = R.parse_call(R.Invocation(cmd=[], returncode=0, stdout=stdout, stderr="", wall_ms=63000))
    row = C.row_from_call(res, kind="ingest", attempt=0, prompt="p", started_at="a", ended_at="b")
    assert row["cache_write_tier"] == "1h"
    assert row["tokens"]["cache_creation_1h"] == row["tokens"]["cache_creation"]
    assert row["usage_raw"]["service_tier"] == "standard"


def test_wait_clock_excludes_backoff_and_usage_waits_from_the_measured_wall(monkeypatch):
    """The pre-registration keeps subscription waiting out of measured durations."""
    slept = []
    monkeypatch.setattr(R.time, "sleep", lambda s: slept.append(s))
    clock = R.WaitClock()
    clock.sleep(30.0)                 # transient backoff
    clock.add(3600.0)                 # a usage-limit window
    assert slept == [30.0]
    assert clock.waited_ms == 3630_000
    assert clock.wall_ms == 0, "nothing but waiting happened, so nothing is measured"
    assert clock.add(None) == 3630.0
