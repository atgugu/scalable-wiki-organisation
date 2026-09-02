"""Claude CLI harness for the LLM arms (B, C, A+).

Fork of ``debater/src/debater/claude_runner.py`` with the additions the thesis
needs: a pinned, neutralised environment, ``modelUsage`` accounting (the
top-level ``usage`` undercounts because it omits subagents), stream-json
parsing, an error classifier with a usage-limit waiter, and retries.

Everything measured must be reproducible outside Claude Code, so every call
pins the model and the effort level explicitly (``configs/prereg.yaml``:
``claude-sonnet-5`` / ``high``, the API default, never this machine's ``xhigh``)
and neutralises the user's settings, MCP servers, plugins, hooks and slash
commands.

Empirically verified on 2026-09-02 with CLI 2.1.258 (``configs/cli_facts.json``):

* ``--tools ""`` -> the ``init`` event lists zero tools;
* ``--tools "Read,Write,Edit,Glob,Grep"`` -> exactly those five;
* ``--max-turns N`` is accepted and yields ``subtype=error_max_turns``;
* ``--max-budget-usd X`` yields ``subtype=error_max_budget_usd``;
* ``--setting-sources project --strict-mcp-config --disable-slash-commands
  --no-session-persistence`` -> 0 MCP servers, 0 slash commands, the user's
  SessionStart hook does not fire;
* ``--permission-mode acceptEdits`` + ``--allowedTools`` writes files with an
  empty ``permission_denials``;
* ``--json-schema`` returns a valid ``structured_output``.

``classify`` returns one of ``ok | transient | usage_limit | permanent |
budget | turns | timeout | interrupted``.  The last two are not CLI subtypes:
``timeout`` is our own kill, ``interrupted`` is a negative return code (the
child died from a signal, i.e. the operator pressed Ctrl-C) which must never be
retried -- the run is resumable, so exiting cleanly is the correct response.
"""
from __future__ import annotations

import json
import os
import random
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------------------------
# pinned configuration (configs/prereg.yaml)
# --------------------------------------------------------------------------------------------
MODEL = "claude-sonnet-5"
EFFORT = "high"
CLI_VERSION_EXPECTED = "2.1.258"

#: Tools arm B is allowed to use.  ``MultiEdit`` is absent: the smoke run showed
#: ``--tools`` resolves to exactly {Read, Write, Edit, Glob, Grep}.
TOOLS_B = "Read,Write,Edit,Glob,Grep"
DISALLOWED_B = "Bash,WebFetch,WebSearch,Task,NotebookEdit,TodoWrite,Skill,ToolSearch"

#: Environment pins.  ``build_env`` also *removes* variables that would change
#: billing or make the CLI behave like a nested session, so that a run launched
#: from a bare terminal is byte-identical to one launched from inside Claude Code.
ENV_PINS = {
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_BUG_COMMAND": "1",
}
ENV_DROP = (
    "ANTHROPIC_API_KEY",        # would bill the API instead of the OAuth subscription
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "CLAUDECODE",               # nested-session markers set by the parent session
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_ACTION",
    "CLAUDE_AGENT_SDK",
    "CLAUDE_CONFIG_DIR",
    "MAX_THINKING_TOKENS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
)

USAGE_LIMIT_MARKERS = (
    "usage limit",
    "hit your limit",
    "weekly usage limit",
    "resets at",
    "reset at",
    "upgrade to increase",
    "you've reached your",
)
TRANSIENT_MARKERS = (
    "rate_limit_error",
    "overloaded_error",
    "overloaded",
    "api_error",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "econnreset",
    "etimedout",
    "enotfound",
    "socket hang up",
    "fetch failed",
    "network error",
    "connection error",
    "stream error",
    "terminated",
    "429",
    "529",
)
PERMANENT_MARKERS = (
    "authentication_error",
    "invalid_request_error",
    "permission_error",
    "not_found_error",
    "prompt is too long",
    "credit balance",
    "invalid api key",
    "invalid bearer token",
    "please run /login",
    "model not found",
)

TRANSIENT_STATUS = (408, 409, 425, 429, 529)
_TOKEN_KEYS = {
    "input": ("inputTokens", "input_tokens"),
    "output": ("outputTokens", "output_tokens"),
    "cache_read": ("cacheReadInputTokens", "cache_read_input_tokens"),
    "cache_creation": ("cacheCreationInputTokens", "cache_creation_input_tokens"),
}


class HarnessError(RuntimeError):
    """Unrecoverable misconfiguration: abort the run rather than burn money."""


# --------------------------------------------------------------------------------------------
# environment and flags
# --------------------------------------------------------------------------------------------
def build_env(extra: dict[str, str] | None = None, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment every measured call runs in.

    Starts from the current environment (the CLI needs ``HOME``/``PATH`` and the
    OAuth credentials it stores under ``~/.claude``), applies :data:`ENV_PINS`
    and removes :data:`ENV_DROP` so that a call made from inside a Claude Code
    session is indistinguishable from one made in a plain terminal.
    """
    env = dict(os.environ if base is None else base)
    for key in ENV_DROP:
        env.pop(key, None)
    env.update(ENV_PINS)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def common_flags(model: str = MODEL, effort: str = EFFORT, budget: float | None = None,
                 max_turns: int | None = None) -> list[str]:
    """Flags shared by every measured call (never ``--fallback-model``)."""
    flags = [
        "--model", model,
        "--effort", effort,
        "--setting-sources", "project",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if budget is not None:
        flags += ["--max-budget-usd", f"{float(budget):.4f}".rstrip("0").rstrip(".")]
    if max_turns is not None:
        flags += ["--max-turns", str(int(max_turns))]
    return flags


def cli_version(env: dict[str, str] | None = None) -> str:
    out = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=60,
                         env=env or build_env())
    m = re.search(r"(\d+\.\d+\.\d+)", out.stdout or "")
    return m.group(1) if m else (out.stdout or "").strip()


def assert_cli_version(expected: str = CLI_VERSION_EXPECTED, env: dict[str, str] | None = None,
                       strict: bool = False) -> str:
    """The pre-registration pins the CLI version; a silent upgrade changes the
    system prompt and therefore the cached prefix and the cost."""
    got = cli_version(env)
    if got != expected:
        msg = f"claude CLI version {got!r} != pre-registered {expected!r}"
        if strict:
            raise HarnessError(msg)
        print(f"WARNING: {msg} (recorded in every call row)", flush=True)
    return got


# --------------------------------------------------------------------------------------------
# subprocess
# --------------------------------------------------------------------------------------------
@dataclass
class Invocation:
    """Raw outcome of one ``claude -p`` subprocess."""

    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    wall_ms: int
    timed_out: bool = False

    @property
    def shell(self) -> str:
        return " ".join(shlex.quote(c) for c in self.cmd)


def run_claude(cmd: Sequence[str], stdin: str | None = None, timeout: float = 1200,
               cwd: str | Path | None = None, env: dict[str, str] | None = None) -> Invocation:
    """Run the CLI once.  Never raises on a non-zero exit or a timeout: the
    partial stdout of a killed child is still worth archiving."""
    cmd = [str(c) for c in cmd]
    env = env if env is not None else build_env()
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=str(cwd) if cwd else None, env=env)
    timed_out = False
    try:
        out, err = proc.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - the kill did not take
            out, err = "", "killed"
    except BaseException:
        # Ctrl-C or SIGTERM. Without this the child survives the parent and keeps
        # writing into the arm-B work tree while the next process resumes the run.
        proc.kill()
        try:
            proc.communicate(timeout=30)
        except Exception:                  # pragma: no cover - best effort
            pass
        raise
    wall_ms = int(round((time.monotonic() - t0) * 1000))
    return Invocation(cmd=list(cmd), returncode=proc.returncode, stdout=out or "", stderr=err or "",
                      wall_ms=wall_ms, timed_out=timed_out)


# --------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------
def stream_events(stdout: str) -> list[dict[str, Any]]:
    """Every JSON object on its own line (``--output-format stream-json``)."""
    events: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def parse_result(stdout: str) -> dict[str, Any] | None:
    """The ``result`` object from either output format.

    ``--output-format json`` prints one object; ``--output-format stream-json``
    prints one object per line and the *last* ``{"type": "result"}`` line is the
    one that carries ``total_cost_usd`` / ``modelUsage`` / ``num_turns``.
    """
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and (obj.get("type") == "result" or "total_cost_usd" in obj
                                      or "subtype" in obj or "result" in obj):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    for obj in reversed(stream_events(text)):
        if obj.get("type") == "result":
            return obj
    return None


def usage_from_model_usage(result: dict[str, Any] | None, require_single_sonnet: bool = True,
                           model: str | None = None,
                           ) -> tuple[dict[str, int], dict[str, Any]]:
    """``(tokens, raw modelUsage)`` -- ``modelUsage`` preferred over ``usage``.

    ``modelUsage`` includes subagent traffic, which the top-level ``usage`` omits;
    the thesis reports list-price dollars, so undercounting is not acceptable.
    Tokens are summed over every model that appears (a stray Haiku entry from a
    CLI-internal call must still be paid for) while
    ``require_single_sonnet`` asserts the pre-registered pin held: exactly one
    Sonnet model served the call, so ``--fallback-model`` or a settings override
    never silently changed the arm.
    """
    result = result or {}
    mu = result.get("modelUsage") or result.get("model_usage") or {}
    if not isinstance(mu, dict):
        mu = {}
    if mu:
        # The pin check is model-family aware: "sonnet" for the v1/v2 pin, "haiku" for prereg v3.
        family = next((f for f in ("haiku", "sonnet", "opus", "fable") if f in str(model or MODEL).lower()), "sonnet")
        matched = [k for k in mu if family in str(k).lower()]
        if require_single_sonnet and len(matched) != 1:
            raise HarnessError(
                f"expected exactly one {family} key in modelUsage, got {sorted(mu)}")
        tokens = {k: 0 for k in _TOKEN_KEYS}
        for entry in mu.values():
            if not isinstance(entry, dict):
                continue
            for key, aliases in _TOKEN_KEYS.items():
                for alias in aliases:
                    if entry.get(alias) is not None:
                        tokens[key] += int(entry[alias] or 0)
                        break
        return tokens, mu
    usage = result.get("usage") or {}
    tokens = {
        "input": int(usage.get("input_tokens", 0) or 0),
        "output": int(usage.get("output_tokens", 0) or 0),
        "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation": int(usage.get("cache_creation_input_tokens", 0) or 0),
    }
    return tokens, {}


def cache_creation_split(result: dict[str, Any] | None) -> dict[str, int]:
    """``{"cache_creation_5m": n, "cache_creation_1h": n}`` from the result's ``usage``.

    Measured 2026-09-02 on CLI 2.1.258: Claude Code writes its prompt cache with the
    **one-hour** TTL (``usage.cache_creation.ephemeral_1h_input_tokens`` carries every
    cache-creation token, ``ephemeral_5m_input_tokens`` is 0), which is billed at
    $4.00/MTok rather than $2.50.  ``modelUsage`` does not carry the split, so it is
    recorded here: pricing arm B's cache writes at the 5-minute rate understates the
    LLM baseline by about 15 % and breaks the 1 % reconciliation the pre-registration
    requires.
    """
    usage = (result or {}).get("usage") or {}
    cc = usage.get("cache_creation") or usage.get("cacheCreation") or {}
    if not isinstance(cc, dict):
        return {}
    out = {}
    for key, aliases in (("cache_creation_5m", ("ephemeral_5m_input_tokens", "ephemeral5mInputTokens")),
                         ("cache_creation_1h", ("ephemeral_1h_input_tokens", "ephemeral1hInputTokens"))):
        for alias in aliases:
            if cc.get(alias) is not None:
                out[key] = int(cc[alias] or 0)
                break
    return out


def cache_write_tier(result: dict[str, Any] | None, default: str = "1h") -> str:
    """Which cache-write price applies to this call (``"1h"`` for Claude Code)."""
    split = cache_creation_split(result)
    if not split:
        return default
    if split.get("cache_creation_1h", 0) and not split.get("cache_creation_5m", 0):
        return "1h"
    if split.get("cache_creation_5m", 0) and not split.get("cache_creation_1h", 0):
        return "5m"
    return "mixed" if any(split.values()) else default


def tool_calls_from_events(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    """``{tool_name: n}`` counted from ``tool_use`` blocks in assistant messages."""
    counts: dict[str, int] = {}
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message") or {}
        for block in (msg.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = str(block.get("name") or "?")
                counts[name] = counts.get(name, 0) + 1
    return counts


def has_thinking(events: Iterable[dict[str, Any]]) -> bool:
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        for block in ((ev.get("message") or {}).get("content") or []):
            if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
                return True
    return False


def init_event(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for ev in events:
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            return ev
    return None


# --------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------
def _api_status(result: dict[str, Any] | None, blob: str) -> int | None:
    if isinstance(result, dict):
        for key in ("api_error_status", "status", "statusCode"):
            v = result.get(key)
            try:
                return int(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    m = re.search(r"api[_ ]error[_ ]status[\"']?\s*[:=]\s*[\"']?(\d{3})", blob)
    return int(m.group(1)) if m else None


def classify(result: dict[str, Any] | None, invocation: Invocation | None = None,
             stdout: str = "", stderr: str = "") -> str:
    """``ok | transient | usage_limit | permanent | budget | turns | timeout | interrupted``.

    The subtype wins when the CLI produced a result object; otherwise the
    decision is made from the text of stdout/stderr.  Usage limits are checked
    before generic transients because a usage-limit message often also mentions
    a rate limit, and the two need different handling (wait for the reset vs
    exponential backoff).
    """
    if invocation is not None:
        if invocation.timed_out:
            return "timeout"
        if invocation.returncode is not None and invocation.returncode < 0:
            return "interrupted"
        stdout = stdout or invocation.stdout
        stderr = stderr or invocation.stderr
    if result is None:
        result = parse_result(stdout)
    blob = f"{stdout}\n{stderr}".lower()

    if isinstance(result, dict):
        subtype = str(result.get("subtype") or "")
        if subtype == "error_max_budget_usd":
            return "budget"
        if subtype == "error_max_turns":
            return "turns"
        if subtype == "success" and not result.get("is_error"):
            return "ok"
        if subtype == "error_max_structured_output_retries":
            return "permanent"

    if any(m in blob for m in USAGE_LIMIT_MARKERS):
        return "usage_limit"
    status = _api_status(result, blob)
    if status is not None:
        return "transient" if (status in TRANSIENT_STATUS or status >= 500) else "permanent"
    if any(m in blob for m in PERMANENT_MARKERS):
        return "permanent"
    if any(m in blob for m in TRANSIENT_MARKERS):
        return "transient"
    if not (stdout or "").strip():
        return "transient"          # empty stdout: the CLI died before saying anything
    if isinstance(result, dict) and result.get("is_error"):
        return "transient"
    return "transient"


# --------------------------------------------------------------------------------------------
# a fully parsed call
# --------------------------------------------------------------------------------------------
@dataclass
class CallResult:
    invocation: Invocation
    result: dict[str, Any] | None
    status: str
    subtype: str | None = None
    is_error: bool = False
    error: str | None = None
    text: str = ""
    tokens: dict[str, int] = field(default_factory=dict)
    model_usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    permission_denials: list = field(default_factory=list)
    structured_output: Any = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    thinking: bool = False
    session_id: str | None = None
    model: str | None = None
    n_events: int = 0
    usage_raw: dict[str, Any] = field(default_factory=dict)
    cache_split: dict[str, int] = field(default_factory=dict)
    cache_write_tier: str = "1h"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def wall_ms(self) -> int:
        return self.invocation.wall_ms

    @property
    def stdout(self) -> str:
        return self.invocation.stdout


def _model_from_cmd(cmd) -> str | None:
    """The ``--model X`` value of an invocation's command line, if any (used for the pin check)."""
    try:
        parts = list(cmd) if not isinstance(cmd, str) else cmd.split()
    except TypeError:
        return None
    for i, tok in enumerate(parts):
        if tok == "--model" and i + 1 < len(parts):
            return str(parts[i + 1])
        if isinstance(tok, str) and tok.startswith("--model="):
            return tok.split("=", 1)[1]
    return None


def parse_call(inv: Invocation, require_single_sonnet: bool = True) -> CallResult:
    """Turn a raw invocation into the record ``calls.jsonl`` stores."""
    events = stream_events(inv.stdout)
    result = parse_result(inv.stdout)
    status = classify(result, inv)
    tokens, mu = ({}, {})
    cmd_model = _model_from_cmd(getattr(inv, "cmd", None) or getattr(inv, "argv", None) or getattr(inv, "command", None))
    if result is not None:
        try:
            tokens, mu = usage_from_model_usage(result, require_single_sonnet=require_single_sonnet, model=cmd_model)
        except HarnessError:
            if status == "ok":
                raise
            tokens, mu = usage_from_model_usage(result, require_single_sonnet=False, model=cmd_model)
    err = None
    if status != "ok":
        err = _error_text(inv, result)
    init = init_event(events) or {}
    return CallResult(
        invocation=inv, result=result, status=status,
        subtype=(result or {}).get("subtype"),
        is_error=bool((result or {}).get("is_error")),
        error=err,
        text=str((result or {}).get("result") or "") if isinstance((result or {}).get("result"), str) else "",
        tokens=tokens, model_usage=mu,
        cost_usd=(float(result["total_cost_usd"]) if result and result.get("total_cost_usd") is not None else None),
        num_turns=(int(result["num_turns"]) if result and result.get("num_turns") is not None else None),
        duration_ms=(int(result["duration_ms"]) if result and result.get("duration_ms") is not None else None),
        duration_api_ms=(int(result["duration_api_ms"]) if result and result.get("duration_api_ms") is not None else None),
        permission_denials=list((result or {}).get("permission_denials") or []),
        structured_output=(result or {}).get("structured_output"),
        tool_calls=tool_calls_from_events(events),
        thinking=has_thinking(events),
        session_id=(result or {}).get("session_id") or init.get("session_id"),
        model=init.get("model") or (result or {}).get("model"),
        n_events=len(events),
        usage_raw=(result or {}).get("usage") or {},
        cache_split=cache_creation_split(result),
        cache_write_tier=cache_write_tier(result),
    )


def _error_text(inv: Invocation, result: dict[str, Any] | None) -> str:
    if inv.timed_out:
        return f"timeout after {inv.wall_ms / 1000:.0f}s"
    if isinstance(result, dict):
        for key in ("error", "api_error_status", "subtype", "result"):
            val = result.get(key)
            if val:
                return f"{key}={val}"[:600]
    return ((inv.stderr or "").strip() or (inv.stdout or "").strip() or f"exit {inv.returncode}")[:600]


# --------------------------------------------------------------------------------------------
# retries and usage-limit waiting
# --------------------------------------------------------------------------------------------
TERMINAL_STATUSES = ("ok", "budget", "turns", "timeout", "permanent", "interrupted")


class WaitClock:
    """Wall-clock for one logical call, with waiting excluded.

    Retry backoff and usage-limit waits are not part of what an ingest *costs* in
    time -- they are properties of the subscription, not of the wiki -- so the
    pre-registration keeps them out of the measured durations and in
    ``waits.jsonl``.  Calendar projections use real elapsed time instead.
    """

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.waited = 0.0

    def sleep(self, seconds: float) -> None:
        self.waited += float(seconds or 0.0)
        time.sleep(seconds)

    def add(self, seconds: float | None) -> float:
        self.waited += float(seconds or 0.0)
        return self.waited

    @property
    def wall_ms(self) -> int:
        return max(0, int(round((time.monotonic() - self.t0 - self.waited) * 1000)))

    @property
    def waited_ms(self) -> int:
        return int(round(self.waited * 1000))


def backoff_seconds(attempt: int, base: float = 30.0, jitter: float = 30.0,
                    rng: random.Random | None = None) -> float:
    """``30 * 2**k + U(0, 30)`` seconds (pre-registered)."""
    r = rng or random
    return base * (2 ** attempt) + r.uniform(0.0, jitter)


def with_retries(call: Callable[[int], CallResult], attempts: int = 5,
                 on_wait: Callable[[CallResult], float] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print,
                 rng: random.Random | None = None) -> CallResult:
    """Run ``call(attempt_index)`` until it stops being transient.

    Only ``transient`` consumes an attempt.  ``usage_limit`` calls ``on_wait``
    (normally :func:`wait_for_usage_reset`) and retries without consuming one --
    a run spanning days will hit the subscription's 5-hour and weekly caps many
    times and must survive all of them.  Everything in
    :data:`TERMINAL_STATUSES` returns immediately.
    """
    res: CallResult | None = None
    attempt = 0
    used = 0
    while True:
        res = call(attempt)
        if res.status in TERMINAL_STATUSES:
            return res
        if res.status == "usage_limit":
            log(f"usage limit hit: {(res.error or '')[:200]}")
            if on_wait is None:
                return res
            on_wait(res)
            attempt += 1
            continue
        used += 1
        attempt += 1
        if used >= attempts:
            log(f"giving up after {used} transient attempts: {(res.error or '')[:200]}")
            return res
        delay = backoff_seconds(used - 1, rng=rng)
        log(f"transient failure ({(res.error or '')[:120]}); retry {used}/{attempts - 1} in {delay:.0f}s")
        sleep(delay)


_UNIX_TS_RE = re.compile(r"\|\s*(\d{10})\b")
_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?\s*(Z|[+-]\d{2}:?\d{2})?")
_CLOCK_RE = re.compile(r"reset[s]?\s+(?:at|on)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.I)


def parse_reset_time(text: str, now: datetime | None = None) -> datetime | None:
    """Best-effort reset instant from a usage-limit message.

    Handles the CLI's ``Claude AI usage limit reached|<unix seconds>`` form, an
    embedded ISO timestamp, and the human ``resets at 3pm`` form (interpreted in
    local time, rolled to tomorrow when already past).  Returns ``None`` when
    nothing parses, in which case the caller falls back to probing.
    """
    text = text or ""
    now = now or datetime.now().astimezone()
    m = _UNIX_TS_RE.search(text)
    if m:
        try:
            return datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc).astimezone()
        except (OverflowError, OSError, ValueError):
            pass
    m = _ISO_RE.search(text)
    if m:
        try:
            date, hh, mm, ss, tz = m.groups()
            stamp = f"{date}T{hh}:{mm}:{ss or '00'}"
            dt = datetime.fromisoformat(stamp)
            if tz:
                dt = datetime.fromisoformat(stamp + ("+00:00" if tz == "Z" else tz))
            else:
                dt = dt.replace(tzinfo=now.tzinfo)
            return dt.astimezone()
        except ValueError:
            pass
    m = _CLOCK_RE.search(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23:
            cand = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if cand <= now:
                cand += timedelta(days=1)
            return cand
    return None


def probe_cmd(model: str = MODEL, effort: str = EFFORT) -> list[str]:
    """The cheapest possible call: no tools, a 2-cent ceiling, one turn."""
    return (["claude", "-p", "Reply with the single word OK.", "--output-format", "json", "--tools", ""]
            + common_flags(model=model, effort=effort, budget=0.02, max_turns=1))


def wait_for_usage_reset(res: CallResult | None = None, text: str | None = None,
                         waits_path: str | Path | None = None, probe_interval: float = 900.0,
                         max_wait: float = 7 * 24 * 3600.0, env: dict[str, str] | None = None,
                         cwd: str | Path | None = None, sleep: Callable[[float], None] = time.sleep,
                         now: Callable[[], datetime] | None = None,
                         probe: Callable[[], CallResult] | None = None,
                         log: Callable[[str], None] = print,
                         context: dict[str, Any] | None = None) -> float:
    """Block until the subscription's usage window reopens; return seconds waited.

    If the error text names a reset instant, sleep until just after it and then
    probe once.  Otherwise probe every ``probe_interval`` seconds with a
    ``--tools "" --max-budget-usd 0.02`` call.  Every wait is appended to
    ``waits.jsonl`` so the evaluation can subtract it: waiting for a
    subscription window is not part of the measured wall-clock of an ingest.
    """
    clock = now or (lambda: datetime.now().astimezone())
    blob = text if text is not None else " ".join(filter(None, [
        (res.error or "") if res else "", (res.stdout or "") if res else "",
        (res.invocation.stderr or "") if res else ""]))
    started = clock()
    t0 = time.monotonic()
    reset_at = parse_reset_time(blob, started)
    n_probes = 0
    reason = "reset_time" if reset_at else "probe"
    if reset_at is not None:
        delta = (reset_at - started).total_seconds() + 60.0
        if 0 < delta <= max_wait:
            log(f"usage limit: sleeping {delta / 60:.0f} min until {reset_at.isoformat()}")
            sleep(delta)
        else:
            reset_at, reason = None, "probe"

    def _probe() -> CallResult:
        inv = run_claude(probe_cmd(), timeout=180, cwd=cwd, env=env or build_env())
        return parse_call(inv, require_single_sonnet=False)

    runner = probe or _probe
    while time.monotonic() - t0 < max_wait:
        n_probes += 1
        pres = runner()
        if pres.status not in ("usage_limit",):
            break
        log(f"usage limit still in force after {(time.monotonic() - t0) / 60:.0f} min; "
            f"probing again in {probe_interval / 60:.0f} min")
        sleep(probe_interval)
    waited = time.monotonic() - t0
    ended = clock()
    if waits_path:
        row = {"kind": "usage_limit_wait", "reason": reason,
               "reset_at": reset_at.isoformat() if reset_at else None,
               "started_at": started.isoformat(), "ended_at": ended.isoformat(),
               "seconds": round(waited, 1), "n_probes": n_probes,
               "message": (blob or "")[:500]}
        row.update(context or {})
        append_jsonl(waits_path, row)
    log(f"usage limit cleared after {waited / 60:.1f} min ({n_probes} probes)")
    return waited


# --------------------------------------------------------------------------------------------
# small io helpers shared by the arms
# --------------------------------------------------------------------------------------------
def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON line and fsync: the run must survive a hard kill."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue          # a torn last line from a kill: ignore it
    return rows


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clean_argv(argv: Sequence[str] | None) -> list[str] | None:
    """Drop the ``--`` separators ``wikiorg.cli`` leaves in front of a
    ``argparse.REMAINDER`` hand-off (``wikiorg run-b -- --run-id ...``)."""
    if argv is None:
        return None
    out = list(argv)
    while out and out[0] == "--":
        out.pop(0)
    return out
