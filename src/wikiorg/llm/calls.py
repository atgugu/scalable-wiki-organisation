"""``calls.jsonl`` — the per-call ledger every LLM arm writes.

One row per **attempt**, not per ingest: an attempt that ended in
``error_max_budget_usd`` really did spend up to its cap, so dropping it would
make ``sum(calls.jsonl)`` disagree with the CLI's own accounting and break the
1 % reconciliation the pre-registration requires (``configs/prereg.yaml``,
``cost_endpoint.accounting``).

The file lives *outside* the arm-B working tree (``<run_id>.meta/``) so the
agent can neither read nor perturb it, and raw CLI stdout is archived next to it
under ``stdout/`` (one file per attempt) so that any row can be re-derived.

Column contract (consumed by ``wikiorg.eval.cost`` and
``wikiorg.adapters.arm_c``)::

    run_id arm kind corpus n order_file doc_id ingest_index attempt status
    subtype is_error error model cli_version effort claude_md_sha256
    prompt_sha256 wiki_pages_before tokens{input,output,cache_read,cache_creation}
    model_usage cost_usd num_turns tool_calls duration_ms duration_api_ms
    wall_ms files_touched diff_stat git_commit permission_denials done_line
    structured_output validation thinking session_id stdout_file
    started_at ended_at
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .runner import CallResult, append_jsonl, now_iso, read_jsonl

FIELDS: tuple[str, ...] = (
    "run_id", "arm", "kind", "corpus", "n", "order_file", "doc_id", "ingest_index", "attempt",
    "status", "subtype", "is_error", "error", "model", "model_reported", "cli_version", "effort",
    "claude_md_sha256", "prompt_sha256", "prompt", "wiki_pages_before", "tokens", "model_usage",
    "cache_write_tier", "usage_raw", "cost_usd", "num_turns", "tool_calls", "duration_ms",
    "duration_api_ms", "wall_ms", "waited_ms",
    "files_touched", "diff_stat", "git_commit", "permission_denials", "done_line",
    "structured_output", "validation", "thinking", "done_strict", "session_id", "budget_usd", "max_turns",
    "timeout_s", "stdout_file", "started_at", "ended_at",
)
OK_STATUSES = ("ok", "success")

#: Measured on CLI 2.1.258 (2026-09-02): every cache-creation token is billed at the
#: 1-hour rate.  Recomputing arm-B cost at the 5-minute rate understates it by ~15 %.
CACHE_WRITE_TIER = "1h"


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def done_line(text: str) -> str | None:
    """The schema requires the reply to be exactly one ``DONE ...`` line; take the
    last line that starts with ``DONE`` so a stray preamble still parses."""
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("DONE"):
            return line
    return None


def parse_done(text: str) -> dict[str, Any]:
    """``DONE topic=x new_topic=yes files=7`` -> ``{"topic": "x", ...}`` plus the
    raw line and whether it was well formed."""
    line = done_line(text)
    body = (text or "").strip()
    out: dict[str, Any] = {"done_line": line,
                           # the schema says "Reply with exactly one line"; measured runs show the
                           # agent sometimes prefixes a sentence, so record compliance as data
                           "done_strict": bool(line) and body == line}
    if not line:
        return out
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, _, val = token.partition("=")
        key = key.strip().lower()
        val = val.strip().strip(",;")
        if key in ("files", "ingest"):
            try:
                out[key] = int(val)
            except ValueError:
                out[key] = val
        elif key in ("new_topic",):
            out[key] = val.lower() in ("yes", "true", "1")
        else:
            out[key] = val
    return out


class CallLog:
    """Append-only ledger plus the stdout archive."""

    def __init__(self, meta_dir: str | Path, run_id: str, arm: str, defaults: Mapping[str, Any] | None = None):
        self.meta_dir = Path(meta_dir)
        self.path = self.meta_dir / "calls.jsonl"
        self.stdout_dir = self.meta_dir / "stdout"
        self.waits_path = self.meta_dir / "waits.jsonl"
        self.decisions_path = self.meta_dir / "decisions.jsonl"
        self.run_id = run_id
        self.arm = arm
        self.defaults = dict(defaults or {})
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_dir.mkdir(parents=True, exist_ok=True)

    # -- reading ------------------------------------------------------------------------------
    def rows(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path)

    def ok_rows(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.rows()
                if str(r.get("status")) in OK_STATUSES and (kind is None or r.get("kind") == kind)]

    # -- writing ------------------------------------------------------------------------------
    def archive_stdout(self, name: str, stdout: str, stderr: str = "") -> str:
        """Write the raw CLI output before anything else can fail; return the file name."""
        p = self.stdout_dir / f"{name}.jsonl"
        text = stdout or ""
        if stderr:
            text += "\n#stderr# " + stderr.replace("\n", "\n#stderr# ")
        p.write_text(text, encoding="utf-8")
        return p.name

    def append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        full = {"run_id": self.run_id, "arm": self.arm}
        full.update(self.defaults)
        full.update({k: v for k, v in row.items()})
        ordered = {k: full.get(k) for k in FIELDS if k in full}
        ordered.update({k: v for k, v in full.items() if k not in FIELDS})
        append_jsonl(self.path, ordered)
        return ordered

    def decision(self, row: Mapping[str, Any]) -> dict[str, Any]:
        out = {"run_id": self.run_id, "arm": self.arm, "at": now_iso(), **dict(row)}
        append_jsonl(self.decisions_path, out)
        return out


def row_from_call(res: CallResult, *, kind: str, attempt: int, prompt: str,
                  started_at: str, ended_at: str, stdout_file: str | None = None,
                  **extra: Any) -> dict[str, Any]:
    """Everything a :class:`~wikiorg.llm.runner.CallResult` contributes to a row."""
    row: dict[str, Any] = {
        "kind": kind,
        "attempt": attempt,
        "status": res.status,
        "subtype": res.subtype,
        "is_error": bool(res.is_error),
        "error": res.error,
        "model_reported": res.model,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "tokens": {**(dict(res.tokens) if res.tokens else
                      {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}),
                   **res.cache_split},
        "model_usage": res.model_usage or None,
        # Claude Code writes its prompt cache with the 1-hour TTL ($4.00/MTok, not $2.50):
        # wikiorg.eval.cost must price these rows with cache_write_tier="1h".
        "cache_write_tier": res.cache_write_tier,
        "usage_raw": res.usage_raw or None,
        "cost_usd": res.cost_usd,
        "num_turns": res.num_turns,
        "tool_calls": res.tool_calls or {},
        "duration_ms": res.duration_ms,
        "duration_api_ms": res.duration_api_ms,
        "wall_ms": res.wall_ms,
        "permission_denials": list(res.permission_denials or []),
        "structured_output": res.structured_output,
        "thinking": bool(res.thinking),
        "session_id": res.session_id,
        "stdout_file": stdout_file,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    row.update(parse_done(res.text))
    row.update(extra)
    return row


def totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Cheap in-harness summary (``wikiorg.eval.cost`` is the authoritative one)."""
    rows = list(rows)
    usd = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    tok = {k: 0 for k in ("input", "output", "cache_read", "cache_creation")}
    for r in rows:
        t = r.get("tokens") or {}
        for k in tok:
            tok[k] += int(t.get(k) or 0)
    ok = [r for r in rows if str(r.get("status")) in OK_STATUSES]
    return {
        "n_calls": len(rows), "n_ok": len(ok), "cost_usd": round(usd, 6), "tokens": tok,
        "num_turns": sum(int(r.get("num_turns") or 0) for r in rows),
        "wall_ms": sum(int(r.get("wall_ms") or 0) for r in rows),
        "by_status": {s: sum(1 for r in rows if r.get("status") == s) for s in sorted({str(r.get("status")) for r in rows})},
    }


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return p
