"""Arm B — the faithful Karpathy incremental-ingest baseline.

One ``claude -p`` call per source document, in a working directory that holds
only the frozen schema (``CLAUDE.md``), the sources ingested so far (``raw/``)
and the wiki the agent has built (``wiki/``).  The agent is its own editor: the
harness never writes into ``wiki/``.

Design points that the days-long run depends on:

* **The agent never sees ``.git``.**  The repository lives at
  ``<runs_root>/<run_id>.git`` with ``core.worktree`` pointing back at the run
  directory, so ``Glob`` and ``Read`` cannot pick up harness state and the
  prompt prefix stays byte-stable for prompt caching.
* **One commit per ingest** (``ingest NNNN <doc_id>``) which also carries the
  freshly copied ``raw/<doc_id>.md``.  That yields files-touched and a diff stat
  for free, gives a resume point, and makes ``git tag`` / ``git archive`` the
  snapshot mechanism.
* **A source becomes visible only immediately before its own ingest**, so the
  agent can never read ahead in the corpus.
* **Resume is the default.**  On start the harness reads ``calls.jsonl``,
  reconciles it with the commit at ``HEAD``, resets the work tree and continues
  at the next order position.  A kill between the commit and the ledger append
  is recovered from the archived stdout rather than aborting the run.
* **Pre-registered stop rule** (``configs/prereg.yaml``, ``guards.arm_b_stop_rule``):
  after 20 ingests the harness fits cost against wiki size, projects both
  dollars and a calendar completion date, prints the projection prominently and
  stops the order at N=200 if it exceeds $400 or 2026-10-15.

Exact invocation (Appendix C of the plan; every flag empirically verified with
CLI 2.1.258 on 2026-09-02)::

    claude -p "Ingest raw/<doc_id>.md following CLAUDE.md." \\
      --model claude-sonnet-5 --effort high --output-format stream-json --verbose \\
      --tools "Read,Write,Edit,Glob,Grep" --allowedTools "Read,Write,Edit,Glob,Grep" \\
      --disallowedTools "Bash,WebFetch,WebSearch,Task,NotebookEdit,TodoWrite,Skill,ToolSearch" \\
      --permission-mode acceptEdits --setting-sources project --strict-mcp-config \\
      --disable-slash-commands --no-session-persistence --max-budget-usd <cap> --max-turns 80
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..data.docs import parse_doc, read_order, sha256_file
from . import runner as R
from .calls import CallLog, row_from_call, totals, write_json

REPO = Path(__file__).resolve().parents[3]
RUNS_ROOT = Path(os.environ.get("WIKIORG_RUNS_ROOT", str(REPO / "runs")))
SCHEMA_PATH = REPO / "src" / "wikiorg" / "llm" / "schema" / "CLAUDE.md"

INGEST_PROMPT = "Ingest raw/{doc_id}.md following CLAUDE.md."
LINT_PROMPT = "Run the lint procedure in CLAUDE.md."

INGEST_TIMEOUT = 1200
INGEST_MAX_TURNS = 80
LINT_TIMEOUT = 1800
LINT_MAX_TURNS = 200
LINT_BUDGET = 6.0
RETRY_FACTOR = 1.5

BUDGET_MIN, BUDGET_MAX, BUDGET_MULT, BUDGET_WINDOW, BUDGET_DEFAULT = 1.5, 4.0, 4.0, 20, 2.0

STOP_RULE_AFTER = 20
STOP_RULE_USD = 400.0
STOP_RULE_DEADLINE = "2026-10-15"
STOP_RULE_N = 200

PAGE_DIRS = ("sources", "topics", "entities", "concepts")
_HEAD_RE = re.compile(r"^(ingest|lint)\s+(\d+)(?:\s+(\S+))?")


# =================================================================================================
# git
# =================================================================================================
class GitStore:
    """A repository whose work tree is the agent's cwd but whose ``.git`` is not.

    ``git init --bare`` + ``core.bare=false`` + ``core.worktree`` is the standard
    way to get a repository with no ``.git`` entry inside the work tree; every
    command additionally passes ``--work-tree`` so behaviour never depends on the
    reader's cwd.
    """

    def __init__(self, workdir: str | Path, gitdir: str | Path):
        self.workdir = Path(workdir)
        self.gitdir = Path(gitdir)

    # -- plumbing ---------------------------------------------------------------------------
    def git(self, *args: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        cmd = ["git", f"--git-dir={self.gitdir}", f"--work-tree={self.workdir}", *[str(a) for a in args]]
        env = dict(os.environ)
        env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_AUTHOR_NAME": "wikiorg", "GIT_AUTHOR_EMAIL": "wikiorg@localhost",
                    "GIT_COMMITTER_NAME": "wikiorg", "GIT_COMMITTER_EMAIL": "wikiorg@localhost"})
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if check and res.returncode != 0:
            raise R.HarnessError(f"git {' '.join(map(str, args))} failed: {res.stderr.strip()[:400]}")
        return res

    @property
    def exists(self) -> bool:
        return (self.gitdir / "HEAD").exists()

    def init(self) -> None:
        if self.exists:
            return
        self.gitdir.parent.mkdir(parents=True, exist_ok=True)
        self.workdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--bare", "-b", "main", str(self.gitdir)],
                       capture_output=True, text=True, check=True)
        for key, val in (("core.bare", "false"), ("core.worktree", str(self.workdir)),
                         ("user.name", "wikiorg"), ("user.email", "wikiorg@localhost"),
                         ("commit.gpgsign", "false"), ("gc.auto", "0"), ("core.autocrlf", "false")):
            self.git("config", key, val)

    # -- state ------------------------------------------------------------------------------
    def head_sha(self) -> str | None:
        res = self.git("rev-parse", "HEAD", check=False)
        return res.stdout.strip() or None if res.returncode == 0 else None

    def head_message(self) -> str | None:
        res = self.git("log", "-1", "--pretty=%s", check=False)
        return res.stdout.strip() or None if res.returncode == 0 else None

    def porcelain(self) -> list[tuple[str, str]]:
        """``[(XY, path)]`` from ``git status --porcelain`` (renames give the new path)."""
        out = self.git("status", "--porcelain", "-z").stdout
        parts = [p for p in out.split("\0") if p]
        rows: list[tuple[str, str]] = []
        i = 0
        while i < len(parts):
            entry = parts[i]
            code, _, path = entry[:2], entry[2:3], entry[3:]
            if code and code[0] in ("R", "C"):
                i += 1                     # the following NUL-separated field is the source path
            rows.append((code, path))
            i += 1
        return rows

    def numstat(self) -> dict[str, int]:
        out = self.git("diff", "--cached", "--numstat").stdout
        files = ins = dels = 0
        for line in out.splitlines():
            bits = line.split("\t")
            if len(bits) < 3:
                continue
            files += 1
            ins += int(bits[0]) if bits[0].isdigit() else 0
            dels += int(bits[1]) if bits[1].isdigit() else 0
        return {"files": files, "insertions": ins, "deletions": dels}

    # -- mutations --------------------------------------------------------------------------
    def add_all(self) -> None:
        self.git("add", "-A", "--", str(self.workdir))

    def commit(self, message: str, allow_empty: bool = True) -> str:
        args = ["commit", "-q", "-m", message]
        if allow_empty:
            args.insert(1, "--allow-empty")
        self.git(*args)
        return self.head_sha() or ""

    def tag(self, name: str) -> None:
        self.git("tag", "-f", name)

    def reset_hard(self, clean_raw: bool) -> None:
        """Discard everything the failed attempt wrote.

        ``clean_raw=False`` is the *retry* reset: the source file copied in for
        this ingest is still untracked, and the retry needs it on disk.
        ``clean_raw=True`` is the *resume* reset: any stray untracked source from
        an interrupted ingest must go, because the harness re-copies it.
        """
        if self.head_sha():
            self.git("reset", "-q", "--hard", "HEAD")
        else:
            self.git("rm", "-r", "-q", "--cached", ".", check=False)
        args = ["clean", "-fdq"]
        if not clean_raw:
            args += ["-e", "/raw/"]
        self.git(*args)

    def restore(self, paths: Sequence[str]) -> None:
        if paths:
            self.git("checkout", "HEAD", "--", *paths, check=False)

    def archive(self, ref: str, dest: str | Path, fmt: str = "tar.gz") -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(["git", f"--git-dir={self.gitdir}", "archive", f"--format={fmt}",
                              "-o", str(dest), ref], capture_output=True, text=True)
        if res.returncode != 0:
            raise R.HarnessError(f"git archive {ref} failed: {res.stderr.strip()[:300]}")
        return dest

    def extract_ref(self, ref: str, dest_dir: str | Path) -> Path:
        """Materialise a committed tree into ``dest_dir`` (used by ``extract.py``)."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(["git", f"--git-dir={self.gitdir}", "archive", "--format=tar", ref],
                                stdout=subprocess.PIPE)
        tar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=proc.stdout, capture_output=True, text=True)
        proc.wait()
        if proc.returncode != 0 or tar.returncode != 0:
            raise R.HarnessError(f"cannot extract {ref}: {tar.stderr[:300]}")
        return dest


# =================================================================================================
# pure helpers (unit-tested without git or the network)
# =================================================================================================
def page_counts(wiki_dir: str | Path) -> dict[str, int]:
    """Pages under ``wiki/`` by type, plus ``total`` (what ``eval.cost`` fits against)."""
    wiki = Path(wiki_dir)
    counts = {d: 0 for d in PAGE_DIRS}
    counts["root"] = 0
    if wiki.is_dir():
        for p in wiki.rglob("*.md"):
            rel = p.relative_to(wiki)
            key = rel.parts[0] if len(rel.parts) > 1 else "root"
            counts[key] = counts.get(key, 0) + 1
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def budget_cap(ok_costs: Sequence[float | None], lo: float = BUDGET_MIN, hi: float = BUDGET_MAX,
               mult: float = BUDGET_MULT, window: int = BUDGET_WINDOW,
               default: float = BUDGET_DEFAULT) -> float:
    """``clamp(4 * median(last 20 successful ingests), 1.5, 4.0)`` (pre-registered).

    A per-call ceiling stops one pathological ingest from eating the budget while
    still leaving headroom of four times the running median.
    """
    recent = [float(c) for c in list(ok_costs)[-window:] if c is not None and float(c) > 0]
    if not recent:
        return default
    return float(min(hi, max(lo, mult * statistics.median(recent))))


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0 or n < 2:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, my - slope * mx


def project_run(rows: Iterable[Mapping[str, Any]], *, n_target: int, started_at: str,
                now: datetime | None = None, lint_every: int = 50, lint_usd: float | None = None,
                usd_limit: float = STOP_RULE_USD, deadline: str = STOP_RULE_DEADLINE,
                stop_at: int = STOP_RULE_N) -> dict[str, Any]:
    """The pre-registered stop rule: project dollars *and* a completion date.

    Cost per ingest is fitted against the wiki size before that ingest
    (``usd = a + b * pages``) and wiki size against the ingest index
    (``pages = c + d * index``), so the two compose into a cost curve that can be
    summed over the remaining ingests -- a flat mean would understate a baseline
    whose whole claim is that maintenance stays cheap as the wiki grows.

    Two guards keep the projection from being optimistic in the two ways that
    would matter:

    * the fitted per-ingest cost is floored at the observed median, because with
      few points the slope is easily negative by noise and an unfloored fit would
      project the remaining ingests as nearly free;
    * calendar time is projected from *active* time (measured call wall-clock
      **plus** the usage-limit waits recorded in ``waits.jsonl``), not from the
      raw elapsed time since the run started.  Active time is the pace the
      harness can actually sustain, including the subscription's caps, while raw
      elapsed also counts the hours the operator simply had the run stopped.
      Both are reported; the decision uses the active pace.
    """
    rows = list(rows)
    now = now or datetime.now().astimezone()
    ingests = [r for r in rows if r.get("kind") == "ingest" and str(r.get("status")) in ("ok", "success")]
    lints = [r for r in rows if r.get("kind") == "lint" and str(r.get("status")) in ("ok", "success")]
    spent = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    n_done = max((int(r.get("ingest_index") or 0) for r in ingests), default=0)
    n_done = max(n_done, len({r.get("doc_id") for r in ingests}))

    xs, ys, idx = [], [], []
    for r in ingests:
        wp = r.get("wiki_pages_before")
        pages = float(wp.get("total", 0)) if isinstance(wp, Mapping) else float(wp or 0)
        xs.append(pages)
        ys.append(float(r.get("cost_usd") or 0.0))
        idx.append(float(r.get("ingest_index") or len(idx) + 1))
    b_cost, a_cost = _ols(xs, ys)
    d_pages, c_pages = _ols(idx, xs)
    mean_usd = (sum(ys) / len(ys)) if ys else 0.0

    median_usd = statistics.median(ys) if ys else 0.0
    floor_usd = max(median_usd, mean_usd)

    def cost_at(i: int) -> float:
        if len(xs) < 3:
            return floor_usd
        pages = c_pages + d_pages * i
        return max(floor_usd, a_cost + b_cost * pages)

    def cost_at_raw(i: int) -> float:
        pages = c_pages + d_pages * i
        return max(0.0, a_cost + b_cost * pages) if len(xs) >= 3 else mean_usd

    remaining = [i for i in range(n_done + 1, n_target + 1)]
    future_ingest_usd = sum(cost_at(i) for i in remaining)
    n_future_lints = sum(1 for i in remaining if lint_every and i % lint_every == 0)
    if remaining and (not lint_every or n_target % lint_every != 0):
        n_future_lints += 1                                    # the final lint
    lint_cost = lint_usd if lint_usd is not None else (
        statistics.median([float(r.get("cost_usd") or 0.0) for r in lints]) if lints else LINT_BUDGET)
    total_usd = spent + future_ingest_usd + n_future_lints * lint_cost
    total_usd_fit = spent + sum(cost_at_raw(i) for i in remaining) + n_future_lints * lint_cost

    try:
        t0 = datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        t0 = now
    elapsed_s = max(1.0, (now - t0).total_seconds())
    wall_only_s = sum(int(r.get("wall_ms") or 0) for r in rows) / 1000.0
    waited_s = sum(int(r.get("waited_ms") or 0) for r in rows) / 1000.0
    active_s = max(1.0, wall_only_s + waited_s)
    per_ingest_s = active_s / max(1, n_done)
    # As the cost curve rises so do the turn count and therefore the time: scale
    # the observed pace by the projected/observed cost ratio (never below 1).
    scale = (future_ingest_usd / len(remaining) / mean_usd) if (remaining and mean_usd > 0) else 1.0
    scale = max(1.0, scale)
    remaining_s = len(remaining) * per_ingest_s * scale + n_future_lints * max(per_ingest_s, 600.0)
    eta = now + timedelta(seconds=remaining_s)
    slack = elapsed_s / active_s                       # >1 means the run has been stopped for a while
    eta_elapsed_pace = now + timedelta(seconds=remaining_s * max(1.0, slack))
    dead = datetime.fromisoformat(deadline).replace(tzinfo=now.tzinfo)

    over_budget = total_usd > usd_limit
    over_deadline = eta > dead
    return {
        "n_done": n_done, "n_target": n_target, "n_remaining": len(remaining),
        "spent_usd": round(spent, 4), "mean_ingest_usd": round(mean_usd, 4),
        "cost_slope_usd_per_page": b_cost, "cost_intercept_usd": a_cost,
        "cost_slope_usd_per_100_pages": b_cost * 100.0,
        "pages_slope_per_ingest": d_pages, "pages_intercept": c_pages,
        "projected_future_ingest_usd": round(future_ingest_usd, 2),
        "n_future_lints": n_future_lints, "lint_usd": round(float(lint_cost), 4),
        "projected_total_usd": round(total_usd, 2),
        "projected_total_usd_unfloored": round(total_usd_fit, 2),
        "median_ingest_usd": round(median_usd, 4), "floor_ingest_usd": round(floor_usd, 4),
        "elapsed_hours": round(elapsed_s / 3600.0, 2),
        "measured_wall_hours": round(wall_only_s / 3600.0, 2),
        "usage_limit_wait_hours": round(waited_s / 3600.0, 2),
        "active_hours": round(active_s / 3600.0, 2), "idle_ratio": round(slack, 2),
        "per_ingest_wall_s": round(per_ingest_s, 1), "pace_scale": round(scale, 3),
        "projected_remaining_hours": round(remaining_s / 3600.0, 2),
        "projected_finish": eta.isoformat(timespec="seconds"),
        "projected_finish_at_elapsed_pace": eta_elapsed_pace.isoformat(timespec="seconds"),
        "deadline": deadline, "usd_limit": usd_limit,
        "over_budget": over_budget, "over_deadline": over_deadline,
        "stop": bool(over_budget or over_deadline), "stop_at": stop_at,
        "reason": ("projected ${:.0f} > ${:.0f}".format(total_usd, usd_limit) if over_budget else "")
                  + (" and " if over_budget and over_deadline else "")
                  + ("projected finish {} > {}".format(eta.date().isoformat(), deadline) if over_deadline else ""),
    }


def format_projection(p: Mapping[str, Any]) -> str:
    bar = "=" * 78
    lines = [
        bar,
        f"ARM B STOP-RULE PROJECTION  ({p['n_done']}/{p['n_target']} ingests done)",
        bar,
        f"  spent so far            ${p['spent_usd']:.2f}   (mean ${p['mean_ingest_usd']:.4f}/ingest)",
        f"  cost vs wiki size       ${p['cost_slope_usd_per_100_pages']:+.4f} per 100 pages "
        f"(intercept ${p['cost_intercept_usd']:.4f}); pages grow {p['pages_slope_per_ingest']:.2f}/ingest",
        f"  projected remaining     ${p['projected_future_ingest_usd']:.2f} ingests "
        f"+ {p['n_future_lints']} lints x ${p['lint_usd']:.2f}   "
        f"(per-ingest floored at the median ${p['floor_ingest_usd']:.4f}; raw fit would give "
        f"${p['projected_total_usd_unfloored']:.2f})",
        f"  PROJECTED TOTAL         ${p['projected_total_usd']:.2f}   (limit ${p['usd_limit']:.0f})",
        f"  active time             {p['active_hours']:.2f} h "
        f"({p['measured_wall_hours']:.2f} h calls + {p['usage_limit_wait_hours']:.2f} h usage-limit waits), "
        f"{p['per_ingest_wall_s']:.0f} s/ingest; wall since start {p['elapsed_hours']:.2f} h "
        f"(idle ratio {p['idle_ratio']:.1f}x)",
        f"  PROJECTED FINISH        {p['projected_finish']}  (+{p['projected_remaining_hours'] / 24:.1f} days"
        f" of running)   (deadline {p['deadline']}; at the observed wall pace "
        f"{p['projected_finish_at_elapsed_pace'][:10]})",
        f"  DECISION                {'STOP at N=' + str(p['stop_at']) if p['stop'] else 'CONTINUE'}"
        + (f"  -- {p['reason']}" if p["stop"] else ""),
        bar,
    ]
    return "\n".join(lines)


def parse_head(message: str | None) -> tuple[str | None, int | None, str | None]:
    """``"ingest 0042 dbp-x"`` -> ``("ingest", 42, "dbp-x")``."""
    m = _HEAD_RE.match((message or "").strip())
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)


# =================================================================================================
# the run
# =================================================================================================
@dataclass
class ArmBConfig:
    run_id: str
    corpus: str = "dbpedia"
    n: int = 300
    corpus_n: int | None = None       # corpus tier to draw from; defaults to n
    order: str = "1"
    corpus_dir: Path | None = None
    runs_root: Path = RUNS_ROOT
    model: str = R.MODEL
    effort: str = R.EFFORT
    max_turns: int = INGEST_MAX_TURNS
    timeout: int = INGEST_TIMEOUT
    lint_every: int = 50
    snapshot_at: tuple[int, ...] = (100, 300)
    tag_every: int = 50
    stop_rule_after: int = STOP_RULE_AFTER
    usd_limit: float = STOP_RULE_USD
    deadline: str = STOP_RULE_DEADLINE
    stop_at: int = STOP_RULE_N
    budget_override: float | None = None
    stop_after: int | None = None          # harness-side early exit (testing / staged runs)
    apply_stop_rule: bool = True
    strict_version: bool = False
    extra_defaults: dict[str, Any] = field(default_factory=dict)


class ArmBRun:
    def __init__(self, cfg: ArmBConfig, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.log = log
        self.workdir = Path(cfg.runs_root) / cfg.run_id
        self.gitdir = Path(cfg.runs_root) / f"{cfg.run_id}.git"
        self.meta = Path(cfg.runs_root) / f"{cfg.run_id}.meta"
        self.raw = self.workdir / "raw"
        self.wiki = self.workdir / "wiki"
        self.git = GitStore(self.workdir, self.gitdir)
        self.corpus_dir = (Path(cfg.corpus_dir) if cfg.corpus_dir
                           else REPO / "data" / cfg.corpus / str(cfg.corpus_n or cfg.n))
        self.order_file = self._resolve_order(cfg.order)
        self.order = read_order(self.order_file)[: cfg.n]
        self.claude_md_sha = sha256_file(SCHEMA_PATH)
        self.env = R.build_env()
        self.cli_version = R.assert_cli_version(env=self.env, strict=cfg.strict_version)
        self.calls = CallLog(self.meta, cfg.run_id, "B", defaults={
            "corpus": cfg.corpus, "n": cfg.n, "order_file": str(self.order_file),
            "model": cfg.model, "effort": cfg.effort, "cli_version": self.cli_version,
            "claude_md_sha256": self.claude_md_sha, **cfg.extra_defaults})
        self._lock_fh = None
        self.started_at = R.now_iso()

    # -- setup ------------------------------------------------------------------------------
    def _resolve_order(self, order: str) -> Path:
        p = Path(order)
        if p.exists():
            return p
        return self.corpus_dir / "orders" / (order if order.endswith(".txt") else f"order-{order}.txt")

    def acquire_lock(self) -> None:
        """One process per run directory: a second one would corrupt the repository."""
        if self._lock_fh is not None:
            return
        self.meta.mkdir(parents=True, exist_ok=True)
        fh = open(self.meta / "lock", "w")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise R.HarnessError(
                f"another wikiorg run-b process holds {self.meta / 'lock'} -- refusing to run "
                f"two writers against the same git repository") from exc
        fh.write(f"{os.getpid()} {R.now_iso()}\n")
        fh.flush()
        self._lock_fh = fh

    def release_lock(self) -> None:
        if self._lock_fh is not None:
            try:
                self._lock_fh.close()
            finally:
                self._lock_fh = None

    def __del__(self) -> None:                       # pragma: no cover - interpreter teardown
        try:
            self.release_lock()
        except Exception:
            pass

    def ensure_dirs(self) -> None:
        """``git clean -fd`` removes empty directories, so recreate them after every reset:
        the agent must always find ``raw/`` and ``wiki/`` in place."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(exist_ok=True)
        self.wiki.mkdir(exist_ok=True)

    def init_run(self) -> None:
        self.ensure_dirs()
        target = self.workdir / "CLAUDE.md"
        if not target.exists() or sha256_file(target) != self.claude_md_sha:
            shutil.copyfile(SCHEMA_PATH, target)
        self.git.init()
        if self.git.head_sha() is None:
            self.git.add_all()
            self.git.commit("init schema", allow_empty=True)
        write_json(self.meta / "run.json", self.manifest())

    def manifest(self) -> dict[str, Any]:
        cfg = self.cfg
        corpus_manifest = self.corpus_dir / "manifest.json"
        return {
            "run_id": cfg.run_id, "arm": "B", "corpus": cfg.corpus, "n": cfg.n,
            "corpus_n": cfg.corpus_n or cfg.n,
            "order_file": str(self.order_file), "n_order": len(self.order),
            "workdir": str(self.workdir), "gitdir": str(self.gitdir), "meta": str(self.meta),
            "corpus_dir": str(self.corpus_dir),
            "corpus_manifest_sha256": sha256_file(corpus_manifest) if corpus_manifest.exists() else None,
            "model": cfg.model, "effort": cfg.effort, "cli_version": self.cli_version,
            "claude_md_sha256": self.claude_md_sha, "schema_path": str(SCHEMA_PATH),
            "tools": R.TOOLS_B, "disallowed_tools": R.DISALLOWED_B,
            "max_turns": cfg.max_turns, "timeout_s": cfg.timeout,
            "lint_every": cfg.lint_every, "lint_budget_usd": LINT_BUDGET, "lint_max_turns": LINT_MAX_TURNS,
            "budget_rule": f"clamp({BUDGET_MULT}*median(last {BUDGET_WINDOW} ok), {BUDGET_MIN}, {BUDGET_MAX})",
            "stop_rule": {"after": cfg.stop_rule_after, "usd_limit": cfg.usd_limit,
                          "deadline": cfg.deadline, "stop_at": cfg.stop_at},
            "env_pins": R.ENV_PINS, "env_dropped": list(R.ENV_DROP),
            "started_at": self.started_at, "python": sys.executable,
        }

    def run_started_at(self) -> str:
        """The first row's timestamp: the calendar projection must not restart on resume."""
        rows = self.calls.rows()
        for r in rows:
            if r.get("started_at"):
                return str(r["started_at"])
        prev = self.meta / "run.json"
        if prev.exists():
            try:
                return str(json.loads(prev.read_text(encoding="utf-8")).get("started_at") or self.started_at)
            except (json.JSONDecodeError, OSError):
                pass
        return self.started_at

    # -- command construction ----------------------------------------------------------------
    def ingest_cmd(self, doc_id: str, budget: float, max_turns: int) -> list[str]:
        return ([
            "claude", "-p", INGEST_PROMPT.format(doc_id=doc_id),
            "--output-format", "stream-json", "--verbose",
            "--tools", R.TOOLS_B,
            "--allowedTools", R.TOOLS_B,
            "--disallowedTools", R.DISALLOWED_B,
            "--permission-mode", "acceptEdits",
        ] + R.common_flags(self.cfg.model, self.cfg.effort, budget=budget, max_turns=max_turns))

    def lint_cmd(self, budget: float = LINT_BUDGET, max_turns: int = LINT_MAX_TURNS) -> list[str]:
        return ([
            "claude", "-p", LINT_PROMPT,
            "--output-format", "stream-json", "--verbose",
            "--tools", R.TOOLS_B,
            "--allowedTools", R.TOOLS_B,
            "--disallowedTools", R.DISALLOWED_B,
            "--permission-mode", "acceptEdits",
        ] + R.common_flags(self.cfg.model, self.cfg.effort, budget=budget, max_turns=max_turns))

    # -- resume -------------------------------------------------------------------------------
    def resume_state(self) -> dict[str, Any]:
        """Reconcile ``calls.jsonl`` with ``HEAD`` and say where to continue.

        ``done`` holds the documents that are settled (successfully ingested or
        definitively failed).  The tricky case is a kill *between* the commit and
        the ledger append: ``HEAD`` is then one ingest ahead of the ledger, which
        the caller repairs from the archived stdout instead of aborting.
        """
        rows = self.calls.rows()
        ok = [r for r in rows if r.get("kind") == "ingest" and str(r.get("status")) in ("ok", "success")]
        failed_ids = {str(r.get("doc_id")) for r in rows
                      if r.get("kind") == "ingest" and str(r.get("status")) == "failed"}
        ok_ids = {str(r.get("doc_id")) for r in ok}
        last_ok = ok[-1] if ok else None
        last_index = int(last_ok.get("ingest_index") or 0) if last_ok else 0
        kind, head_index, head_doc = parse_head(self.git.head_message())
        settled = ok_ids | failed_ids
        next_pos = 0
        for i, doc_id in enumerate(self.order):
            if doc_id not in settled:
                next_pos = i
                break
        else:
            next_pos = len(self.order)
        state = {
            "n_rows": len(rows), "n_ok": len(ok), "n_failed": len(failed_ids),
            "last_ok_index": last_index, "last_ok_doc": str(last_ok.get("doc_id")) if last_ok else None,
            "head_kind": kind, "head_index": head_index, "head_doc": head_doc,
            "head_sha": self.git.head_sha(), "next_pos": next_pos,
            "next_doc": self.order[next_pos] if next_pos < len(self.order) else None,
            "consistent": True, "orphan_commit": False, "problem": None,
        }
        if kind is None:                                   # only the "init schema" commit
            if last_index != 0:
                state.update(consistent=False, problem=f"HEAD has no ingest commit but ledger has {last_index}")
            return state
        if kind == "lint":
            # a lint commit sits on top of ingest `last_index`; it is an orphan when
            # the process died between its commit and its ledger append (a lint is
            # worth up to $6, so losing it breaks the cost reconciliation)
            if not self._lint_row_at(head_index, rows):
                state.update(orphan_commit=True)
            return state
        if head_index == last_index and (head_doc is None or head_doc == state["last_ok_doc"]):
            return state
        if head_index == last_index + 1:
            state.update(orphan_commit=True)
            return state
        state.update(consistent=False,
                     problem=f"HEAD is 'ingest {head_index} {head_doc}' but the last ok ledger row is "
                             f"{last_index} ({state['last_ok_doc']})")
        return state

    def _lint_row_at(self, index: int | None, rows: Sequence[Mapping[str, Any]] | None = None) -> bool:
        rows = self.calls.rows() if rows is None else rows
        return any(r.get("kind") == "lint" and int(r.get("ingest_index") or -1) == index
                   and str(r.get("status")) in ("ok", "success") for r in rows)

    def _recover_orphan(self, state: Mapping[str, Any]) -> bool:
        """Rebuild the ledger row for a commit whose append never happened.

        The archived stdout is written *before* the commit, so it is always there
        when the commit is; everything except ``wall_ms`` can be re-derived from
        it.  If it is missing we drop the commit instead and redo the ingest.
        """
        kind = str(state["head_kind"])
        index = int(state["head_index"])
        doc_id = state["head_doc"]
        label = f"{kind} {index}" + (f" {doc_id}" if doc_id else "")
        prompt = INGEST_PROMPT.format(doc_id=doc_id) if kind == "ingest" else LINT_PROMPT
        candidates = sorted(self.calls.stdout_dir.glob(f"{kind}-{index:04d}-a*.jsonl"))
        if not candidates:
            self.log(f"resume: commit '{label}' has no archived stdout; dropping it and redoing")
            self.git.git("reset", "-q", "--hard", "HEAD~1")
            return False
        path = candidates[-1]
        inv = R.Invocation(cmd=[], returncode=0, stdout=path.read_text(encoding="utf-8"), stderr="",
                           wall_ms=0, timed_out=False)
        res = R.parse_call(inv, require_single_sonnet=False)
        if res.status != "ok":
            self.log(f"resume: archived stdout for {label} is not a success; dropping the commit and redoing")
            self.git.git("reset", "-q", "--hard", "HEAD~1")
            return False
        self.calls.append(row_from_call(
            res, kind=kind, attempt=int(path.stem.rsplit("-a", 1)[-1] or 0), prompt=prompt,
            started_at=R.now_iso(), ended_at=R.now_iso(), stdout_file=path.name,
            doc_id=doc_id, ingest_index=index, git_commit=state["head_sha"], recovered=True,
            wiki_pages_before=None, files_touched=None, diff_stat=None, wall_ms=None))
        self.log(f"resume: recovered the ledger row for '{label}' from {path.name}")
        return True

    def prepare(self, fresh: bool = False) -> dict[str, Any]:
        self.acquire_lock()
        if fresh and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
            shutil.rmtree(self.gitdir, ignore_errors=True)
            shutil.rmtree(self.meta, ignore_errors=True)
            self.calls = CallLog(self.meta, self.cfg.run_id, "B", defaults=self.calls.defaults)
        self.init_run()
        state = self.resume_state()
        for _ in range(3):                       # a lint orphan can sit on an ingest orphan
            if not state["orphan_commit"]:
                break
            self._recover_orphan(state)
            state = self.resume_state()
        if not state["consistent"]:
            raise R.HarnessError(
                f"cannot resume {self.cfg.run_id}: {state['problem']}. Inspect "
                f"{self.calls.path} and `git --git-dir={self.gitdir} log --oneline` by hand.")
        self.git.reset_hard(clean_raw=True)
        self.ensure_dirs()
        self._sync_raw(state["next_pos"])
        if state["n_rows"]:
            self.log(f"resume: {state['n_ok']} ingests done, {state['n_failed']} failed, "
                     f"continuing at position {state['next_pos'] + 1}/{len(self.order)} "
                     f"({state['next_doc']})")
        return state

    def _sync_raw(self, next_pos: int) -> None:
        """Only settled sources may be visible; a stray file would leak look-ahead."""
        allowed = {f"{d}.md" for d in self.order[:next_pos]}
        for p in sorted(self.raw.glob("*.md")):
            if p.name not in allowed:
                self.log(f"resume: removing stray source {p.name}")
                p.unlink()

    # -- one call -----------------------------------------------------------------------------
    def _call(self, cmd: Sequence[str], prompt: str, kind: str, index: int, attempt: int,
              timeout: float, **row_extra: Any) -> tuple[R.CallResult, dict[str, Any]]:
        started = R.now_iso()
        clock = R.WaitClock()

        def once(_k: int) -> R.CallResult:
            inv = R.run_claude(cmd, timeout=timeout, cwd=self.workdir, env=self.env)
            return R.parse_call(inv, require_single_sonnet=False)

        res = R.with_retries(
            once, attempts=5, log=self.log, sleep=clock.sleep,
            on_wait=lambda r: clock.add(R.wait_for_usage_reset(
                r, waits_path=self.calls.waits_path, env=self.env, cwd=self.workdir, log=self.log,
                context={"run_id": self.cfg.run_id, "kind": kind, "index": index})))
        ended = R.now_iso()
        name = f"{kind}-{index:04d}-a{attempt}"
        stdout_file = self.calls.archive_stdout(name, res.stdout, res.invocation.stderr)
        if res.status == "ok" and res.model_usage:
            R.usage_from_model_usage(res.result, require_single_sonnet=True, model=self.cfg.model)   # pre-registration guard
        row = row_from_call(res, kind=kind, attempt=attempt, prompt=prompt, started_at=started,
                            ended_at=ended, stdout_file=stdout_file,
                            ingest_index=index, wall_ms=clock.wall_ms, waited_ms=clock.waited_ms,
                            **row_extra)
        if res.permission_denials:
            self.calls.append({**row, "status": "aborted_permission_denial"})
            raise R.HarnessError(
                f"permission_denials is not empty on {kind} {index}: {res.permission_denials[:3]} -- "
                f"the tool configuration is wrong; fix it before spending more.")
        return res, row

    # -- ingest -------------------------------------------------------------------------------
    def ingest_one(self, index: int, doc_id: str) -> dict[str, Any]:
        self.ensure_dirs()
        src = self.corpus_dir / "docs" / f"{doc_id}.md"
        # A byte-for-byte copy: arm A embeds exactly these bytes, so any normalisation here
        # would silently give the two arms different inputs.  The parse is only a guard.
        raw_bytes = src.read_bytes()
        doc = parse_doc(raw_bytes.decode("utf-8"))
        if doc.id != doc_id:
            raise R.HarnessError(f"{src}: front-matter id {doc.id!r} does not match the order entry {doc_id!r}")
        (self.raw / f"{doc_id}.md").write_bytes(raw_bytes)

        pages_before = page_counts(self.wiki)
        ok_costs = [r.get("cost_usd") for r in self.calls.ok_rows("ingest")]
        base_budget = self.cfg.budget_override or budget_cap(ok_costs)
        prompt = INGEST_PROMPT.format(doc_id=doc_id)

        row: dict[str, Any] = {}
        for attempt in range(2):
            budget = base_budget * (RETRY_FACTOR ** attempt)
            max_turns = int(self.cfg.max_turns * (RETRY_FACTOR ** attempt))
            timeout = self.cfg.timeout * (RETRY_FACTOR ** attempt)
            res, row = self._call(self.ingest_cmd(doc_id, budget, max_turns), prompt, "ingest", index,
                                  attempt, timeout, doc_id=doc_id, wiki_pages_before=pages_before,
                                  budget_usd=round(budget, 4), max_turns=max_turns, timeout_s=timeout)
            if res.status == "interrupted":
                self.calls.append({**row, "status": "interrupted"})
                raise KeyboardInterrupt("claude was killed by a signal")
            if res.status == "ok":
                commit = self._commit_ingest(index, doc_id, row)
                row.update(commit)
                self.calls.append(row)
                return row
            self.log(f"ingest {index} {doc_id}: {res.status} ({res.subtype or res.error}) on attempt {attempt}")
            self.calls.append(row)
            if res.status not in ("budget", "turns", "timeout"):
                break
            self.git.reset_hard(clean_raw=False)
            self.ensure_dirs()
        # give up on this document: it must not appear half-ingested in the wiki
        self.git.reset_hard(clean_raw=False)
        self.ensure_dirs()
        (self.raw / f"{doc_id}.md").unlink(missing_ok=True)
        failed = {**row, "status": "failed", "kind": "ingest", "doc_id": doc_id, "ingest_index": index,
                  "attempt": "final", "git_commit": None, "wiki_pages_before": pages_before,
                  "note": "reset and source removed; document is absent from the wiki"}
        self.calls.append(failed)
        self.log(f"ingest {index} {doc_id}: FAILED after 2 attempts; continuing")
        return failed

    def _commit_ingest(self, index: int, doc_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        self.git.add_all()
        touched, violations = self._classify_status(doc_id)
        if violations:
            self.git.restore(violations)
            self.git.add_all()
            touched, _ = self._classify_status(doc_id)
            self.log(f"WARNING: ingest {index} touched protected files {violations}; reverted")
        stat = self.git.numstat()
        sha = self.git.commit(f"ingest {index:04d} {doc_id}")
        return {"git_commit": sha, "files_touched": touched, "diff_stat": stat,
                "protected_files_touched": violations or None}

    def _classify_status(self, doc_id: str | None) -> tuple[dict[str, Any], list[str]]:
        created: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []
        violations: list[str] = []
        allowed_raw = f"raw/{doc_id}.md" if doc_id else None
        for code, path in self.git.porcelain():
            c = code[0] if code else " "
            if c in ("A", "?"):
                created.append(path)
            elif c == "D":
                deleted.append(path)
            else:
                modified.append(path)
            protected = path == "CLAUDE.md" or (path.startswith("raw/") and path != allowed_raw)
            if protected and c in ("M", "D", "R", "C", "T"):
                violations.append(path)
        wiki_files = [p for p in created + modified + deleted if p.startswith("wiki/")]
        return ({"created": created, "modified": modified, "deleted": deleted,
                 "n_total": len(created) + len(modified) + len(deleted),
                 "n_wiki": len(wiki_files), "wiki": sorted(wiki_files)}, violations)

    # -- lint ---------------------------------------------------------------------------------
    def lint(self, index: int) -> dict[str, Any]:
        pages_before = page_counts(self.wiki)
        res, row = self._call(self.lint_cmd(), LINT_PROMPT, "lint", index, 0, LINT_TIMEOUT,
                              doc_id=None, wiki_pages_before=pages_before,
                              budget_usd=LINT_BUDGET, max_turns=LINT_MAX_TURNS, timeout_s=LINT_TIMEOUT)
        if res.status == "interrupted":
            self.calls.append({**row, "status": "interrupted"})
            raise KeyboardInterrupt("claude was killed by a signal")
        if res.status == "ok":
            self.git.add_all()
            touched, violations = self._classify_status(None)
            if violations:
                self.git.restore(violations)
                self.git.add_all()
                touched, _ = self._classify_status(None)
            stat = self.git.numstat()
            sha = self.git.commit(f"lint {index:04d}")
            row.update({"git_commit": sha, "files_touched": touched, "diff_stat": stat,
                        "protected_files_touched": violations or None})
        else:
            self.log(f"lint at {index}: {res.status} ({res.subtype or res.error}); resetting")
            self.git.reset_hard(clean_raw=False)
            self.ensure_dirs()
            row["status"] = "failed" if res.status in ("budget", "turns", "timeout") else row["status"]
        self.calls.append(row)
        return row

    # -- milestones ---------------------------------------------------------------------------
    def _after_ingest(self, index: int, n_target: int) -> None:
        """Lint / tag / snapshot for a completed ingest, idempotently.

        A resume replays this for the ingest it lands on, because a kill *inside*
        a 1800-second milestone lint leaves the ingest committed and the lint
        never run -- and with it the tag and the pre-registered N=100 snapshot.
        Re-running a finished run must also not lint twice, so the ledger (and,
        for the very last one, ``HEAD``) decides whether the lint already happened.
        """
        if (self.cfg.lint_every and index % self.cfg.lint_every == 0 and index < n_target
                and not self._lint_row_at(index)
                and self.git.head_message() != f"lint {index:04d}"):
            self.lint(index)
        if self.cfg.tag_every and index % self.cfg.tag_every == 0:
            self.git.tag(f"ingest-{index:04d}")
        if index in tuple(self.cfg.snapshot_at):
            self.snapshot(index)

    def snapshot(self, index: int) -> Path:
        dest = self.meta / "snapshots" / f"{self.cfg.run_id}_N{index:04d}.tar.gz"
        self.git.archive("HEAD", dest)
        self.log(f"snapshot N={index}: {dest}")
        return dest

    # -- the loop -----------------------------------------------------------------------------
    def run(self, fresh: bool = False) -> dict[str, Any]:
        try:
            return self._run(fresh=fresh)
        finally:
            self.release_lock()

    def _run(self, fresh: bool = False) -> dict[str, Any]:
        state = self.prepare(fresh=fresh)
        start_pos = int(state["next_pos"])
        n_target = self.cfg.n
        started_at = self.run_started_at()
        stop_decision: dict[str, Any] | None = None
        limit = len(self.order)
        if self.cfg.stop_after is not None:
            limit = min(limit, start_pos + self.cfg.stop_after)

        if start_pos > 0:
            self._after_ingest(start_pos, n_target)   # a kill inside a milestone lint

        pos = start_pos
        while pos < limit:
            doc_id = self.order[pos]
            index = pos + 1
            t0 = time.monotonic()
            row = self.ingest_one(index, doc_id)
            usd = float(row.get("cost_usd") or 0.0)
            self.log(f"[{index}/{n_target}] {doc_id} {row.get('status')} "
                     f"${usd:.4f} turns={row.get('num_turns')} "
                     f"files={(row.get('files_touched') or {}).get('n_wiki')} "
                     f"cache_read={(row.get('tokens') or {}).get('cache_read')} "
                     f"{time.monotonic() - t0:.0f}s")
            pos += 1
            self._after_ingest(index, n_target)

            if (self.cfg.apply_stop_rule and stop_decision is None
                    and index >= self.cfg.stop_rule_after):
                proj = project_run(self.calls.rows(), n_target=n_target, started_at=started_at,
                                   lint_every=self.cfg.lint_every, usd_limit=self.cfg.usd_limit,
                                   deadline=self.cfg.deadline, stop_at=self.cfg.stop_at)
                if index == self.cfg.stop_rule_after or index % 50 == 0:
                    self.log("\n" + format_projection(proj) + "\n")
                    self.calls.decision({**proj, "decision": "stop_rule_projection"})
                if proj["stop"]:
                    stop_decision = proj
                    n_target = min(n_target, self.cfg.stop_at)
                    limit = min(limit, n_target)
                    if n_target not in tuple(self.cfg.snapshot_at):
                        self.cfg.snapshot_at = tuple(self.cfg.snapshot_at) + (n_target,)
                    self.log(f"\n*** PRE-REGISTERED STOP RULE TRIGGERED: {proj['reason']} -> "
                             f"stopping this order at N={n_target} ***\n")
                    self.calls.decision({**proj, "decision": "stop_rule_triggered", "n_target": n_target})

        did_final_lint = False
        if pos and pos >= min(n_target, len(self.order)) and self.cfg.lint_every:
            if not self._lint_row_at(pos):
                self.lint(pos)
            did_final_lint = True
            if self.cfg.tag_every:
                self.git.tag(f"ingest-{pos:04d}")
            if pos in tuple(self.cfg.snapshot_at):
                self.snapshot(pos)              # retaken after the final lint

        summary = {
            **self.manifest(), "started_at": started_at, "finished_at": R.now_iso(),
            "n_ingested": pos, "final_lint": did_final_lint,
            "stop_rule": stop_decision, "head": self.git.head_sha(),
            "totals": totals(self.calls.rows()), "pages": page_counts(self.wiki),
        }
        if len(self.calls.ok_rows("ingest")) >= 3:
            summary["projection"] = project_run(self.calls.rows(), n_target=self.cfg.n,
                                                started_at=started_at, lint_every=self.cfg.lint_every,
                                                usd_limit=self.cfg.usd_limit, deadline=self.cfg.deadline)
        write_json(self.meta / "run.json", summary)
        self.log(f"arm B {self.cfg.run_id}: {pos} ingests, ${summary['totals']['cost_usd']:.4f}, "
                 f"{summary['pages']['total']} wiki pages, HEAD={(summary['head'] or '')[:8]}")
        return summary


# =================================================================================================
# cli
# =================================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wikiorg run-b", description="arm B: incremental LLM wiki ingest")
    p.add_argument("--run-id", required=True)
    p.add_argument("--corpus", default="dbpedia")
    p.add_argument("--n", type=int, default=300, help="number of documents to ingest")
    p.add_argument("--corpus-n", type=int, help="corpus tier directory to draw from (default: --n)")
    p.add_argument("--order", default="1", help="1, 2, or a path to an order file")
    p.add_argument("--corpus-dir")
    p.add_argument("--runs-root", default=str(RUNS_ROOT))
    p.add_argument("--model", default=R.MODEL)
    p.add_argument("--effort", default=R.EFFORT)
    p.add_argument("--max-turns", type=int, default=INGEST_MAX_TURNS)
    p.add_argument("--timeout", type=int, default=INGEST_TIMEOUT)
    p.add_argument("--lint-every", type=int, default=50, help="0 disables lints")
    p.add_argument("--snapshot-at", default="100,300")
    p.add_argument("--tag-every", type=int, default=50)
    p.add_argument("--budget", type=float, help="override the per-call budget cap (USD)")
    p.add_argument("--stop-after", type=int, help="stop after N further ingests (staged / test runs)")
    p.add_argument("--stop-rule-after", type=int, default=STOP_RULE_AFTER)
    p.add_argument("--usd-limit", type=float, default=STOP_RULE_USD)
    p.add_argument("--deadline", default=STOP_RULE_DEADLINE)
    p.add_argument("--no-stop-rule", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted run (the default; accepted for explicitness)")
    p.add_argument("--fresh", action="store_true", help="delete the run directory first (default is resume)")
    p.add_argument("--strict-version", action="store_true", help="abort if the CLI version moved")
    p.add_argument("--dry-run", action="store_true", help="print the first command and exit")
    return p


def _install_sigterm_handler() -> None:
    """``nohup ... &`` + ``kill <pid>`` is how a days-long run gets stopped; the
    default SIGTERM handler skips ``finally``, orphaning the ``claude`` child in
    the work tree.  Turning it into KeyboardInterrupt takes the same clean path."""
    def handler(_signum, _frame):
        raise KeyboardInterrupt("SIGTERM")

    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:                          # pragma: no cover - not the main thread
        pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(R.clean_argv(argv))
    _install_sigterm_handler()
    cfg = ArmBConfig(
        run_id=args.run_id, corpus=args.corpus, n=args.n, corpus_n=args.corpus_n, order=args.order,
        corpus_dir=Path(args.corpus_dir) if args.corpus_dir else None,
        runs_root=Path(args.runs_root), model=args.model, effort=args.effort,
        max_turns=args.max_turns, timeout=args.timeout, lint_every=args.lint_every,
        snapshot_at=tuple(int(x) for x in str(args.snapshot_at).split(",") if x.strip()),
        tag_every=args.tag_every, budget_override=args.budget, stop_after=args.stop_after,
        stop_rule_after=args.stop_rule_after, usd_limit=args.usd_limit, deadline=args.deadline,
        apply_stop_rule=not args.no_stop_rule, strict_version=args.strict_version,
    )
    run = ArmBRun(cfg)
    if args.dry_run:
        doc = run.order[0]
        print(" ".join(run.ingest_cmd(doc, budget_cap([]), cfg.max_turns)))
        print(json.dumps(run.manifest(), indent=2, default=str))
        return 0
    try:
        run.run(fresh=args.fresh)
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
