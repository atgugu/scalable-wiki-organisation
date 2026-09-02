"""Arm A+ — arm A plus one LLM call per cluster to name it.

The point of A+ is to separate two things the label metric otherwise confounds:
whether the cheap pipeline puts the right documents together (arm A's
partition, untouched here) and whether a comma-joined c-TF-IDF term list is a
readable category *name*.  A+ keeps the partition and the links of an existing
arm-A run and replaces only ``categories[cid]["label"]``.

One no-tools call per cluster, at most four in flight (the CLI is a subprocess
per call, and the run is otherwise cheap enough that concurrency only saves
wall-clock).  The reply is validated against a one-field schema and a
``<= 6 words`` rule with a single retry; if it still fails, the c-TF-IDF label
is kept and the fallback is recorded.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import runner as R
from .arm_b import REPO, RUNS_ROOT
from .calls import CallLog, row_from_call, sha256_text, totals, write_json

SCHEMA = json.dumps({"type": "object", "additionalProperties": False,
                     "properties": {"label": {"type": "string"}}, "required": ["label"]},
                    separators=(",", ":"))
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
# The system prompt is the file the thesis appendix includes verbatim (single source of truth).
SYSTEM = (_PROMPTS_DIR / "aplus_system.txt").read_text(encoding="utf-8")
BUDGET = 0.05
MAX_TURNS = 2
TIMEOUT = 180
MAX_WORDS = 6
SEMAPHORE = 4
_BAD_WORDS = re.compile(r"\b(cluster|category|categories|topic group)\b", re.I)


def cluster_prompt(terms: Sequence[str], titles: Sequence[str], n_terms: int = 10, n_titles: int = 5) -> str:
    return ("Terms: " + ", ".join(str(t) for t in terms[:n_terms])
            + "\nTitles:\n" + "\n".join(f"- {t}" for t in titles[:n_titles]) + "\n")


def check_label(label: str, max_words: int = MAX_WORDS) -> tuple[bool, str]:
    """``(ok, reason)`` for the ``<= 6 words`` and no-meta-words rule."""
    s = str(label or "").strip().strip('"').strip()
    if not s:
        return False, "empty"
    if len(s.split()) > max_words:
        return False, f"{len(s.split())} words > {max_words}"
    if _BAD_WORDS.search(s):
        return False, "uses a meta word"
    return True, ""


def parse_label(res: R.CallResult) -> str | None:
    so = res.structured_output
    if isinstance(so, Mapping) and so.get("label"):
        return str(so["label"])
    if isinstance(so, str):
        try:
            obj = json.loads(so)
            if isinstance(obj, Mapping) and obj.get("label"):
                return str(obj["label"])
        except (json.JSONDecodeError, ValueError):
            pass
    text = (res.text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, Mapping) and obj.get("label"):
                return str(obj["label"])
        except (json.JSONDecodeError, ValueError):
            pass
    return text.splitlines()[0].strip() if text else None


class ArmAPlusRun:
    def __init__(self, run_dir: str | Path, run_id: str | None = None, runs_root: Path = RUNS_ROOT,
                 model: str = R.MODEL, effort: str = R.EFFORT, out_dir: str | Path | None = None,
                 concurrency: int = SEMAPHORE, log=print):
        self.src = Path(run_dir)
        self.state = json.loads((self.src / "wiki_state.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.src / "manifest.json").read_text(encoding="utf-8")) if (
            self.src / "manifest.json").exists() else {}
        self.run_id = run_id or f"Aplus_{self.manifest.get('run_id', self.src.name)}"
        self.model, self.effort, self.log = model, effort, log
        self.cwd = Path(runs_root) / "_inference"
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.meta = Path(runs_root) / f"{self.run_id}.meta"
        self.out = Path(out_dir) if out_dir else (
            REPO / "runs" / "A+" / str(self.manifest.get("corpus", "corpus"))
            / str(self.manifest.get("n", len(self.state.get("pages", {})))) / self.src.name)
        self.env = R.build_env()
        self.cli_version = R.assert_cli_version(env=self.env)
        self.sem = threading.Semaphore(max(1, concurrency))
        self.lock = threading.Lock()
        self.calls = CallLog(self.meta, self.run_id, "A+", defaults={
            "corpus": self.manifest.get("corpus"), "n": self.manifest.get("n"), "model": model,
            "effort": effort, "cli_version": self.cli_version, "source_run": str(self.src)})

    def cmd(self) -> list[str]:
        return (["claude", "-p", "--output-format", "json", "--tools", "",
                 "--system-prompt", SYSTEM, "--json-schema", SCHEMA]
                + R.common_flags(self.model, self.effort, budget=BUDGET, max_turns=MAX_TURNS))

    def titles_by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for page in self.state.get("pages", {}).values():
            cid = page.get("category")
            if cid:
                out.setdefault(str(cid), []).append(str(page.get("title") or ""))
        return out

    def name_cluster(self, cid: str, terms: Sequence[str], titles: Sequence[str],
                     index: int) -> dict[str, Any]:
        prompt = cluster_prompt(terms, titles)
        fallback = str(self.state["categories"][cid].get("label") or "")
        label, reason, rows = fallback, "no attempt", []
        for attempt in range(2):
            started = R.now_iso()
            clock = R.WaitClock()
            with self.sem:
                def once(_k: int) -> R.CallResult:
                    inv = R.run_claude(self.cmd(), stdin=prompt, timeout=TIMEOUT, cwd=self.cwd, env=self.env)
                    return R.parse_call(inv, require_single_sonnet=False)

                res = R.with_retries(once, attempts=5, log=self.log, sleep=clock.sleep,
                                     on_wait=lambda r: clock.add(R.wait_for_usage_reset(
                                         r, waits_path=self.calls.waits_path, env=self.env,
                                         cwd=self.cwd, log=self.log,
                                         context={"run_id": self.run_id, "cid": cid})))
            ended = R.now_iso()
            with self.lock:
                stdout_file = self.calls.archive_stdout(f"aplus-{index:04d}-a{attempt}", res.stdout,
                                                        res.invocation.stderr)
            cand = parse_label(res) if res.status == "ok" else None
            ok, reason = check_label(cand or "")
            row = row_from_call(res, kind="name_cluster", attempt=attempt, prompt=prompt,
                                started_at=started, ended_at=ended, stdout_file=stdout_file,
                                doc_id=cid, ingest_index=index, budget_usd=BUDGET,
                                max_turns=MAX_TURNS, timeout_s=TIMEOUT,
                                wall_ms=clock.wall_ms, waited_ms=clock.waited_ms)
            row.update({"prompt": None, "prompt_sha256": sha256_text(prompt),
                        "validation": {"label": cand, "ok": ok, "reason": reason,
                                       "n_words": len(str(cand or "").split())},
                        "status": "ok" if (res.status == "ok" and ok) else
                                  ("invalid_label" if res.status == "ok" else res.status)})
            rows.append(row)
            with self.lock:
                self.calls.append(row)
            if ok and cand:
                label = str(cand).strip().strip('"')
                break
        return {"cid": cid, "label": label, "fallback": label == fallback, "reason": reason,
                "attempts": len(rows)}

    def run(self) -> dict[str, Any]:
        cats = self.state.get("categories", {})
        titles = self.titles_by_category()
        jobs = [(i, cid, list(c.get("terms") or []), titles.get(cid, []))
                for i, (cid, c) in enumerate(sorted(cats.items()), start=1)]
        with ThreadPoolExecutor(max_workers=SEMAPHORE) as pool:
            results = list(pool.map(lambda j: self.name_cluster(j[1], j[2], j[3], j[0]), jobs))
        for r in results:
            self.state["categories"][r["cid"]]["label"] = r["label"]
        self.state["arm"] = "A+"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "wiki_state.json").write_text(
            json.dumps(self.state, sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for name in ("timing.json", "config.yaml"):
            if (self.src / name).exists():
                shutil.copyfile(self.src / name, self.out / name)
        shutil.copyfile(self.calls.path, self.out / "llm_calls.jsonl")
        rows = self.calls.rows()
        summary = {**self.manifest, "run_id": self.run_id, "arm": "A+", "source_run": str(self.src),
                   "n_clusters": len(jobs), "n_fallback": sum(1 for r in results if r["fallback"]),
                   "model": self.model, "effort": self.effort, "cli_version": self.cli_version,
                   "system_prompt_sha256": sha256_text(SYSTEM), "llm_calls": len(rows),
                   "calls_file": str(self.out / "llm_calls.jsonl"), "totals": totals(rows),
                   "labels": {r["cid"]: r["label"] for r in results}, "finished_at": R.now_iso()}
        write_json(self.out / "manifest.json", summary)
        write_json(self.meta / "run.json", summary)
        self.log(f"arm A+ {self.run_id}: {len(jobs)} clusters named ({summary['n_fallback']} fell back), "
                 f"${summary['totals']['cost_usd']:.4f} -> {self.out}")
        return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wikiorg run-aplus", description="arm A+: name each cluster with one LLM call")
    p.add_argument("--run-dir", required=True, help="an arm-A run directory (with wiki_state.json)")
    p.add_argument("--run-id")
    p.add_argument("--runs-root", default=str(RUNS_ROOT))
    p.add_argument("--out")
    p.add_argument("--model", default=R.MODEL)
    p.add_argument("--effort", default=R.EFFORT)
    p.add_argument("--concurrency", type=int, default=SEMAPHORE)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(R.clean_argv(argv))
    run = ArmAPlusRun(args.run_dir, run_id=args.run_id, runs_root=Path(args.runs_root),
                      model=args.model, effort=args.effort, out_dir=args.out,
                      concurrency=args.concurrency)
    if args.dry_run:
        cats = run.state.get("categories", {})
        titles = run.titles_by_category()
        cid = sorted(cats)[0]
        print(" ".join(run.cmd()[:8]), "...")
        print(cluster_prompt(cats[cid].get("terms") or [], titles.get(cid, [])))
        return 0
    run.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
