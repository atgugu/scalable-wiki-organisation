"""Arm C — the one-shot, no-tools LLM labeller: one call per document.

The cheap control for arm B.  The model sees the rules, the full numbered list
of corpus titles, the topics used so far, and the document; it returns a
schema-validated object with a topic slug, a label, a one-line summary and up to
five related titles.  No tools, no wiki state on disk, no memory between calls
other than the topic list the harness carries forward.

Prompt caching is why the corpus title list lives in the **system** prompt: it
is byte-identical on every call of a run, so from the second call on it is a
cache read.  Only the topics-so-far block and the document itself change.

Validation (pre-registered, ``configs/prereg.yaml`` / plan W2.3):

* ``related_titles`` resolve to corpus ids by **exact** match after NFKC +
  casefold + punctuation stripping (``wikiorg.adapters.common.norm_title``).
  Self references, repeats and titles outside the corpus are dropped and
  counted in ``n_invalid``; near misses (difflib >= 0.9) are counted separately
  and never become links, so the link-validity table can distinguish a typo from
  a hallucinated document.
* Topic identity is the normalised match of the ``topic`` string against the
  topics already used.  That match is trusted over the model's own
  ``is_new_topic`` flag, and the number of disagreements is recorded
  (``n_is_new_disagreements``) as a descriptive measure of self-consistency.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..adapters.common import TitleIndex, norm_title
from ..data.docs import Doc, doc_markdown, read_docs, read_order
from . import runner as R
from .arm_b import REPO, RUNS_ROOT
from .calls import CallLog, row_from_call, sha256_text, totals, write_json

PROMPTS = Path(__file__).resolve().parent / "prompts"
SCHEMA_PATH = PROMPTS / "c_schema.json"
RULES_PATH = PROMPTS / "c_rules.txt"

BUDGET = 0.10
MAX_TURNS = 3
TIMEOUT = 300
MAX_SYSTEM_BYTES = 120_000        # Linux MAX_ARG_STRLEN is 128 KiB per argv entry
MAX_RELATED = 5


# ------------------------------------------------------------------------------------------------
# prompts
# ------------------------------------------------------------------------------------------------
def system_prompt(titles: Sequence[str], rules: str | None = None,
                  candidates: str = "full") -> str:
    """Rules + the numbered corpus title list: byte-stable for the whole run."""
    rules = rules if rules is not None else RULES_PATH.read_text(encoding="utf-8")
    if candidates == "processed":
        rules += ("\nAdditional restriction: `related_titles` may only name documents listed as "
                  "already processed in the user message. Before any document has been processed, "
                  "return an empty list.\n")
    lines = [f"{i + 1}. {t}" for i, t in enumerate(titles)]
    return f"{rules.rstrip()}\n\nCorpus ({len(titles)} documents):\n" + "\n".join(lines) + "\n"


def user_prompt(doc: Doc, topics: Sequence[tuple[str, str]], processed: Sequence[str] | None = None) -> str:
    """Topics so far + (optionally) the processed titles + the document itself."""
    parts: list[str] = []
    if topics:
        parts.append("Topics used so far (slug — label):\n"
                     + "\n".join(f"- {slug} — {label}" for slug, label in topics))
    else:
        parts.append("Topics used so far: none. This is the first document.")
    if processed is not None:
        if processed:
            parts.append("Documents already processed (the only valid related_titles):\n"
                         + "\n".join(f"- {t}" for t in processed))
        else:
            parts.append("Documents already processed: none, so related_titles must be empty.")
    parts.append("Document:\n" + doc_markdown(doc).rstrip())
    return "\n\n".join(parts) + "\n"


# ------------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------------
@dataclass
class TopicBook:
    """Topics seen so far, keyed by the normalised topic string."""

    by_key: dict[str, str] = None          # norm(topic) -> canonical slug
    labels: dict[str, str] = None          # slug -> first label seen
    n_disagreements: int = 0

    def __post_init__(self) -> None:
        self.by_key = self.by_key or {}
        self.labels = self.labels or {}

    def resolve(self, topic: str, label: str, is_new_flag: Any) -> dict[str, Any]:
        key = norm_title(topic)
        known = self.by_key.get(key)
        is_new = known is None
        if bool(is_new_flag) != is_new:
            self.n_disagreements += 1
        slug = known or str(topic).strip()
        if known is None and key:
            self.by_key[key] = slug
            self.labels[slug] = str(label or slug)
        return {"topic": slug, "is_new_topic": is_new, "model_is_new_topic": bool(is_new_flag),
                "is_new_disagreement": bool(is_new_flag) != is_new}

    def as_pairs(self) -> list[tuple[str, str]]:
        return [(slug, self.labels.get(slug, slug)) for slug in self.by_key.values()]


def validate_related(raw: Iterable[Any], doc_id: str, index: TitleIndex,
                     allowed: set[str] | None = None, max_links: int = MAX_RELATED) -> dict[str, Any]:
    """Exact title -> id resolution with the pre-registered drop rules."""
    resolved: list[str] = []
    seen: set[str] = set()
    n_self = n_dup = n_unresolved = n_near = n_not_allowed = 0
    raw_list = [str(x) for x in (raw or [])]
    for text in raw_list:
        rid = index.resolve(text)
        if rid is None:
            kind, near = index.classify(text)
            if kind == "near_miss":
                n_near += 1
            n_unresolved += 1
            continue
        if rid == doc_id:
            n_self += 1
            continue
        if rid in seen:
            n_dup += 1
            continue
        if allowed is not None and rid not in allowed:
            n_not_allowed += 1
            continue
        seen.add(rid)
        resolved.append(rid)
    over = max(0, len(resolved) - max_links)
    resolved = resolved[:max_links]
    n_invalid = n_self + n_dup + n_unresolved + n_not_allowed
    return {"related": resolved, "related_raw": raw_list, "n_related_raw": len(raw_list),
            "n_related": len(resolved), "n_invalid": n_invalid, "n_self": n_self,
            "n_duplicate": n_dup, "n_unresolved": n_unresolved, "n_near_miss": n_near,
            "n_not_processed": n_not_allowed, "n_over_limit": over}


def parse_structured(res: R.CallResult) -> dict[str, Any] | None:
    """``structured_output`` when the CLI validated it, else JSON in the reply text."""
    so = res.structured_output
    if isinstance(so, Mapping) and "topic" in so:
        return dict(so)
    if isinstance(so, str):
        try:
            obj = json.loads(so)
            if isinstance(obj, Mapping):
                return dict(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    text = (res.text or "").strip()
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return dict(obj) if isinstance(obj, Mapping) else None


# ------------------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------------------
class ArmCRun:
    def __init__(self, run_id: str, corpus: str, n: int, order: str = "1",
                 corpus_n: int | None = None, corpus_dir: Path | None = None, runs_root: Path = RUNS_ROOT,
                 model: str = R.MODEL, effort: str = R.EFFORT, related_candidates: str = "full",
                 budget: float = BUDGET, max_turns: int = MAX_TURNS, timeout: int = TIMEOUT,
                 stop_after: int | None = None, out_root: Path | None = None, log=print):
        self.run_id, self.corpus, self.n, self.log = run_id, corpus, n, log
        self.related_candidates = related_candidates
        self.model, self.effort = model, effort
        self.budget, self.max_turns, self.timeout = budget, max_turns, timeout
        self.stop_after = stop_after
        self.out_root = Path(out_root) if out_root else REPO / "runs"
        self.corpus_n = corpus_n or n
        self.corpus_dir = Path(corpus_dir) if corpus_dir else REPO / "data" / corpus / str(self.corpus_n)
        order_path = Path(order)
        self.order_file = order_path if order_path.exists() else self.corpus_dir / "orders" / f"order-{order}.txt"
        self.cwd = Path(runs_root) / "_inference"
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.meta = Path(runs_root) / f"{run_id}.meta"
        docs = {d.id: d for d in read_docs(self.corpus_dir)}
        self.ids = read_order(self.order_file)[:n]
        self.docs = [docs[i] for i in self.ids]
        self.titles = {d.id: d.title for d in self.docs}
        self.index = TitleIndex(self.titles)
        self.schema = SCHEMA_PATH.read_text(encoding="utf-8").strip()
        self.system = system_prompt([self.titles[i] for i in self.ids], candidates=related_candidates)
        if len(self.system.encode("utf-8")) > MAX_SYSTEM_BYTES:
            raise R.HarnessError(
                f"system prompt is {len(self.system.encode('utf-8'))} bytes; the OS caps one argv "
                f"entry at {MAX_SYSTEM_BYTES}. Reduce N or move the title list to the user prompt.")
        self.env = R.build_env()
        self.cli_version = R.assert_cli_version(env=self.env)
        self.calls = CallLog(self.meta, run_id, "C", defaults={
            "corpus": corpus, "n": n, "order_file": str(self.order_file), "model": model,
            "effort": effort, "cli_version": self.cli_version,
            "claude_md_sha256": None, "system_prompt_sha256": sha256_text(self.system),
            "related_candidates": related_candidates})

    # ------------------------------------------------------------------------------------------
    def cmd(self) -> list[str]:
        return (["claude", "-p", "--output-format", "json", "--tools", "",
                 "--system-prompt", self.system, "--json-schema", self.schema]
                + R.common_flags(self.model, self.effort, budget=self.budget, max_turns=self.max_turns))

    def label_one(self, index: int, doc: Doc, book: TopicBook, processed: list[str]) -> dict[str, Any]:
        prompt = user_prompt(doc, book.as_pairs(),
                             processed if self.related_candidates == "processed" else None)
        started = R.now_iso()
        clock = R.WaitClock()

        def once(_k: int) -> R.CallResult:
            inv = R.run_claude(self.cmd(), stdin=prompt, timeout=self.timeout, cwd=self.cwd, env=self.env)
            return R.parse_call(inv, require_single_sonnet=False)

        res = R.with_retries(once, attempts=5, log=self.log, sleep=clock.sleep,
                             on_wait=lambda r: clock.add(R.wait_for_usage_reset(
                                 r, waits_path=self.calls.waits_path, env=self.env, cwd=self.cwd,
                                 log=self.log, context={"run_id": self.run_id, "kind": "label", "index": index})))
        ended = R.now_iso()
        stdout_file = self.calls.archive_stdout(f"label-{index:04d}", res.stdout, res.invocation.stderr)
        so = parse_structured(res) if res.status in ("ok", "turns", "budget") else None
        validation: dict[str, Any] = {"schema_ok": so is not None}
        status = res.status
        if so is not None:
            topic = book.resolve(str(so.get("topic") or ""), str(so.get("label") or ""), so.get("is_new_topic"))
            allowed = set(processed_ids(self.ids, index)) if self.related_candidates == "processed" else None
            rel = validate_related(so.get("related_titles"), doc.id, self.index, allowed=allowed)
            so = {**so, "topic": topic["topic"], "is_new_topic": topic["is_new_topic"]}
            validation.update(topic)
            validation.update({k: v for k, v in rel.items() if k not in ("related", "related_raw")})
            validation["related"] = rel["related"]
            status = "ok"
        elif res.status == "ok":
            status = "invalid_output"
        row = row_from_call(res, kind="label", attempt=0, prompt=prompt, started_at=started, ended_at=ended,
                            stdout_file=stdout_file, doc_id=doc.id, ingest_index=index,
                            budget_usd=self.budget, max_turns=self.max_turns, timeout_s=self.timeout,
                            wall_ms=clock.wall_ms, waited_ms=clock.waited_ms)
        row.update({"status": status, "structured_output": so, "validation": validation,
                    "prompt": None, "prompt_sha256": sha256_text(prompt),
                    "title": doc.title})
        if res.permission_denials:
            self.calls.append({**row, "status": "aborted_permission_denial"})
            raise R.HarnessError(f"permission_denials on label {index}: {res.permission_denials[:3]}")
        self.calls.append(row)
        return row

    def run(self) -> dict[str, Any]:
        done = {str(r.get("doc_id")) for r in self.calls.rows() if str(r.get("status")) == "ok"}
        book = TopicBook()
        processed: list[str] = []
        for row in self.calls.rows():                       # rebuild the topic list on resume
            so = row.get("structured_output") or {}
            if str(row.get("status")) == "ok" and so.get("topic"):
                book.resolve(str(so["topic"]), str(so.get("label") or ""), True)
                processed.append(self.titles.get(str(row.get("doc_id")), ""))
        n_run = 0
        for i, doc in enumerate(self.docs, start=1):
            if doc.id in done:
                continue
            if self.stop_after is not None and n_run >= self.stop_after:
                break
            row = self.label_one(i, doc, book, processed)
            n_run += 1
            if str(row.get("status")) == "ok":
                processed.append(doc.title)
            v = row.get("validation") or {}
            self.log(f"[{i}/{len(self.docs)}] {doc.id} {row.get('status')} topic={(row.get('structured_output') or {}).get('topic')} "
                     f"related={v.get('n_related')}/{v.get('n_related_raw')} invalid={v.get('n_invalid')} "
                     f"${float(row.get('cost_usd') or 0):.4f}")
        rows = self.calls.rows()
        summary = {
            "run_id": self.run_id, "arm": "C", "corpus": self.corpus, "n": self.n,
            "order_file": str(self.order_file), "related_candidates": self.related_candidates,
            "model": self.model, "effort": self.effort, "cli_version": self.cli_version,
            "system_prompt_sha256": sha256_text(self.system), "schema_sha256": sha256_text(self.schema),
            "n_topics": len(book.by_key),
            # counted from the stored rows, not from `book`: the resume rebuild replays
            # every earlier topic with is_new=True and would inflate the live counter
            "n_is_new_disagreements": sum(1 for r in rows
                                          if (r.get("validation") or {}).get("is_new_disagreement")),
            "n_invalid_related": sum(int((r.get("validation") or {}).get("n_invalid") or 0) for r in rows),
            "n_near_miss": sum(int((r.get("validation") or {}).get("n_near_miss") or 0) for r in rows),
            "totals": totals(rows), "finished_at": R.now_iso(),
        }
        write_json(self.meta / "run.json", summary)
        run_dir = self.out_root / "C" / self.corpus / str(self.n) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.calls.path, run_dir / "llm_calls.jsonl")
        write_json(run_dir / "manifest.json", {**summary, "calls_file": str(run_dir / "llm_calls.jsonl")})
        (run_dir / "system_prompt.txt").write_text(self.system, encoding="utf-8")
        self.log(f"arm C {self.run_id}: {summary['totals']['n_ok']} labelled, {summary['n_topics']} topics, "
                 f"${summary['totals']['cost_usd']:.4f} -> {run_dir}")
        return summary


def processed_ids(ids: Sequence[str], index: int) -> list[str]:
    """Ids visible to document number ``index`` (1-based) under ``--related-candidates processed``."""
    return list(ids[: max(0, index - 1)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wikiorg run-c", description="arm C: one-shot no-tools labeller")
    p.add_argument("--run-id", required=True)
    p.add_argument("--corpus", default="dbpedia")
    p.add_argument("--n", type=int, required=True, help="number of documents to label")
    p.add_argument("--corpus-n", type=int, help="corpus tier directory to draw from (default: --n)")
    p.add_argument("--order", default="1")
    p.add_argument("--corpus-dir")
    p.add_argument("--runs-root", default=str(RUNS_ROOT))
    p.add_argument("--model", default=R.MODEL)
    p.add_argument("--effort", default=R.EFFORT)
    p.add_argument("--related-candidates", choices=("full", "processed"), default="full")
    p.add_argument("--budget", type=float, default=BUDGET)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--timeout", type=int, default=TIMEOUT)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--out-root", help="where to write runs/C/... (default: <repo>/runs)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(R.clean_argv(argv))
    run = ArmCRun(run_id=args.run_id, corpus=args.corpus, n=args.n, order=args.order, corpus_n=args.corpus_n,
                  corpus_dir=Path(args.corpus_dir) if args.corpus_dir else None,
                  runs_root=Path(args.runs_root), model=args.model, effort=args.effort,
                  related_candidates=args.related_candidates, budget=args.budget,
                  max_turns=args.max_turns, timeout=args.timeout, stop_after=args.stop_after,
                  out_root=Path(args.out_root) if args.out_root else None)
    if args.dry_run:
        print(" ".join(run.cmd()[:6]), "...")
        print(run.system[:2000])
        print("---- user prompt ----")
        print(user_prompt(run.docs[0], [], [] if args.related_candidates == "processed" else None)[:2000])
        return 0
    run.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
