"""Turn an arm-B wiki snapshot into the rows the evaluation consumes.

Input: a git ref in the run's repository (``ingest-0100``, ``HEAD``, a sha), a
``.tar.gz`` snapshot, or a plain directory.  Output, under ``extract/``:

``<run_id>_N<n>.csv``
    one row per corpus document, in the column contract
    ``wikiorg.adapters.arm_b.from_extract_csv`` reads:
    ``doc_id,title,status,category,label,related,related_raw,related_valid,
    entities,concepts,ingest,topic_page_exists,words``.
    Documents with no source page get ``status=missing`` so the partition
    metrics score them as singletons (``configs/prereg.yaml``).
``<run_id>_N<n>.audit.json``
    the link audit and page/word counts from
    ``wikiorg.adapters.arm_b.link_audit`` (broken links, index coverage,
    orphans, topics without sources) plus the run's call totals.

The adapter is imported rather than reimplemented so the CSV can never drift
from what the evaluation expects; ``tests/test_llm_extract.py`` additionally
asserts that the CSV round-trips to the same ``WikiState`` as reading the wiki
tree directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapters.arm_b import find_wiki_root, from_wiki_dir, link_audit, read_wiki_tree, target_id
from ..adapters.common import as_str_list
from ..data.docs import read_docs, read_order, sha256_file
from .arm_b import REPO, RUNS_ROOT, SCHEMA_PATH, GitStore, page_counts
from .calls import totals, write_json
from .runner import EFFORT as R_EFFORT
from .runner import MODEL as R_MODEL
from .runner import clean_argv, now_iso, read_jsonl

COLUMNS = ("doc_id", "title", "status", "category", "label", "related", "related_raw",
           "related_valid", "entities", "concepts", "ingest", "topic_page_exists", "words",
           "source_page", "n_related_raw", "n_related_valid", "n_related_dropped")


# ------------------------------------------------------------------------------------------------
# materialising a snapshot
# ------------------------------------------------------------------------------------------------
def materialise(source: str | Path, ref: str | None = None, gitdir: str | Path | None = None,
                dest: str | Path | None = None) -> tuple[Path, Path | None]:
    """``(directory containing wiki/, tempdir to clean up)``.

    ``source`` may be a directory (a live run dir or an already-extracted
    snapshot), a ``.tar``/``.tar.gz`` produced by :meth:`GitStore.archive`, or a
    git directory when ``ref`` is given.
    """
    src = Path(source)
    if gitdir or (ref and src.suffix == ".git") or (ref and (src / "HEAD").exists()):
        gd = Path(gitdir or src)
        tmp = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="wikiorg-extract-"))
        GitStore(tmp, gd).extract_ref(ref or "HEAD", tmp)
        return tmp, (None if dest else tmp)
    if src.is_file() and (src.name.endswith(".tar.gz") or src.name.endswith(".tgz") or src.suffix == ".tar"):
        tmp = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="wikiorg-extract-"))
        with tarfile.open(src) as tf:
            tf.extractall(tmp, filter="data")                     # our own archive; "data" is the safe filter
        return tmp, (None if dest else tmp)
    if src.is_dir():
        return src, None
    raise FileNotFoundError(f"cannot materialise {src} (ref={ref!r})")


# ------------------------------------------------------------------------------------------------
# rows
# ------------------------------------------------------------------------------------------------
def _join(values: Sequence[str]) -> str:
    """JSON so a title or slug containing a comma cannot split a cell."""
    return json.dumps(list(values), ensure_ascii=False)


def doc_rows(wiki_dir: str | Path, corpus_ids: Sequence[str],
             corpus_titles: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """One row per corpus document, in the adapter's CSV contract."""
    pages = read_wiki_tree(wiki_dir)
    sources = {p.stem: p for p in pages.values() if p.type == "source"}
    topics = {p.stem: p for p in pages.values() if p.type == "topic"}
    titles = dict(corpus_titles or {})
    ids = list(corpus_ids) + [s for s in sorted(sources) if s not in set(corpus_ids)]
    rows: list[dict[str, Any]] = []
    for pid in ids:
        sp = sources.get(pid)
        if sp is None:
            rows.append({"doc_id": pid, "title": titles.get(pid, ""), "status": "missing",
                         "category": "", "label": "", "related": _join([]), "related_raw": _join([]),
                         "related_valid": _join([]), "entities": _join([]), "concepts": _join([]),
                         "ingest": "", "topic_page_exists": "", "words": 0, "source_page": "",
                         "n_related_raw": 0, "n_related_valid": 0, "n_related_dropped": 0})
            continue
        slug = target_id(sp.fields.get("topic"))
        tp = topics.get(slug)
        raw = [target_id(r) for r in as_str_list(sp.fields.get("related"))]
        raw = [r for r in raw if r]
        valid: list[str] = []
        for r in raw:
            if r != pid and r in sources and r not in valid:
                valid.append(r)
        rows.append({
            "doc_id": pid,
            "title": titles.get(pid) or sp.title or pid,
            "status": "ok",
            "category": slug,
            "label": (tp.title if tp else "") or (slug.replace("-", " ") if slug else ""),
            "related": _join(valid),
            "related_raw": _join(raw),
            "related_valid": _join(valid),
            "entities": _join(as_str_list(sp.fields.get("entities"))),
            "concepts": _join(as_str_list(sp.fields.get("concepts"))),
            "ingest": sp.fields.get("ingest", ""),
            "topic_page_exists": bool(slug and slug in topics),
            "words": sp.words,
            "source_page": sp.rel,
            "n_related_raw": len(raw),
            "n_related_valid": len(valid),
            "n_related_dropped": len(raw) - len(valid),
        })
    return rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in COLUMNS})
    return p


def audit(wiki_dir: str | Path) -> dict[str, Any]:
    pages = read_wiki_tree(wiki_dir)
    out = link_audit(pages)
    out["page_counts"] = page_counts(find_wiki_root(wiki_dir))
    return out


# ------------------------------------------------------------------------------------------------
# top level
# ------------------------------------------------------------------------------------------------
def extract(source: str | Path, out_dir: str | Path, run_id: str, n: int | None = None,
            ref: str | None = None, gitdir: str | Path | None = None,
            corpus_dir: str | Path | None = None, order_file: str | Path | None = None,
            calls_path: str | Path | None = None, run_dir: str | Path | None = None,
            arm: str = "B", extra_manifest: Mapping[str, Any] | None = None,
            log=print) -> dict[str, Any]:
    materialised, tmp = materialise(source, ref=ref, gitdir=gitdir)
    try:
        wiki_root = find_wiki_root(materialised)
        ids: list[str] = []
        titles: dict[str, str] = {}
        if corpus_dir:
            docs = read_docs(corpus_dir)
            titles = {d.id: d.title for d in docs}
            ids = read_order(order_file)[:n] if order_file else sorted(titles)
            if n:
                ids = ids[:n]
        rows = doc_rows(wiki_root, ids, titles)
        n_docs = n or len([r for r in rows if r["status"] == "ok"]) or len(rows)
        out_dir = Path(out_dir)
        csv_path = write_csv(out_dir / f"{run_id}_N{n_docs}.csv", rows)
        report = {
            "run_id": run_id, "n": n_docs, "ref": ref, "source": str(source),
            "wiki_root": str(wiki_root), "csv": str(csv_path),
            "n_rows": len(rows), "n_ok": sum(1 for r in rows if r["status"] == "ok"),
            "n_missing": sum(1 for r in rows if r["status"] == "missing"),
            "n_categories": len({r["category"] for r in rows if r["category"]}),
            "audit": audit(wiki_root),
        }
        if calls_path and Path(calls_path).exists():
            report["calls"] = totals(read_jsonl(calls_path))
            shutil.copyfile(calls_path, out_dir / f"{run_id}_N{n_docs}.calls.jsonl")
        if run_dir is not None:
            report["run_dir"] = str(write_run_dir(Path(run_dir), wiki_root, ids, titles, run_id=run_id,
                                                  n=n_docs, arm=arm, calls_path=calls_path,
                                                  audit=report, extra=extra_manifest))
        write_json(out_dir / f"{run_id}_N{n_docs}.audit.json", report)
        log(f"extract {run_id} N={n_docs}: {report['n_ok']} source pages, "
            f"{report['n_categories']} topics, {report['audit']['n_broken_links']} broken links, "
            f"index coverage {report['audit']['index_coverage']:.2f} -> {csv_path}")
        return report
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def write_run_dir(run_dir: Path, wiki_root: Path, ids: Sequence[str], titles: Mapping[str, str],
                  run_id: str, n: int, arm: str = "B", calls_path: str | Path | None = None,
                  audit: Mapping[str, Any] | None = None,
                  extra: Mapping[str, Any] | None = None) -> Path:
    """The run directory the plan's layout expects: ``wiki_state.json`` (which
    ``wikiorg.eval.report.load_state`` prefers over any adapter path),
    ``manifest.json`` and a copy of the call ledger."""
    run_dir.mkdir(parents=True, exist_ok=True)
    state = from_wiki_dir(wiki_root, corpus_ids=ids, corpus_titles=titles, arm=arm)
    state.save(run_dir / "wiki_state.json")
    manifest: dict[str, Any] = {
        "run_id": run_id, "arm": arm, "n": n, "n_docs": len(ids),
        "claude_md_sha256": sha256_file(SCHEMA_PATH) if SCHEMA_PATH.exists() else None,
        "model": R_MODEL, "effort": R_EFFORT, "wiki_source": str(wiki_root),
        "n_categories": len(state.categories),
        "created_at": now_iso(),
    }
    if calls_path and Path(calls_path).exists():
        shutil.copyfile(calls_path, run_dir / "llm_calls.jsonl")
        manifest["calls_file"] = str(run_dir / "llm_calls.jsonl")
        manifest["totals"] = totals(read_jsonl(calls_path))
    if audit:
        manifest["audit"] = {k: v for k, v in audit.get("audit", {}).items()
                             if not isinstance(v, list)}
    manifest.update(dict(extra or {}))
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wikiorg extract", description="arm B snapshot -> evaluation rows")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default=str(RUNS_ROOT))
    p.add_argument("--source", help="run dir, snapshot tar.gz or git dir (default: <runs-root>/<run-id>.git)")
    p.add_argument("--ref", default="HEAD", help="git ref (ingest-0100, HEAD, sha)")
    p.add_argument("--corpus", default="dbpedia")
    p.add_argument("--n", type=int, help="corpus size for the id list and the output name")
    p.add_argument("--corpus-dir")
    p.add_argument("--order", default="1")
    p.add_argument("--out", help="output directory (default: <repo>/results/extract)")
    p.add_argument("--run-dir", help="also write wiki_state.json + manifest.json here "
                                     "(default: <repo>/runs/B/<corpus>/<n>/order-<k>)")
    p.add_argument("--no-run-dir", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(clean_argv(argv))
    runs_root = Path(args.runs_root)
    gitdir = runs_root / f"{args.run_id}.git"
    source = Path(args.source) if args.source else gitdir
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else (
        REPO / "data" / args.corpus / str(args.n) if args.n else None)
    order_file = None
    if corpus_dir:
        cand = Path(args.order) if Path(args.order).exists() else corpus_dir / "orders" / f"order-{args.order}.txt"
        order_file = cand if cand.exists() else None
    calls = runs_root / f"{args.run_id}.meta" / "calls.jsonl"
    run_dir = None
    if not args.no_run_dir:
        run_dir = Path(args.run_dir) if args.run_dir else (
            REPO / "runs" / "B" / args.corpus / str(args.n or "all") / f"order-{args.order}"
            if args.n else None)
    extract(source, Path(args.out) if args.out else REPO / "results" / "extract", args.run_id,
            n=args.n, ref=args.ref, gitdir=gitdir if source == gitdir else None,
            corpus_dir=corpus_dir, order_file=order_file, calls_path=calls, run_dir=run_dir,
            extra_manifest={"corpus": args.corpus, "order_file": str(order_file) if order_file else None,
                            "ref": args.ref, "source": str(source)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
