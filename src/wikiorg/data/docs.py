"""Shared document format used by every arm.

A corpus directory looks like::

    data/<corpus>/<N>/
        docs/<id>.md        frontmatter with ONLY ``id`` and ``title``, blank line, cleaned body
        gold.jsonl          one JSON object per document: {"id": ..., <gold fields>}
        orders/order-1.txt  one id per line (ingest order for arm B / C)
        manifest.json       provenance, statistics, acceptance checks

``doc_markdown(doc)`` produces the exact text every arm consumes: arm B copies it into
``raw/<id>.md``, arm C pastes it into the prompt, arm A embeds it.  Gold labels live only in
``gold.jsonl`` and must never appear in a document file (see ``tests/test_docs.py``).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import frontmatter

FRONTMATTER_KEYS = ("id", "title")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    body: str

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise ValueError(f"invalid doc id {self.id!r} (must be filesystem/URL safe)")
        if not self.title.strip():
            raise ValueError(f"doc {self.id}: empty title")


def _yaml_str(value: str) -> str:
    """A JSON string literal is a valid YAML double-quoted scalar (PyYAML accepts every JSON escape)."""
    return json.dumps(value, ensure_ascii=False)


def doc_markdown(doc: Doc) -> str:
    """The canonical on-disk text of a document.  Only ``id`` and ``title`` are in the frontmatter."""
    body = doc.body.strip("\n")
    return f"---\nid: {_yaml_str(doc.id)}\ntitle: {_yaml_str(doc.title)}\n---\n\n{body}\n"


def parse_doc(text: str) -> Doc:
    post = frontmatter.loads(text)
    meta = dict(post.metadata)
    extra = set(meta) - set(FRONTMATTER_KEYS)
    missing = set(FRONTMATTER_KEYS) - set(meta)
    if extra or missing:
        raise ValueError(f"frontmatter must contain exactly {FRONTMATTER_KEYS}; extra={extra} missing={missing}")
    return Doc(id=str(meta["id"]), title=str(meta["title"]), body=post.content.strip("\n"))


def docs_dir(corpus_dir: str | Path) -> Path:
    """Accept either ``data/<corpus>/<N>`` or ``data/<corpus>/<N>/docs``."""
    p = Path(corpus_dir)
    return p if p.name == "docs" else p / "docs"


def write_doc(corpus_dir: str | Path, doc: Doc) -> Path:
    d = docs_dir(corpus_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{doc.id}.md"
    path.write_text(doc_markdown(doc), encoding="utf-8")
    return path


def write_docs(corpus_dir: str | Path, docs: Iterable[Doc]) -> int:
    n = 0
    for doc in docs:
        write_doc(corpus_dir, doc)
        n += 1
    return n


def read_docs(corpus_dir: str | Path, ids: Iterable[str] | None = None) -> list[Doc]:
    """Read documents sorted by id (or in the order of ``ids`` when given)."""
    d = docs_dir(corpus_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"no docs directory at {d}")
    if ids is None:
        paths = sorted(d.glob("*.md"))
    else:
        paths = [d / f"{i}.md" for i in ids]
    docs = []
    for p in paths:
        doc = parse_doc(p.read_text(encoding="utf-8"))
        if doc.id != p.stem:
            raise ValueError(f"{p}: frontmatter id {doc.id!r} does not match file name")
        docs.append(doc)
    return docs


def read_gold(path: str | Path) -> dict[str, dict]:
    """``gold.jsonl`` (or a corpus dir containing it) -> {id: {field: value, ...}} without the id."""
    p = Path(path)
    if p.is_dir():
        p = p / "gold.jsonl"
    gold: dict[str, dict] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gid = str(row.pop("id"))
            if gid in gold:
                raise ValueError(f"duplicate id {gid} in {p}")
            gold[gid] = row
    return gold


def write_gold(path: str | Path, rows: Iterable[dict]) -> Path:
    p = Path(path)
    if p.is_dir() or p.suffix != ".jsonl":
        p = p / "gold.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return p


def read_order(path: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_order(path: str | Path, ids: Iterable[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return p


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_docs(docs: Iterable[Doc]) -> str:
    """One hash over the canonical markdown of all documents in id order."""
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.id):
        h.update(doc_markdown(doc).encode("utf-8"))
    return h.hexdigest()


def read_manifest(corpus_dir: str | Path) -> dict:
    p = Path(corpus_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def length_stats(docs: Iterable[Doc]) -> dict:
    docs = list(docs)
    lens = sorted(len(d.body) for d in docs)
    if not lens:
        return {"n": 0}
    words = [len(d.body.split()) for d in docs]
    q = lambda f: lens[min(len(lens) - 1, int(round(f * (len(lens) - 1))))]  # noqa: E731
    return {
        "n": len(lens),
        "chars_min": lens[0],
        "chars_p10": q(0.10),
        "chars_p25": q(0.25),
        "chars_median": q(0.50),
        "chars_p75": q(0.75),
        "chars_p90": q(0.90),
        "chars_max": lens[-1],
        "chars_mean": round(sum(lens) / len(lens), 1),
        "words_mean": round(sum(words) / len(words), 1),
    }
