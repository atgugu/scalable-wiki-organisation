"""Arm B adapter: an LLM-maintained wiki tree (the pre-registered schema in
``src/wikiorg/llm/schema/CLAUDE.md``) -> WikiState, plus a link audit.

Mapping (per the schema):
* ``wiki/sources/<doc_id>.md`` front matter ``topic: <slug>`` -> the document's
  category (``<slug>``); ``related: [<doc_id>, ...]`` -> links.  ``related`` is
  a list of document ids, so ``related_raw`` holds the ids *as written* and
  ``related`` only those whose source page exists (validity = file exists);
  self references and repeats are dropped from ``related``.
* ``wiki/topics/<slug>.md`` front matter ``title:`` (H1 fallback, then the
  de-hyphenated slug) -> the category label.
* Entity, concept, topic, index, log and overview pages -> ``extra_pages``
  (path, type, title, words) for the structure metrics.
* Corpus documents without a source page (failed or skipped ingests) get a page
  with ``category = None`` (singleton in the partition metrics).

The audit reports broken relative links, pages missing from ``index.md``
(the schema's orphan definition), pages with no inbound link at all, topic
pages without sources, sources whose topic page is missing, and page/word
counts by type.  W2's ``extract.py`` produces the same per-document rows as a
CSV; :func:`from_extract_csv` consumes that format too.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from wikiorg.eval.state import Category, Page, WikiState

from .common import as_str_list, dedupe, split_frontmatter

PAGE_DIRS = {"sources": "source", "topics": "topic", "entities": "entity", "concepts": "concept"}
ROOT_TYPES = {"index.md": "index", "log.md": "log", "overview.md": "overview"}
CONTENT_TYPES = ("source", "topic", "entity", "concept")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)
_MD_LINK_TARGET_RE = re.compile(r"\]\(([^)\s]+)\)")


@dataclass
class WikiPage:
    rel: str
    type: str
    fields: dict[str, Any]
    body: str
    title: str
    words: int
    links: list[str] = field(default_factory=list)  # raw targets

    @property
    def stem(self) -> str:
        return Path(self.rel).stem


def find_wiki_root(path: str | Path) -> Path:
    p = Path(path)
    if (p / "wiki").is_dir():
        return p / "wiki"
    if p.is_dir() and (p.name == "wiki" or (p / "index.md").exists() or (p / "sources").is_dir()):
        return p
    raise FileNotFoundError(f"no wiki tree at {p}")


def _page_type(rel: Path) -> str:
    if len(rel.parts) == 1:
        return ROOT_TYPES.get(rel.name, "other")
    return PAGE_DIRS.get(rel.parts[0], "other")


def read_wiki_tree(root: str | Path) -> dict[str, WikiPage]:
    root = find_wiki_root(root)
    pages: dict[str, WikiPage] = {}
    for f in sorted(root.rglob("*.md")):
        rel = f.relative_to(root)
        text = f.read_text(encoding="utf-8", errors="replace")
        fields, body = split_frontmatter(text)
        m = _H1_RE.search(body)
        title = str(fields.get("title") or (m.group(1) if m else "") or "").strip()
        links = [t for _, t in _LINK_RE.findall(text)]
        pages[rel.as_posix()] = WikiPage(rel=rel.as_posix(), type=_page_type(rel), fields=fields, body=body,
                                          title=title, words=len(body.split()), links=links)
    return pages


def target_id(raw: Any) -> str:
    """``[Danube](../sources/d1.md)`` / ``sources/d1.md`` / ``d1.md`` / ``d1`` -> ``d1``."""
    s = str(raw or "").strip().strip("'\"")
    m = _MD_LINK_TARGET_RE.search(s)
    if m:
        s = m.group(1)
    s = s.split("#", 1)[0]
    s = s.rsplit("/", 1)[-1]
    if s.endswith(".md"):
        s = s[:-3]
    return s.strip()


def _resolve(rel_from: str, target: str) -> str | None:
    """Resolve a relative markdown link target against the linking page's directory."""
    t = target.split("#", 1)[0].strip()
    if not t or t.startswith(("http://", "https://", "mailto:")):
        return None
    base = Path(rel_from).parent
    parts: list[str] = list(base.parts)
    for seg in Path(t).parts:
        if seg == "..":
            if parts:
                parts.pop()
        elif seg not in (".", ""):
            parts.append(seg)
    return Path(*parts).as_posix() if parts else None


def link_audit(pages: Mapping[str, WikiPage]) -> dict[str, Any]:
    existing = set(pages)
    broken: list[dict[str, str]] = []
    inbound: dict[str, set[str]] = defaultdict(set)
    index_listed: set[str] = set()
    n_links = 0
    for rel, page in pages.items():
        for raw in page.links:
            tgt = _resolve(rel, raw)
            if tgt is None:
                continue
            n_links += 1
            if tgt in existing:
                inbound[tgt].add(rel)
                if page.type == "index":
                    index_listed.add(tgt)
            elif tgt.startswith("raw/") or tgt.startswith("../raw/"):
                continue  # source: pointers are not wiki links
            else:
                broken.append({"from": rel, "target": raw})
    content = {rel: p for rel, p in pages.items() if p.type in CONTENT_TYPES}
    topics_used: Counter = Counter()
    sources_missing_topic_page: list[str] = []
    for rel, p in content.items():
        if p.type == "source":
            slug = target_id(p.fields.get("topic"))
            if slug:
                topics_used[slug] += 1
                if f"topics/{slug}.md" not in existing:
                    sources_missing_topic_page.append(rel)
    topics_without_sources = [rel for rel, p in content.items() if p.type == "topic" and topics_used.get(p.stem, 0) == 0]
    by_type: Counter = Counter(p.type for p in pages.values())
    words_by_type: Counter = Counter()
    for p in pages.values():
        words_by_type[p.type] += p.words
    not_in_index = sorted(rel for rel in content if rel not in index_listed)
    no_inbound = sorted(rel for rel in content if not (inbound.get(rel) - {rel} if inbound.get(rel) else set()))
    return {
        "n_pages": len(pages), "n_links": n_links, "n_broken_links": len(broken), "broken_links": broken,
        "has_index": "index.md" in existing,
        "n_content_pages": len(content), "n_not_in_index": len(not_in_index), "not_in_index": not_in_index,
        "index_coverage": (1 - len(not_in_index) / len(content)) if content else float("nan"),
        "n_no_inbound": len(no_inbound), "no_inbound": no_inbound,
        "topics_without_sources": sorted(topics_without_sources),
        "sources_missing_topic_page": sorted(sources_missing_topic_page),
        "pages_by_type": dict(by_type), "words_by_type": dict(words_by_type),
        "total_words": int(sum(words_by_type.values())),
    }


def from_wiki_dir(path: str | Path, corpus_ids: Iterable[str] | None = None, corpus_titles: Mapping[str, str] | None = None,
                  arm: str = "B", max_links: int | None = 5) -> WikiState:
    """Build the WikiState (with ``meta["audit"]``) from a wiki tree."""
    pages = read_wiki_tree(path)
    sources = {p.stem: p for p in pages.values() if p.type == "source"}
    topics = {p.stem: p for p in pages.values() if p.type == "topic"}
    ids = list(dedupe([str(i) for i in corpus_ids])) if corpus_ids is not None else sorted(sources)
    for s in sources:
        if s not in ids:
            ids.append(s)
    corpus_titles = dict(corpus_titles or {})
    state_pages: dict[str, Page] = {}
    cats: dict[str, Category] = {}
    for pid in ids:
        sp = sources.get(pid)
        title = corpus_titles.get(pid) or (sp.title if sp else "") or pid
        if sp is None:
            state_pages[pid] = Page(title=title, category=None, related=[], related_raw=[], extra={"status": "missing"})
            continue
        slug = target_id(sp.fields.get("topic")) or None
        raw = [target_id(r) for r in as_str_list(sp.fields.get("related"))]
        raw = [r for r in raw if r]
        resolved = dedupe(r for r in raw if r != pid and r in sources)
        if max_links is not None:
            resolved = resolved[:max_links]
        extra = {"status": "ok", "entities": as_str_list(sp.fields.get("entities")),
                 "concepts": as_str_list(sp.fields.get("concepts")), "ingest": sp.fields.get("ingest"),
                 "topic_page_exists": bool(slug and slug in topics), "words": sp.words}
        state_pages[pid] = Page(title=title, category=slug, related=resolved, related_raw=raw, extra=extra)
        if slug and slug not in cats:
            tp = topics.get(slug)
            label = (tp.title if tp else "") or slug.replace("-", " ")
            cats[slug] = Category(label=label, terms=[], parent=None, extra={"page_exists": tp is not None})
    for slug, tp in topics.items():
        cats.setdefault(slug, Category(label=tp.title or slug.replace("-", " "), extra={"page_exists": True}))
    extra_pages = [{"path": p.rel, "type": p.type, "title": p.title, "words": p.words}
                   for p in pages.values() if p.type != "source"]
    audit = link_audit(pages)
    meta = {"audit": audit, "n_sources": len(sources), "n_topics": len(topics),
            "n_missing_sources": sum(1 for pid in ids if pid not in sources), "source": str(find_wiki_root(path))}
    return WikiState(arm=arm, pages=state_pages, categories=cats, extra_pages=extra_pages, meta=meta)


def _parse_list_cell(v: Any) -> list[str]:
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except json.JSONDecodeError:
            return as_str_list(s)
    return [x.strip() for x in re.split(r"[;|,]", s) if x.strip()]


def from_extract_csv(path: str | Path, corpus_ids: Iterable[str] | None = None, corpus_titles: Mapping[str, str] | None = None,
                     arm: str = "B", max_links: int | None = 5) -> WikiState:
    """WikiState from W2's per-document extract CSV.

    Expected columns (tolerant): ``doc_id``, ``category`` (topic slug),
    ``label`` (topic title), ``related`` (list of ids; JSON or ``;``-joined),
    optional ``related_raw``, ``valid``/``related_valid`` (list of ids whose
    page exists), ``title``, ``status``.
    """
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))
    by_id = {str(r.get("doc_id") or r.get("id")): r for r in rows}
    ids = list(dedupe([str(i) for i in corpus_ids])) if corpus_ids is not None else sorted(by_id)
    for i in by_id:
        if i not in ids:
            ids.append(i)
    corpus_titles = dict(corpus_titles or {})
    pages: dict[str, Page] = {}
    cats: dict[str, Category] = {}
    for pid in ids:
        r = by_id.get(pid)
        title = corpus_titles.get(pid) or (r.get("title") if r else "") or pid
        if r is None or str(r.get("status", "ok")).lower() in ("missing", "failed", "error"):
            pages[pid] = Page(title=title, category=None, extra={"status": "missing"})
            continue
        cat = target_id(r.get("category") or r.get("topic")) or None
        raw = _parse_list_cell(r.get("related_raw") or r.get("related"))
        valid_col = r.get("related_valid") or r.get("valid")
        valid = set(_parse_list_cell(valid_col)) if valid_col not in (None, "") else set(ids)
        resolved = dedupe(x for x in (target_id(x) for x in raw) if x and x != pid and x in valid and x in set(ids))
        if max_links is not None:
            resolved = resolved[:max_links]
        pages[pid] = Page(title=title, category=cat, related=resolved, related_raw=[target_id(x) for x in raw],
                          extra={"status": "ok"})
        if cat and cat not in cats:
            cats[cat] = Category(label=str(r.get("label") or cat.replace("-", " ")))
    return WikiState(arm=arm, pages=pages, categories=cats, extra_pages=[], meta={"source": str(path)})
