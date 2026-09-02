"""WikiState contract (consumer-side mirror) and gold-standard loading.

The producer of ``wiki_state.json`` is ``wikiorg.render`` (workstream W1).  The
on-disk contract, which every arm adapter writes and every metric reads, is::

    {"arm": "A", 
     "pages": {id: {"title": str, "category": cid | null,
                    "related": [ids], "related_raw": [titles]}},
     "categories": {cid: {"label": str, "terms": [str], "parent": cid | null}},
     "extra_pages": [ {...} ]}

This module keeps a local dataclass mirror so that the evaluation package works
even when ``wikiorg.render`` is not importable (or defines its class with a
different constructor).  ``coerce_state`` accepts any of: this mirror, a plain
``dict``, a path to ``wiki_state.json``, or an arbitrary object exposing
``pages``/``categories`` attributes (for example ``wikiorg.render.WikiState``).

Documents whose ``category`` is ``None`` (arms B/C may fail an ingest or leave a
document unclassified) are treated as *singleton clusters* by
:meth:`WikiState.partition` -- see the pre-registration
(``configs/prereg.yaml``): "Uncategorised B/C docs = singleton clusters".
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:  # producer-side class, optional
    from wikiorg.render import WikiState as RenderWikiState  # type: ignore
except Exception:  # pragma: no cover - depends on W1's code being present
    RenderWikiState = None  # type: ignore

SINGLETON_PREFIX = "__singleton__"
GOLD_LEVELS = ("l1", "l2", "l3", "class")
LINK_KEYS = ("links", "gold_links", "neighbors", "neighbours", "edges")
ID_KEYS = ("id", "doc_id", "pageid", "page_id")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key lookup so that dicts and objects are handled alike."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class Page:
    title: str = ""
    category: str | None = None
    related: list[str] = field(default_factory=list)
    related_raw: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"title": self.title, "category": self.category,
             "related": list(self.related), "related_raw": list(self.related_raw)}
        d.update(self.extra)
        return d

    @classmethod
    def from_any(cls, obj: Any) -> "Page":
        if isinstance(obj, Page):
            return obj
        known = {"title", "category", "related", "related_raw"}
        if isinstance(obj, dict):
            extra = {k: v for k, v in obj.items() if k not in known}
        elif dataclasses.is_dataclass(obj):
            extra = {k: v for k, v in dataclasses.asdict(obj).items() if k not in known}
        else:
            extra = {}
        cat = _get(obj, "category")
        return cls(title=str(_get(obj, "title", "") or ""),
                   category=None if cat is None else str(cat),
                   related=[str(x) for x in (_get(obj, "related", None) or [])],
                   related_raw=[str(x) for x in (_get(obj, "related_raw", None) or [])],
                   extra=extra)


@dataclass
class Category:
    label: str = ""
    terms: list[str] = field(default_factory=list)
    parent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"label": self.label, "terms": list(self.terms), "parent": self.parent}
        d.update(self.extra)
        return d

    @classmethod
    def from_any(cls, obj: Any) -> "Category":
        if isinstance(obj, Category):
            return obj
        known = {"label", "terms", "parent"}
        if isinstance(obj, dict):
            extra = {k: v for k, v in obj.items() if k not in known}
        elif dataclasses.is_dataclass(obj):
            extra = {k: v for k, v in dataclasses.asdict(obj).items() if k not in known}
        else:
            extra = {}
        parent = _get(obj, "parent")
        return cls(label=str(_get(obj, "label", "") or ""),
                   terms=[str(t) for t in (_get(obj, "terms", None) or [])],
                   parent=None if parent is None else str(parent),
                   extra=extra)


@dataclass
class WikiState:
    """Consumer-side mirror of the ``wiki_state.json`` contract."""

    arm: str = ""
    pages: dict[str, Page] = field(default_factory=dict)
    categories: dict[str, Category] = field(default_factory=dict)
    extra_pages: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "arm": self.arm,
            "pages": {pid: p.to_dict() for pid, p in self.pages.items()},
            "categories": {cid: c.to_dict() for cid, c in self.categories.items()},
            "extra_pages": list(self.extra_pages),
        }
        if self.meta:
            d["meta"] = dict(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WikiState":
        pages = {str(k): Page.from_any(v) for k, v in (d.get("pages") or {}).items()}
        cats = {str(k): Category.from_any(v) for k, v in (d.get("categories") or {}).items()}
        known = {"arm", "pages", "categories", "extra_pages", "meta"}
        meta = dict(d.get("meta") or {})
        meta.update({k: v for k, v in d.items() if k not in known})
        return cls(arm=str(d.get("arm", "") or ""), pages=pages, categories=cats,
                   extra_pages=list(d.get("extra_pages") or []), meta=meta)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WikiState":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- views -----------------------------------------------------------------
    def ids(self) -> list[str]:
        return sorted(self.pages)

    def title_of(self, pid: str) -> str:
        p = self.pages.get(pid)
        return p.title if p else ""

    def partition(self, ids: Iterable[str] | None = None,
                  singleton_uncategorised: bool = True) -> dict[str, str]:
        """Map document id -> cluster id.

        Uncategorised documents (``category is None`` or a category id that does
        not exist) become singleton clusters ``__singleton__<id>`` when
        ``singleton_uncategorised`` is true (the pre-registered rule); otherwise
        they are dropped from the mapping.  Documents in ``ids`` that the state
        does not know at all (e.g. a failed ingest that produced no page) are
        treated the same way as uncategorised ones.
        """
        ids = list(ids) if ids is not None else self.ids()
        out: dict[str, str] = {}
        for pid in ids:
            p = self.pages.get(pid)
            cat = p.category if p is not None else None
            if cat is None:
                if singleton_uncategorised:
                    out[pid] = SINGLETON_PREFIX + pid
            else:
                out[pid] = cat
        return out

    def members(self, include_singletons: bool = False) -> dict[str, list[str]]:
        """Category id -> member ids (sorted)."""
        m: dict[str, list[str]] = defaultdict(list)
        for pid, cl in self.partition(singleton_uncategorised=include_singletons).items():
            m[cl].append(pid)
        for cl in self.categories:
            m.setdefault(cl, [])
        return {k: sorted(v) for k, v in m.items()}

    def k_produced(self) -> int:
        """Number of non-empty produced categories (singletons excluded)."""
        used = {p.category for p in self.pages.values() if p.category is not None}
        return len(used)

    def n_uncategorised(self) -> int:
        return sum(1 for p in self.pages.values() if p.category is None)

    def links(self) -> dict[str, list[str]]:
        """Document id -> related ids restricted to known pages, self-links and
        duplicates removed, order preserved."""
        out: dict[str, list[str]] = {}
        for pid, p in self.pages.items():
            seen: set[str] = set()
            lst: list[str] = []
            for r in p.related:
                if r == pid or r in seen or r not in self.pages:
                    continue
                seen.add(r)
                lst.append(r)
            out[pid] = lst
        return out


def coerce_state(obj: Any) -> WikiState:
    """Return a :class:`WikiState` mirror from anything that carries the contract."""
    if isinstance(obj, WikiState):
        return obj
    if isinstance(obj, (str, Path)):
        return WikiState.load(obj)
    if isinstance(obj, dict):
        return WikiState.from_dict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        d = to_dict()
        if isinstance(d, dict):
            return WikiState.from_dict(d)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return WikiState.from_dict(dataclasses.asdict(obj))
    pages = _get(obj, "pages")
    if pages is None:
        raise TypeError(f"cannot interpret {type(obj).__name__} as a WikiState")
    d = {"arm": _get(obj, "arm", ""), "pages": dict(pages),
         "categories": dict(_get(obj, "categories", {}) or {}),
         "extra_pages": list(_get(obj, "extra_pages", []) or [])}
    return WikiState.from_dict(d)


# --------------------------------------------------------------------------
# Gold standard
# --------------------------------------------------------------------------
@dataclass
class Gold:
    """Gold classes (one dict per level) and, optionally, gold links.

    ``labels[level][doc_id] = class name``; ``links[doc_id] = set(doc_ids)`` is
    the *undirected union* of the gold hyperlinks restricted to the corpus.
    """

    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    titles: dict[str, str] = field(default_factory=dict)
    links: dict[str, set[str]] | None = None
    texts: dict[str, str] = field(default_factory=dict)

    @property
    def levels(self) -> list[str]:
        return [lvl for lvl in GOLD_LEVELS if lvl in self.labels] + \
               [lvl for lvl in self.labels if lvl not in GOLD_LEVELS]

    def ids(self) -> list[str]:
        s: set[str] = set(self.titles)
        for m in self.labels.values():
            s.update(m)
        return sorted(s)

    def classes(self, level: str, ids: Iterable[str] | None = None) -> list[str]:
        ids = list(ids) if ids is not None else self.ids()
        m = self.labels[level]
        return [m[i] for i in ids]

    def class_counts(self, level: str, ids: Iterable[str] | None = None) -> Counter:
        return Counter(self.classes(level, ids))

    def has_links(self) -> bool:
        return bool(self.links)

    def degree(self, pid: str) -> int:
        return len(self.links.get(pid, ())) if self.links else 0

    def to_rows(self) -> list[dict[str, Any]]:
        rows = []
        for pid in self.ids():
            row: dict[str, Any] = {"id": pid, "title": self.titles.get(pid, "")}
            for lvl, m in self.labels.items():
                if pid in m:
                    row[lvl] = m[pid]
            if self.links is not None:
                row["links"] = sorted(self.links.get(pid, ()))
            rows.append(row)
        return rows


def _row_id(row: dict[str, Any]) -> str:
    for k in ID_KEYS:
        if k in row and row[k] is not None:
            return str(row[k])
    raise KeyError(f"gold row without an id field ({ID_KEYS}): {row!r}")


def gold_from_rows(rows: Iterable[dict[str, Any]], symmetrise_links: bool = True) -> Gold:
    """Build a :class:`Gold` from jsonl-style rows (tolerant to key names)."""
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    titles: dict[str, str] = {}
    texts: dict[str, str] = {}
    raw_links: dict[str, list[str]] = {}
    any_links = False
    for row in rows:
        pid = _row_id(row)
        titles[pid] = str(row.get("title", "") or "")
        if row.get("text"):
            texts[pid] = str(row["text"])
        for lvl in GOLD_LEVELS:
            if row.get(lvl) is not None:
                labels[lvl][pid] = str(row[lvl])
        for lk in LINK_KEYS:
            if lk in row and row[lk] is not None:
                any_links = True
                raw_links[pid] = [str(x) for x in row[lk]]
                break
    links: dict[str, set[str]] | None = None
    if any_links:
        known = set(titles)
        links = {pid: set() for pid in known}
        for src, dsts in raw_links.items():
            for dst in dsts:
                if dst == src or dst not in known:
                    continue
                links[src].add(dst)
                if symmetrise_links:
                    links[dst].add(src)
    return Gold(labels=dict(labels), titles=titles, links=links, texts=texts)


def gold_from_mapping(mapping: dict[str, dict[str, Any]]) -> Gold:
    """From ``wikiorg.data.docs.read_gold`` output (``{id: {field: value}}``)."""
    return gold_from_rows([{"id": pid, **fields} for pid, fields in mapping.items()])


def read_gold(path: str | Path) -> Gold:
    """Read ``gold.jsonl`` (one JSON object per line; ``id``, ``title``, class
    fields among ``l1/l2/l3/class`` and optionally ``links``).  A corpus
    directory containing ``gold.jsonl`` is accepted too."""
    path = Path(path)
    if path.is_dir():
        path = path / "gold.jsonl"
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return gold_from_rows(rows)


def write_gold(gold: Gold, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in gold.to_rows():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def common_ids(state: WikiState, gold: Gold, level: str | None = None) -> list[str]:
    """Documents present in the gold (and, if a level is given, labelled at it).

    The *gold* defines the evaluation universe: a document the arm produced no
    page for still counts (as a singleton cluster); a page that is not in the
    gold (e.g. an extra non-source page) is ignored.
    """
    ids = gold.ids()
    if level is not None:
        m = gold.labels[level]
        ids = [i for i in ids if i in m]
    return ids
