"""Arm A outputs (cluster labels, c-TF-IDF terms, MMR links) -> WikiState contract."""
from __future__ import annotations

import numpy as np

from ..data.docs import Doc
from ..label import make_label
from ..render import WikiState


def cid_for(index: int, k: int, prefix: str = "c") -> str:
    width = max(2, len(str(max(0, k - 1))))
    return f"{prefix}{index:0{width}d}"


def build_wiki_state(docs: list[Doc], labels: np.ndarray, terms: dict, links: list[list[int]],
                     parents: np.ndarray | None = None, parent_terms: dict | None = None,
                     arm: str = "A", label_terms: int = 3, top_terms: int = 10) -> WikiState:
    """``terms``: {leaf cluster index: [(term, weight), ...]}; ``parents``: leaf index -> top index;
    ``parent_terms``: {top index: [(term, weight), ...]} (two-level runs only)."""
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if labels.size else 0
    state = WikiState(arm=arm)
    if parents is not None:
        n_top = int(np.max(parents)) + 1 if len(parents) else 0
        for t in range(n_top):
            tt = (parent_terms or {}).get(t, [])
            state.categories[cid_for(t, n_top, "p")] = {"label": make_label(tt, label_terms),
                                                        "terms": [w for w, _ in tt[:top_terms]], "parent": None}
    for c in range(k):
        tt = terms.get(c, [])
        parent = cid_for(int(parents[c]), int(np.max(parents)) + 1, "p") if parents is not None else None
        state.categories[cid_for(c, k)] = {"label": make_label(tt, label_terms),
                                           "terms": [w for w, _ in tt[:top_terms]], "parent": parent}
    for i, doc in enumerate(docs):
        rel = [docs[j].id for j in links[i]]
        state.pages[doc.id] = {"title": doc.title, "category": cid_for(int(labels[i]), k),
                               "related": rel, "related_raw": [docs[j].title for j in links[i]]}
    return state


# -------------------------------------------------------------------------------------------------
# Generic builder (workstream W3): WikiState from plain arrays / dicts, independent of the pipeline
# classes, so that evaluation code, tests and ablation sweeps can assemble states directly.
# ``build_wiki_state`` above is the producer-side path used by ``wikiorg run-a``.
# -------------------------------------------------------------------------------------------------
from pathlib import Path as _Path  # noqa: E402
from typing import Any as _Any, Mapping as _Mapping, Sequence as _Sequence  # noqa: E402

from wikiorg.eval.state import Category as _Category, Page as _Page, WikiState as _EvalWikiState  # noqa: E402

from .common import dedupe as _dedupe  # noqa: E402


def category_id(label: _Any) -> str:
    """Integer cluster labels -> ``c000``-style ids; strings are kept."""
    if isinstance(label, bool):
        return str(label)
    if isinstance(label, int) or (isinstance(label, float) and float(label).is_integer()):
        return f"c{int(label):03d}"
    return str(label)


def build_state(ids: _Sequence[str], titles: _Mapping[str, str] | _Sequence[str], labels: _Mapping[str, _Any] | _Sequence[_Any],
                links: _Mapping[str, _Sequence[str]] | _Sequence[_Sequence[str]],
                terms: _Mapping[_Any, _Sequence[str]] | _Sequence[_Sequence[str]] | None = None,
                category_labels: _Mapping[_Any, str] | None = None, parents: _Mapping[_Any, _Any] | None = None,
                arm: str = "A", meta: _Mapping[str, _Any] | None = None, n_label_terms: int = 3, n_terms: int = 10) -> _EvalWikiState:
    """Assemble an evaluation-side WikiState (``wikiorg.eval.state.WikiState``).

    ``labels`` / ``links`` / ``titles`` may be dicts keyed by id or sequences aligned with ``ids``;
    ``terms`` maps a cluster label to its top terms (or is a list indexed by integer cluster label).
    The category label is ``category_labels[cluster]`` when given (arm A+), else the top
    ``n_label_terms`` terms joined by ", " (arm A), else ``cluster <id>``.
    """
    ids = [str(i) for i in ids]
    if not isinstance(titles, _Mapping):
        titles = dict(zip(ids, titles))
    if not isinstance(labels, _Mapping):
        labels = dict(zip(ids, labels))
    if not isinstance(links, _Mapping):
        links = dict(zip(ids, links))
    pages: dict[str, _Page] = {}
    for pid in ids:
        lab = labels.get(pid)
        rel = _dedupe(str(r) for r in (links.get(pid) or []) if str(r) != pid)
        pages[pid] = _Page(title=str(titles.get(pid, pid)), category=None if lab is None else category_id(lab),
                           related=rel, related_raw=[str(titles.get(r, r)) for r in rel])
    cats: dict[str, _Category] = {}
    used = {lab for lab in labels.values() if lab is not None}
    for lab in sorted(used, key=lambda x: (str(type(x)), x)):
        cid = category_id(lab)
        if isinstance(terms, _Mapping):
            t = list(terms.get(lab, terms.get(cid, [])))
        elif terms is not None and isinstance(lab, int) and 0 <= lab < len(terms):
            t = list(terms[lab])
        else:
            t = []
        t = [x[0] if isinstance(x, (tuple, list)) else str(x) for x in t][:n_terms]
        label = None
        if category_labels:
            label = category_labels.get(lab, category_labels.get(cid))
        if not label:
            label = ", ".join(t[:n_label_terms]) if t else f"cluster {cid}"
        parent = None
        if parents:
            p = parents.get(lab, parents.get(cid))
            parent = category_id(p) if p is not None else None
        cats[cid] = _Category(label=str(label), terms=t, parent=parent)
    if parents:
        for p in set(parents.values()):
            if p is not None and category_id(p) not in cats:
                name = category_labels.get(p) if category_labels else None
                cats[category_id(p)] = _Category(label=str(name or f"group {category_id(p)}"))
    return _EvalWikiState(arm=arm, pages=pages, categories=cats, extra_pages=[], meta=dict(meta or {}))


def from_run_dir(run_dir: str | _Path) -> _EvalWikiState:
    """Load a run's ``wiki_state.json`` (the renderer's output) as an evaluation-side WikiState."""
    run_dir = _Path(run_dir)
    path = run_dir if run_dir.suffix == ".json" else run_dir / "wiki_state.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _EvalWikiState.load(path)
