"""Arm C adapter: one-shot no-tools LLM labelling rows -> WikiState.

Each call row (``calls.jsonl`` / a run's ``llm_calls.jsonl``) carries the
document id and the schema-validated ``structured_output``::

    {"topic": str, "is_new_topic": bool, "label": str,
     "one_line_summary": str, "related_titles": [str, ...]}

Rules (pre-registration, W2): topic identity is the casefold/NFKC match of the
``topic`` string (``norm_title``), trusted over ``is_new_topic``; the category
id is the slug of the first-seen spelling and its label the most frequent
``label`` string (falling back to the topic string).  ``related_titles`` are
resolved to corpus ids by **exact** normalised-title match only; self
references, repeats and non-corpus titles are dropped from ``related`` and
counted, while ``related_raw`` keeps the titles as written so that the link
validity report can classify near misses (difflib >= 0.9) separately.
Documents with no successful row get ``category = None`` (status ``failed``;
their page still exists in the rendered wiki, unlike a failed arm-B ingest).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from wikiorg.eval.state import Category, Page, WikiState

from .common import TitleIndex, dedupe, norm_title, slugify

OK_STATUSES = ("ok", "success")


def structured_output(row: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("structured_output", "output", "parsed", "result_json"):
        v = row.get(key)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                continue
        if isinstance(v, Mapping) and "topic" in v:
            return dict(v)
    if "topic" in row and "related_titles" in row:
        return {k: row[k] for k in ("topic", "is_new_topic", "label", "one_line_summary", "related_titles") if k in row}
    return None


def from_rows(rows: Iterable[Mapping[str, Any]], corpus_titles: Mapping[str, str], arm: str = "C",
              ok_statuses: tuple[str, ...] = OK_STATUSES, max_links: int | None = 5) -> WikiState:
    index = TitleIndex(corpus_titles)
    last_ok: dict[str, dict[str, Any]] = {}
    n_rows = n_failed = 0
    for row in rows:
        doc_id = row.get("doc_id") or row.get("id")
        if doc_id is None:
            continue
        doc_id = str(doc_id)
        n_rows += 1
        so = structured_output(row)
        status = str(row.get("status", "ok" if so else "error")).lower()
        if so is None or status not in ok_statuses:
            n_failed += 1
            continue
        last_ok[doc_id] = so
    topic_key_to_cid: dict[str, str] = {}
    labels_by_cid: dict[str, Counter] = defaultdict(Counter)
    first_topic_string: dict[str, str] = {}
    pages: dict[str, Page] = {}
    n_related_raw = n_resolved = n_self = n_dup = n_unresolved = n_new_flags = 0
    for pid in dedupe(list(corpus_titles) + list(last_ok)):
        title = corpus_titles.get(pid, pid)
        so = last_ok.get(pid)
        if so is None:
            pages[pid] = Page(title=title, category=None, extra={"status": "failed"})
            continue
        topic = str(so.get("topic") or "").strip()
        key = norm_title(topic)
        if key:
            cid = topic_key_to_cid.get(key)
            if cid is None:
                cid = slugify(topic)
                if cid in topic_key_to_cid.values():  # slug collision between distinct spellings
                    cid = f"{cid}-{len(topic_key_to_cid) + 1}"
                topic_key_to_cid[key] = cid
                first_topic_string[cid] = topic
            label = str(so.get("label") or "").strip()
            if label:
                labels_by_cid[cid][label] += 1
        else:
            cid = None
        if so.get("is_new_topic"):
            n_new_flags += 1
        raw = [str(t) for t in (so.get("related_titles") or [])]
        resolved: list[str] = []
        seen: set[str] = set()
        for t in raw:
            n_related_raw += 1
            rid = index.resolve(t)
            if rid is None:
                n_unresolved += 1
                continue
            if rid == pid:
                n_self += 1
                continue
            if rid in seen:
                n_dup += 1
                continue
            seen.add(rid)
            resolved.append(rid)
            n_resolved += 1
        if max_links is not None:
            resolved = resolved[:max_links]
        pages[pid] = Page(title=title, category=cid, related=resolved, related_raw=raw,
                          extra={"status": "ok", "is_new_topic": bool(so.get("is_new_topic")),
                                 "summary": str(so.get("one_line_summary") or "")})
    cats = {}
    for key, cid in topic_key_to_cid.items():
        cnt = labels_by_cid.get(cid)
        label = cnt.most_common(1)[0][0] if cnt else first_topic_string[cid]
        cats[cid] = Category(label=label, terms=[], parent=None, extra={"topic": first_topic_string[cid]})
    meta = {"n_rows": n_rows, "n_failed_rows": n_failed, "n_docs": len(corpus_titles), "n_ok_docs": len(last_ok),
            "n_topics": len(cats), "n_new_topic_flags": n_new_flags, "n_related_raw": n_related_raw,
            "n_related_resolved": n_resolved, "n_related_self": n_self, "n_related_duplicate": n_dup,
            "n_related_unresolved": n_unresolved, "ambiguous_titles": len(index.ambiguous)}
    return WikiState(arm=arm, pages=pages, categories=cats, extra_pages=[], meta=meta)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def from_jsonl(path: str | Path, corpus_titles: Mapping[str, str], arm: str = "C", run_id: str | None = None,
               kinds: tuple[str, ...] | None = None) -> WikiState:
    rows = read_jsonl(path)
    if run_id is not None:
        rows = [r for r in rows if str(r.get("run_id", run_id)) == run_id]
    if kinds is not None:
        rows = [r for r in rows if str(r.get("kind", kinds[0])) in kinds]
    st = from_rows(rows, corpus_titles, arm=arm)
    st.meta["source"] = str(path)
    st.meta["run_id"] = run_id
    return st
