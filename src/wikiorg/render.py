"""WikiState contract and the browsable Markdown wiki.

``wiki_state.json`` (exactly these keys, byte-deterministic: sorted keys, no timestamps)::

    {"arm": "A",
     "pages": {id: {"title": str, "category": cid, "related": [ids], "related_raw": [titles]}},
     "categories": {cid: {"label": str, "terms": [str], "parent": cid | null}},
     "extra_pages": [ {"path": str, "title": str, "type": str}, ... ]}

Markdown wiki (relative links only, so it browses offline from any Markdown viewer)::

    wiki/index.md                      categories (hierarchy) with counts, then every page
    wiki/categories/<cid>-<slug>.md    label, terms, parent/children, member pages
    wiki/pages/<id>.md                 title, category, related pages, then the document text
    wiki/graph.json                    nodes (id, title, category) and related-link edges
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .data.docs import Doc

CONTRACT_KEYS = ("arm", "pages", "categories", "extra_pages")


@dataclass
class WikiState:
    arm: str
    pages: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    extra_pages: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"arm": self.arm, "pages": self.pages, "categories": self.categories, "extra_pages": self.extra_pages}

    @classmethod
    def from_dict(cls, d: dict) -> "WikiState":
        return cls(arm=d["arm"], pages=dict(d["pages"]), categories=dict(d["categories"]),
                   extra_pages=list(d.get("extra_pages", [])))

    def validate(self) -> list[str]:
        problems = []
        for pid, p in self.pages.items():
            for key in ("title", "category", "related", "related_raw"):
                if key not in p:
                    problems.append(f"page {pid}: missing {key}")
            if p.get("category") is not None and p["category"] not in self.categories:
                problems.append(f"page {pid}: unknown category {p['category']}")
            for r in p.get("related", []):
                if r not in self.pages:
                    problems.append(f"page {pid}: related {r} is not a page")
                if r == pid:
                    problems.append(f"page {pid}: links to itself")
        for cid, c in self.categories.items():
            for key in ("label", "terms", "parent"):
                if key not in c:
                    problems.append(f"category {cid}: missing {key}")
            if c.get("parent") is not None and c["parent"] not in self.categories:
                problems.append(f"category {cid}: unknown parent {c['parent']}")
        return problems


def dumps_state(state: WikiState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True, ensure_ascii=False, indent=1) + "\n"


def write_wiki_state(state: WikiState, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_state(state), encoding="utf-8")
    return p


def read_wiki_state(path: str | Path) -> WikiState:
    return WikiState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# -------------------------------------------------------------------------------------------------
# markdown
# -------------------------------------------------------------------------------------------------
def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0] or s[:max_len]
    return s or "untitled"


def _md_text(s: str) -> str:
    return s.replace("[", "(").replace("]", ")").replace("\n", " ")


_ACCIDENTAL_LINK_RE = re.compile(r"\]\(")


def _escape_body(text: str) -> str:
    """Verbatim source prose occasionally contains ``](`` by coincidence (ISA addressing notation
    such as ``[Rx](Rn)``, not a link): CommonMark reads that as a real link to a nonexistent file.
    Insert a backslash so it renders as literal text.  Applied only to the copy of the body placed
    in a wiki page, never to the corpus document itself (docs.py's contract stays byte-verbatim)."""
    return _ACCIDENTAL_LINK_RE.sub(r"]\\(", text)


def category_filename(cid: str, label: str) -> str:
    return f"{cid}-{slugify(label)}.md"


def _sorted_pages(state: WikiState, ids) -> list[str]:
    return sorted(ids, key=lambda i: (state.pages[i]["title"].casefold(), i))


def write_wiki(state: WikiState, docs: dict[str, Doc] | list[Doc], out_dir: str | Path, title: str = "Wiki",
               include_bodies: bool = True) -> Path:
    """Write the browsable Markdown wiki for ``state`` into ``out_dir``."""
    if not isinstance(docs, dict):
        docs = {d.id: d for d in docs}
    out = Path(out_dir)
    (out / "categories").mkdir(parents=True, exist_ok=True)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    for old in list((out / "categories").glob("*.md")) + list((out / "pages").glob("*.md")):
        old.unlink()

    cat_file = {cid: category_filename(cid, c["label"]) for cid, c in state.categories.items()}
    members: dict[str, list[str]] = {cid: [] for cid in state.categories}
    uncategorised: list[str] = []
    for pid, p in state.pages.items():
        (members[p["category"]] if p["category"] in members else uncategorised).append(pid)
    children: dict[str, list[str]] = {cid: [] for cid in state.categories}
    for cid, c in state.categories.items():
        if c.get("parent") is not None:
            children[c["parent"]].append(cid)

    def cat_total(cid: str) -> int:
        return len(members[cid]) + sum(cat_total(ch) for ch in children[cid])

    def cat_sort_key(cid: str):
        return (-cat_total(cid), state.categories[cid]["label"].casefold(), cid)

    # ---- index --------------------------------------------------------------------------------
    lines = [f"# {_md_text(title)}", "",
             f"{len(state.pages)} pages in {len(state.categories)} categories"
             + (f"; {len(state.extra_pages)} extra pages" if state.extra_pages else "") + ".", "",
             "## Categories", ""]

    def emit_category(cid: str, depth: int) -> None:
        c = state.categories[cid]
        lines.append(f"{'  ' * depth}- [{_md_text(c['label'])}](categories/{cat_file[cid]}) — {cat_total(cid)} pages")
        for ch in sorted(children[cid], key=cat_sort_key):
            emit_category(ch, depth + 1)

    for cid in sorted((c for c in state.categories if state.categories[c].get("parent") is None), key=cat_sort_key):
        emit_category(cid, 0)
    if uncategorised:
        lines += ["", f"Uncategorised: {len(uncategorised)} pages"]
    if state.extra_pages:
        lines += ["", "## Extra pages", ""]
        for ep in sorted(state.extra_pages, key=lambda e: (str(e.get("type")), str(e.get("title")))):
            lines.append(f"- {_md_text(str(ep.get('title')))} ({ep.get('type')}: `{ep.get('path')}`)")
    lines += ["", "## All pages", ""]
    for pid in _sorted_pages(state, state.pages):
        p = state.pages[pid]
        cat = p["category"]
        cat_link = (f" — [{_md_text(state.categories[cat]['label'])}](categories/{cat_file[cat]})"
                    if cat in cat_file else "")
        lines.append(f"- [{_md_text(p['title'])}](pages/{pid}.md){cat_link}")
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- categories -----------------------------------------------------------------------------
    for cid, c in state.categories.items():
        lines = [f"# {_md_text(c['label'])}", "", "[Index](../index.md)", ""]
        if c.get("terms"):
            lines += ["**Terms:** " + ", ".join(_md_text(t) for t in c["terms"]), ""]
        if c.get("parent") is not None:
            par = c["parent"]
            lines += [f"**Parent:** [{_md_text(state.categories[par]['label'])}]({cat_file[par]})", ""]
        if children[cid]:
            lines += ["## Subcategories", ""]
            for ch in sorted(children[cid], key=cat_sort_key):
                lines.append(f"- [{_md_text(state.categories[ch]['label'])}]({cat_file[ch]}) — {cat_total(ch)} pages")
            lines.append("")
        lines += [f"## Pages ({len(members[cid])})", ""]
        for pid in _sorted_pages(state, members[cid]):
            lines.append(f"- [{_md_text(state.pages[pid]['title'])}](../pages/{pid}.md)")
        (out / "categories" / cat_file[cid]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- pages ----------------------------------------------------------------------------------
    for pid, p in state.pages.items():
        cat = p["category"]
        lines = [f"# {_md_text(p['title'])}", "", "[Index](../index.md)", ""]
        if cat in cat_file:
            lines += [f"**Category:** [{_md_text(state.categories[cat]['label'])}](../categories/{cat_file[cat]})", ""]
        lines += ["## Related", ""]
        if p.get("related"):
            for r in p["related"]:
                if r in state.pages:
                    lines.append(f"- [{_md_text(state.pages[r]['title'])}](../pages/{r}.md)")
        elif p.get("related_raw"):
            for r in p["related_raw"]:
                lines.append(f"- {_md_text(str(r))} (unresolved)")
        else:
            lines.append("- (none)")
        if include_bodies and pid in docs:
            lines += ["", "---", "", _escape_body(docs[pid].body.strip("\n"))]
        (out / "pages" / f"{pid}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- graph ----------------------------------------------------------------------------------
    graph = {
        "nodes": [{"id": pid, "title": p["title"], "category": p["category"]}
                  for pid, p in sorted(state.pages.items())],
        "edges": [{"source": pid, "target": r, "rank": rank}
                  for pid, p in sorted(state.pages.items()) for rank, r in enumerate(p.get("related", []), 1)
                  if r in state.pages],
        "categories": [{"id": cid, "label": c["label"], "parent": c.get("parent"), "file": f"categories/{cat_file[cid]}"}
                       for cid, c in sorted(state.categories.items())],
    }
    (out / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
                                    encoding="utf-8")
    return out


_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def check_links(wiki_dir: str | Path) -> list[tuple[str, str]]:
    """Every relative Markdown link target under ``wiki_dir`` must exist.  Returns the broken ones."""
    root = Path(wiki_dir)
    broken = []
    for md in root.rglob("*.md"):
        for m in _LINK_RE.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1).split("#", 1)[0]
            if not target or "://" in target:
                continue
            if not (md.parent / target).resolve().exists():
                broken.append((str(md.relative_to(root)), target))
    return broken
