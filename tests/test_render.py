"""Markdown wiki rendering: every relative link resolves; structure of index/category/page files."""
import json
from pathlib import Path

from wikiorg.data.docs import Doc
from wikiorg.render import WikiState, check_links, read_wiki_state, slugify, write_wiki, write_wiki_state


def _state(two_level=False):
    docs = [Doc(id=f"p{i}", title=f"Page [{i}] & co", body=f"Body of page {i}.\n\n## Section\n\nMore.") for i in range(6)]
    state = WikiState(arm="A")
    if two_level:
        state.categories["p00"] = {"label": "top a", "terms": ["a"], "parent": None}
        state.categories["p01"] = {"label": "top b", "terms": ["b"], "parent": None}
    state.categories["c00"] = {"label": "rivers, sea, delta", "terms": ["rivers", "sea", "delta"],
                               "parent": "p00" if two_level else None}
    state.categories["c01"] = {"label": "club: football/league", "terms": ["club", "football"],
                               "parent": "p01" if two_level else None}
    for i, d in enumerate(docs):
        others = [f"p{j}" for j in range(6) if j != i][:3]
        state.pages[d.id] = {"title": d.title, "category": "c00" if i < 4 else "c01", "related": others,
                             "related_raw": [f"Page [{j}] & co" for j in range(6) if j != i][:3]}
    return docs, state


def test_wiki_links_resolve_and_files_exist(tmp_path):
    docs, state = _state()
    assert state.validate() == []
    out = write_wiki(state, docs, tmp_path / "wiki", title="Test wiki")
    assert check_links(out) == []
    assert (out / "index.md").exists() and (out / "graph.json").exists()
    cats = sorted(p.name for p in (out / "categories").glob("*.md"))
    assert cats == ["c00-rivers-sea-delta.md", "c01-club-football-league.md"]
    assert sorted(p.stem for p in (out / "pages").glob("*.md")) == [f"p{i}" for i in range(6)]
    index = (out / "index.md").read_text()
    assert "6 pages in 2 categories" in index and "(pages/p0.md)" in index
    page = (out / "pages" / "p0.md").read_text()
    assert "## Related" in page and "(../pages/p1.md)" in page and "Body of page 0." in page
    graph = json.loads((out / "graph.json").read_text())
    assert len(graph["nodes"]) == 6 and len(graph["edges"]) == 18


def test_two_level_hierarchy_rendered(tmp_path):
    docs, state = _state(two_level=True)
    assert state.validate() == []
    out = write_wiki(state, docs, tmp_path / "wiki")
    assert check_links(out) == []
    index = (out / "index.md").read_text()
    assert "- [top a](categories/p00-top-a.md)" in index and "  - [rivers, sea, delta]" in index
    cat = (out / "categories" / "c00-rivers-sea-delta.md").read_text()
    assert "**Parent:** [top a](p00-top-a.md)" in cat


def test_state_round_trip_and_validation(tmp_path):
    docs, state = _state()
    p = write_wiki_state(state, tmp_path / "wiki_state.json")
    back = read_wiki_state(p)
    assert back.to_dict() == state.to_dict()
    assert set(json.loads(p.read_text())) == {"arm", "pages", "categories", "extra_pages"}
    state.pages["p0"]["related"].append("nope")
    state.pages["p1"]["category"] = "c99"
    problems = state.validate()
    assert any("nope" in x for x in problems) and any("c99" in x for x in problems)


def test_broken_link_detection(tmp_path):
    (tmp_path / "a.md").write_text("[x](missing.md) [ok](b.md) [web](https://example.org)")
    (tmp_path / "b.md").write_text("fine")
    assert check_links(tmp_path) == [("a.md", "missing.md")]


def test_slugify():
    assert slugify("Rivers of Europe; the Danube!") == "rivers-of-europe-the-danube"
    assert slugify("   ") == "untitled"
    assert len(slugify("x" * 100)) <= 40


def test_body_with_accidental_link_syntax_is_escaped(tmp_path):
    # Real-world case: WikiCS "Clipper architecture" prose has ISA notation "[Rx](Rn), [Rx](PC)"
    # that is syntactically a Markdown link to nonexistent files "Rn"/"PC".
    docs = [Doc(id="p0", title="ISA doc", body="Addressing modes: [Rx](Rn), [Rx](PC) are indexed.")]
    state = WikiState(arm="A")
    state.categories["c00"] = {"label": "cpu", "terms": ["cpu"], "parent": None}
    state.pages["p0"] = {"title": "ISA doc", "category": "c00", "related": [], "related_raw": []}
    out = write_wiki(state, docs, tmp_path / "wiki")
    assert check_links(out) == []
    page = (out / "pages" / "p0.md").read_text()
    assert "[Rx]\\(Rn)" in page and "[Rx]\\(PC)" in page
    assert "[Rx](Rn)" not in page  # would parse as a link to a nonexistent file "Rn"
