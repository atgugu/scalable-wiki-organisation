"""``extract.py``: a wiki snapshot -> the CSV the evaluation adapter consumes.

The load-bearing assertion is the round trip: the ``WikiState`` built from the
CSV by ``wikiorg.adapters.arm_b.from_extract_csv`` must equal the one built by
reading the wiki tree directly (``from_wiki_dir``).  If the column contract ever
drifts, this fails rather than silently producing an arm-B result that scores
different documents.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wikiorg.adapters.arm_b import from_extract_csv, from_wiki_dir
from wikiorg.data.docs import Doc, write_doc, write_order
from wikiorg.llm import arm_b as B
from wikiorg.llm import extract as E

pytestmark = pytest.mark.timeout(120)

IDS = ["d1", "d2", "d3", "d4"]
TITLES = {"d1": "Danube", "d2": "Rhine", "d3": "Marie Curie", "d4": "Never Ingested"}


def source_page(doc_id: str, topic: str, related: list[str]) -> str:
    return (f"---\ntitle: \"{TITLES[doc_id]}\"\ntype: source\nsource: raw/{doc_id}.md\n"
            f"topic: {topic}\nrelated: [{', '.join(related)}]\nentities: [europe]\n"
            f"concepts: [hydrology]\ningest: 1\n---\n"
            f"# {TITLES[doc_id]}\nA summary of {TITLES[doc_id]} in prose.\n\n"
            "## Related\n" + "".join(f"- [{TITLES[r]}]({r}.md) — why\n" for r in related if r in TITLES))


@pytest.fixture()
def wiki(tmp_path) -> Path:
    root = tmp_path / "run" / "wiki"
    (root / "sources").mkdir(parents=True)
    (root / "topics").mkdir()
    (root / "entities").mkdir()
    (root / "sources" / "d1.md").write_text(source_page("d1", "rivers", ["d2"]), encoding="utf-8")
    (root / "sources" / "d2.md").write_text(source_page("d2", "rivers", ["d1", "d1", "d2", "d9"]), encoding="utf-8")
    (root / "sources" / "d3.md").write_text(source_page("d3", "scientists", []), encoding="utf-8")
    (root / "topics" / "rivers.md").write_text(
        "---\ntitle: \"Rivers of Europe\"\ntype: topic\n---\n# Rivers of Europe\nScope.\n\n"
        "## Sources\n- [Danube](../sources/d1.md) — gloss\n- [Rhine](../sources/d2.md) — gloss\n",
        encoding="utf-8")
    (root / "topics" / "scientists.md").write_text(
        "---\ntitle: \"Scientists\"\ntype: topic\n---\n# Scientists\nScope.\n\n"
        "## Sources\n- [Marie Curie](../sources/d3.md) — gloss\n", encoding="utf-8")
    (root / "entities" / "europe.md").write_text(
        "---\ntitle: \"Europe\"\ntype: entity\n---\n# Europe\nA continent.\n\n"
        "## Mentioned in\n- [Danube](../sources/d1.md)\n", encoding="utf-8")
    (root / "index.md").write_text(
        "# Index\n\n## Topics\n- [Rivers of Europe](topics/rivers.md) — 2 sources\n"
        "- [Scientists](topics/scientists.md) — 1 sources\n\n## Sources\n"
        "- [Danube](sources/d1.md) — [Rivers of Europe](topics/rivers.md)\n"
        "- [Rhine](sources/d2.md) — [Rivers of Europe](topics/rivers.md)\n"
        "- [Marie Curie](sources/d3.md) — [Scientists](topics/scientists.md)\n\n"
        "## Entities\n- [Europe](entities/europe.md) — a continent\n", encoding="utf-8")
    (root / "log.md").write_text("## [2026-09-02] ingest | Danube\ncreated: ...\n", encoding="utf-8")
    return root.parent


# ================================================================================================
# rows
# ================================================================================================
def test_doc_rows_cover_every_corpus_id_and_drop_invalid_links(wiki):
    rows = {r["doc_id"]: r for r in E.doc_rows(wiki, IDS, TITLES)}
    assert set(rows) == set(IDS)
    assert rows["d1"]["category"] == "rivers" and rows["d1"]["label"] == "Rivers of Europe"
    assert json.loads(rows["d1"]["related"]) == ["d2"]
    # self references, repeats and links to non-existent pages never become links,
    # but related_raw keeps them so the link-validity table can classify them
    assert json.loads(rows["d2"]["related"]) == ["d1"]
    assert json.loads(rows["d2"]["related_raw"]) == ["d1", "d1", "d2", "d9"]
    assert rows["d2"]["n_related_dropped"] == 3
    assert json.loads(rows["d1"]["entities"]) == ["europe"]
    assert rows["d1"]["topic_page_exists"] is True
    # a document that was never ingested is present with status=missing (a singleton)
    assert rows["d4"]["status"] == "missing" and rows["d4"]["category"] == ""
    assert json.loads(rows["d4"]["related_valid"]) == []


def test_related_valid_is_always_written(tmp_path, wiki):
    """An empty ``related_valid`` cell would make the adapter treat every id as valid."""
    path = E.write_csv(tmp_path / "x.csv", E.doc_rows(wiki, IDS, TITLES))
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assert row["related_valid"] != "", row


def test_csv_round_trips_to_the_same_wiki_state_as_the_tree(tmp_path, wiki):
    csv_path = E.write_csv(tmp_path / "x.csv", E.doc_rows(wiki, IDS, TITLES))
    from_tree = from_wiki_dir(wiki, corpus_ids=IDS, corpus_titles=TITLES)
    from_csv = from_extract_csv(csv_path, corpus_ids=IDS, corpus_titles=TITLES)
    assert set(from_csv.pages) == set(from_tree.pages)
    for pid in IDS:
        a, b = from_tree.pages[pid], from_csv.pages[pid]
        assert a.category == b.category, pid
        assert a.related == b.related, pid
        assert a.title == b.title, pid
    assert {c: v.label for c, v in from_csv.categories.items()} == {
        c: v.label for c, v in from_tree.categories.items() if c in from_csv.categories}
    assert from_tree.partition() and from_csv.partition()


def test_audit_reports_links_index_coverage_and_page_counts(wiki):
    rep = E.audit(wiki)
    assert rep["n_pages"] == 8 and rep["has_index"] is True
    assert rep["n_broken_links"] == 0
    assert rep["index_coverage"] == pytest.approx(1.0)
    assert rep["pages_by_type"]["source"] == 3 and rep["pages_by_type"]["topic"] == 2
    assert rep["page_counts"]["total"] == 8


def test_audit_flags_a_broken_link(wiki):
    (wiki / "wiki" / "sources" / "d3.md").write_text(
        source_page("d3", "scientists", []) + "\n- [Ghost](../topics/ghost.md)\n", encoding="utf-8")
    rep = E.audit(wiki)
    assert rep["n_broken_links"] == 1
    assert rep["broken_links"][0]["from"] == "sources/d3.md"


# ================================================================================================
# sources: directory, tar, git ref
# ================================================================================================
def test_extract_from_a_directory_writes_csv_and_audit(tmp_path, wiki):
    corpus = tmp_path / "corpus"
    for pid, title in TITLES.items():
        write_doc(corpus, Doc(id=pid, title=title, body=f"Body about {title}. " * 10))
    write_order(corpus / "orders" / "order-1.txt", IDS)
    out = tmp_path / "extract"
    rep = E.extract(wiki, out, "runX", n=4, corpus_dir=corpus,
                    order_file=corpus / "orders" / "order-1.txt", log=lambda _m: None)
    assert (out / "runX_N4.csv").exists() and (out / "runX_N4.audit.json").exists()
    assert rep["n_ok"] == 3 and rep["n_missing"] == 1 and rep["n_categories"] == 2
    state = from_extract_csv(out / "runX_N4.csv", corpus_ids=IDS, corpus_titles=TITLES)
    assert state.pages["d4"].category is None


def test_extract_from_a_git_ref_and_from_a_tar(tmp_path, wiki):
    work, gitdir = wiki, tmp_path / "run.git"
    store = B.GitStore(work, gitdir)
    store.init()
    store.add_all()
    store.commit("ingest 0003 d3")
    store.tag("ingest-0003")
    out = tmp_path / "extract"
    rep = E.extract(gitdir, out, "runG", n=4, ref="ingest-0003", gitdir=gitdir, log=lambda _m: None)
    assert rep["n_ok"] == 3 and rep["ref"] == "ingest-0003"

    tar = store.archive("HEAD", tmp_path / "snap.tar.gz")
    rep2 = E.extract(tar, tmp_path / "extract2", "runT", n=4, log=lambda _m: None)
    assert rep2["n_ok"] == rep["n_ok"] and rep2["n_categories"] == rep["n_categories"]


def test_extract_accepts_head_when_no_tag_exists_yet(tmp_path, wiki):
    store = B.GitStore(wiki, tmp_path / "run.git")
    store.init()
    store.add_all()
    store.commit("ingest 0003 d3")
    rep = E.extract(tmp_path / "run.git", tmp_path / "out", "runH", ref="HEAD",
                    gitdir=tmp_path / "run.git", log=lambda _m: None)
    assert rep["n_ok"] == 3


def test_extract_copies_the_call_ledger_when_present(tmp_path, wiki):
    calls = tmp_path / "calls.jsonl"
    calls.write_text(json.dumps({"kind": "ingest", "status": "ok", "cost_usd": 0.1,
                                 "tokens": {"input": 1, "output": 2, "cache_read": 3,
                                            "cache_creation": 4}}) + "\n", encoding="utf-8")
    rep = E.extract(wiki, tmp_path / "out", "runC", n=4, calls_path=calls, log=lambda _m: None)
    assert rep["calls"]["n_calls"] == 1 and rep["calls"]["cost_usd"] == pytest.approx(0.1)
    assert (tmp_path / "out" / "runC_N4.calls.jsonl").exists()


def test_extract_also_writes_the_run_directory_the_evaluation_reads(tmp_path, wiki):
    """``wikiorg.eval.report.load_state`` prefers ``wiki_state.json``; without one
    W3 would have to plumb the CSV by hand."""
    calls = tmp_path / "calls.jsonl"
    calls.write_text(json.dumps({"kind": "ingest", "status": "ok", "cost_usd": 0.2,
                                 "tokens": {"input": 1, "output": 2, "cache_read": 3,
                                            "cache_creation": 4}}) + "\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "B" / "test" / "4" / "order-1"
    rep = E.extract(wiki, tmp_path / "out", "runR", n=4, calls_path=calls, run_dir=run_dir,
                    extra_manifest={"corpus": "test"}, log=lambda _m: None)
    assert rep["run_dir"] == str(run_dir)
    from wikiorg.eval.state import WikiState
    state = WikiState.load(run_dir / "wiki_state.json")
    assert state.arm == "B" and len(state.categories) == 2
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "runR" and manifest["arm"] == "B" and manifest["corpus"] == "test"
    assert manifest["claude_md_sha256"] and manifest["model"] == "claude-sonnet-5"
    assert manifest["totals"]["cost_usd"] == pytest.approx(0.2)
    assert (run_dir / "llm_calls.jsonl").exists()
