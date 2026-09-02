"""Adapters: title normalisation and the B / C fixtures under tests/fixtures/eval."""
import csv
from pathlib import Path

import pytest

from wikiorg.adapters.arm_a import build_state, category_id, from_run_dir
from wikiorg.adapters.arm_b import from_extract_csv, from_wiki_dir, link_audit, read_wiki_tree, target_id
from wikiorg.adapters.arm_c import from_jsonl, from_rows, structured_output
from wikiorg.adapters.common import TitleIndex, as_str_list, norm_title, slugify, split_frontmatter
from wikiorg.eval.categories import evaluate_categories
from wikiorg.eval.links import link_validity
from wikiorg.eval.state import WikiState, coerce_state, read_gold

FX = Path(__file__).parent / "fixtures" / "eval"


@pytest.fixture(scope="module")
def gold():
    return read_gold(FX / "gold.jsonl")


def test_norm_title_and_index():
    assert norm_title("  Émile_Zola’s (Novel)!  ") == "émile zola s novel"
    assert norm_title("Angela Merkel") == norm_title("angela  MERKEL") == norm_title("ANGELA MERKEL.")
    assert norm_title("ﬁne") == norm_title("fine")  # NFKC ligature
    assert slugify("Rivers of Europe!") == "rivers-of-europe" and slugify("") == "untitled"
    idx = TitleIndex({"d1": "Danube", "d2": "Rhine", "d3": "Angela Merkel"})
    assert idx.resolve("danube") == "d1" and idx.resolve("d2") == "d2" and idx.resolve("Nile") is None
    assert idx.near_miss("Danub")[0] == "d1" and idx.near_miss("Nile")[0] is None
    assert idx.classify("angela merkel") == ("resolved", "d3")
    assert idx.classify("Angela Merkl") == ("near_miss", "d3")
    assert idx.classify("Barack Obama") == ("dangling", None)


def test_frontmatter_parsers():
    fields, body = split_frontmatter('---\ntitle: "X"\nrelated: [a, b]\ningest: 3\n---\n# X\nbody')
    assert fields == {"title": "X", "related": ["a", "b"], "ingest": 3} and body.startswith("# X")
    fields, _ = split_frontmatter('---\ntitle: Bad: colon "x"\nrelated: [a, b]\ntopic: t  # comment\n---\nbody')
    assert fields["related"] == ["a", "b"] and fields["topic"] == "t"
    assert split_frontmatter("no front matter") == ({}, "no front matter")
    assert as_str_list("[d1, d2]") == ["d1", "d2"] and as_str_list("d1, d2") == ["d1", "d2"] and as_str_list(None) == []
    assert target_id("[Danube](../sources/d1.md)") == "d1" and target_id("sources/d1.md#x") == "d1" and target_id("d1") == "d1"


def test_arm_b_fixture(gold):
    st = from_wiki_dir(FX / "arm_b_wiki", corpus_ids=gold.ids(), corpus_titles=gold.titles)
    assert st.arm == "B" and set(st.pages) == {"d1", "d2", "d3", "d4", "d5"}
    assert st.pages["d1"].category == "rivers-of-europe" and st.pages["d3"].category == "politicians"
    assert st.pages["d5"].category is None and st.pages["d5"].extra["status"] == "missing"
    assert st.pages["d2"].related == ["d1"] and st.pages["d2"].related_raw == ["d1", "d5"]  # d5 has no page
    assert st.pages["d3"].related_raw == ["d4", "d9"] and st.pages["d3"].related == ["d4"]
    assert st.pages["d4"].related == ["d3"]  # self and duplicate dropped
    assert st.categories["rivers-of-europe"].label == "Rivers of Europe"
    assert st.categories["video-games"].label == "Video games"
    assert st.k_produced() == 2 and st.n_uncategorised() == 1
    types = sorted(e["type"] for e in st.extra_pages)
    assert types == ["concept", "entity", "index", "log", "overview", "topic", "topic", "topic"]
    audit = st.meta["audit"]
    assert audit["n_broken_links"] == 1 and audit["broken_links"][0]["target"] == "../entities/nonexistent.md"
    assert audit["not_in_index"] == ["sources/d4.md"] and audit["index_coverage"] == pytest.approx(8 / 9)
    assert audit["topics_without_sources"] == ["topics/video-games.md"] and audit["sources_missing_topic_page"] == []
    assert audit["pages_by_type"]["source"] == 4 and audit["total_words"] > 0
    v = link_validity(st)
    assert (v["resolved"], v["dangling"], v["self"], v["duplicate"]) == (4, 2, 1, 1)  # d5 and d9 dangle
    res = evaluate_categories(st, gold, n_random=0)
    assert res["levels"]["l2"]["purity"] == 1.0 and res["levels"]["l2"]["n_singletons"] == 1
    # round trip through the on-disk contract
    d = st.to_dict()
    assert coerce_state(d).to_dict() == d
    assert read_wiki_tree(FX / "arm_b_wiki" / "wiki")["index.md"].type == "index"
    assert link_audit(read_wiki_tree(FX / "arm_b_wiki"))["n_pages"] == 12


def test_arm_c_fixture(gold):
    st = from_jsonl(FX / "arm_c_calls.jsonl", gold.titles, run_id="c-test")
    assert st.arm == "C" and set(st.categories) == {"rivers-of-europe", "politicians"}  # casefold merge
    assert st.pages["d1"].category == "rivers-of-europe" and st.pages["d1"].related == ["d2", "d5"]  # last ok row wins
    assert st.pages["d2"].related == ["d1"] and st.pages["d2"].related_raw == ["Danube", "Danub", "Nile"]
    assert st.pages["d3"].related == ["d4"]  # self reference dropped
    assert st.pages["d4"].related == ["d3", "d1"]  # duplicate dropped
    assert st.pages["d5"].category is None and st.pages["d5"].extra["status"] == "failed"
    assert st.categories["rivers-of-europe"].label == "Rivers of Europe"  # most frequent label wins ties by first seen
    m = st.meta
    assert (m["n_rows"], m["n_failed_rows"], m["n_ok_docs"], m["n_topics"]) == (6, 2, 4, 2)
    assert (m["n_related_raw"], m["n_related_resolved"], m["n_related_self"], m["n_related_duplicate"], m["n_related_unresolved"]) == (10, 6, 1, 1, 2)
    v = link_validity(st)
    assert (v["resolved"], v["near_miss"], v["dangling"], v["self"], v["duplicate"]) == (6, 1, 1, 1, 1)
    assert structured_output({"structured_output": '{"topic": "T", "related_titles": []}'}) == {"topic": "T", "related_titles": []}
    assert structured_output({"result": "nope"}) is None
    two = from_rows([{"doc_id": "d1", "structured_output": {"topic": "A", "label": "L1", "related_titles": []}},
                     {"doc_id": "d2", "structured_output": {"topic": "a", "label": "L2", "related_titles": []}},
                     {"doc_id": "d3", "structured_output": {"topic": "a", "label": "L2", "related_titles": []}}], {"d1": "x", "d2": "y", "d3": "z"})
    assert list(two.categories) == ["a"] and two.categories["a"].label == "L2"


def test_arm_a_build_state(tmp_path, gold):
    st = build_state(gold.ids(), gold.titles, [0, 0, 1, 1, 0], {"d1": ["d2", "d5", "d1"], "d2": ["d1"], "d3": ["d4"], "d4": ["d3"], "d5": ["d1"]},
                     terms=[["river", "europe", "sea", "flow"], ["politician", "president", "chancellor"]], category_labels={1: "Heads of state"}, arm="A+")
    assert st.categories["c000"].label == "river, europe, sea" and st.categories["c001"].label == "Heads of state"
    assert st.pages["d1"].related == ["d2", "d5"] and st.pages["d1"].related_raw == ["Rhine", "Volga"]
    assert category_id(3) == "c003" and category_id("x") == "x"
    two_level = build_state(["a", "b"], {"a": "A", "b": "B"}, {"a": 0, "b": 1}, {"a": [], "b": []}, parents={0: "g0", 1: "g0"})
    assert two_level.categories["c000"].parent == "g0" and "g0" in two_level.categories
    path = st.save(tmp_path / "wiki_state.json")
    assert from_run_dir(tmp_path).to_dict() == st.to_dict() and from_run_dir(path).arm == "A+"
    with pytest.raises(FileNotFoundError):
        from_run_dir(tmp_path / "missing")


def test_from_extract_csv(tmp_path, gold):
    p = tmp_path / "extract.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "category", "label", "related", "related_valid", "status"])
        w.writeheader()
        w.writerow({"doc_id": "d1", "category": "rivers-of-europe", "label": "Rivers of Europe", "related": "d2;d5", "related_valid": "d2", "status": "ok"})
        w.writerow({"doc_id": "d2", "category": "rivers-of-europe", "label": "Rivers of Europe", "related": '["d1"]', "related_valid": '["d1"]', "status": "ok"})
        w.writerow({"doc_id": "d3", "category": "politicians", "label": "Politicians", "related": "", "related_valid": "", "status": "ok"})
        w.writerow({"doc_id": "d5", "category": "", "label": "", "related": "", "related_valid": "", "status": "failed"})
    st = from_extract_csv(p, corpus_ids=gold.ids(), corpus_titles=gold.titles)
    assert st.pages["d1"].related == ["d2"] and st.pages["d1"].related_raw == ["d2", "d5"]
    assert st.pages["d4"].category is None and st.pages["d5"].category is None
    assert st.categories["politicians"].label == "Politicians" and st.k_produced() == 2
