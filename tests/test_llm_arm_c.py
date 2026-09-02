"""Arm C: prompt construction, ``related_titles`` validation and topic identity.

The validation rules are pre-registered (plan W2.3): exact normalised-title
matching only, self/duplicate/non-corpus entries dropped and counted, topic
identity by normalised match trusted over the model's ``is_new_topic`` flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wikiorg.adapters.common import TitleIndex
from wikiorg.data.docs import Doc, write_doc, write_order
from wikiorg.llm import arm_c as C
from wikiorg.llm import runner as R

pytestmark = pytest.mark.timeout(120)

TITLES = {"d1": "Danube", "d2": "Marie Curie", "d3": "Association Football",
          "d4": "Rhône (river)", "d5": "Café Society"}


# ================================================================================================
# prompts
# ================================================================================================
def test_system_prompt_is_stable_and_numbers_the_corpus():
    titles = list(TITLES.values())
    a = C.system_prompt(titles)
    b = C.system_prompt(titles)
    assert a == b, "the cached prefix must be byte-identical on every call"
    assert "1. Danube" in a and "5. Café Society" in a
    assert f"Corpus ({len(titles)} documents)" in a
    assert "related_titles" in a and "at most 5" in a


def test_processed_mode_adds_a_restriction_to_the_system_prompt():
    full = C.system_prompt(list(TITLES.values()))
    proc = C.system_prompt(list(TITLES.values()), candidates="processed")
    assert proc != full and "already processed" in proc


def test_user_prompt_carries_the_topics_so_far_and_the_document():
    doc = Doc(id="d1", title="Danube", body="The Danube is a river.")
    first = C.user_prompt(doc, [])
    assert "none" in first and "Danube is a river" in first
    later = C.user_prompt(doc, [("rivers", "Rivers"), ("people", "People")])
    assert "- rivers — Rivers" in later and "- people — People" in later
    restricted = C.user_prompt(doc, [], processed=[])
    assert "must be empty" in restricted
    restricted = C.user_prompt(doc, [], processed=["Marie Curie"])
    assert "- Marie Curie" in restricted


def test_processed_ids_only_exposes_earlier_documents():
    ids = ["d1", "d2", "d3"]
    assert C.processed_ids(ids, 1) == []
    assert C.processed_ids(ids, 3) == ["d1", "d2"]


# ================================================================================================
# related_titles validation
# ================================================================================================
@pytest.fixture()
def index() -> TitleIndex:
    return TitleIndex(TITLES)


def test_exact_match_resolves_and_normalisation_is_nfkc_casefold(index):
    got = C.validate_related(["Danube", "MARIE CURIE", "association football"], "d5", index)
    assert got["related"] == ["d1", "d2", "d3"]
    assert got["n_invalid"] == 0 and got["n_related"] == 3


def test_self_reference_duplicates_and_non_corpus_are_dropped_and_counted(index):
    got = C.validate_related(["Danube", "Danube", "Café Society", "Atlantis", ""], "d5", index)
    assert got["related"] == ["d1"]
    assert got["n_self"] == 1 and got["n_duplicate"] == 1 and got["n_unresolved"] == 2
    assert got["n_invalid"] == 4 and got["n_related_raw"] == 5
    assert got["related_raw"] == ["Danube", "Danube", "Café Society", "Atlantis", ""]


def test_near_misses_are_counted_but_never_become_links(index):
    got = C.validate_related(["Danub", "Marie Curi"], "d1", index)
    assert got["related"] == []
    assert got["n_near_miss"] == 2 and got["n_unresolved"] == 2


def test_more_than_five_related_are_truncated(index):
    big = TitleIndex({f"x{i}": f"Title {i}" for i in range(8)})
    got = C.validate_related([f"Title {i}" for i in range(8)], "z", big)
    assert len(got["related"]) == 5 and got["n_over_limit"] == 3


def test_processed_restriction_drops_unseen_documents(index):
    got = C.validate_related(["Danube", "Marie Curie"], "d5", index, allowed={"d1"})
    assert got["related"] == ["d1"]
    assert got["n_not_processed"] == 1 and got["n_invalid"] == 1


# ================================================================================================
# topic identity
# ================================================================================================
def test_topic_identity_uses_the_normalised_match_not_the_model_flag():
    book = C.TopicBook()
    a = book.resolve("rivers-of-europe", "Rivers of Europe", True)
    assert a["is_new_topic"] is True and a["is_new_disagreement"] is False
    # the model says "new" again for a topic it has already used: the match wins
    b = book.resolve("Rivers-Of-Europe", "Rivers of Europe", True)
    assert b["is_new_topic"] is False and b["topic"] == "rivers-of-europe"
    assert b["is_new_disagreement"] is True and book.n_disagreements == 1
    # and the reverse: it says "not new" for a topic never seen
    c = book.resolve("politicians", "Politicians", False)
    assert c["is_new_topic"] is True and book.n_disagreements == 2
    assert sorted(s for s, _ in book.as_pairs()) == ["politicians", "rivers-of-europe"]


def test_topic_book_keeps_the_first_label_for_a_slug():
    book = C.TopicBook()
    book.resolve("video-games", "Video games", True)
    book.resolve("video-games", "Computer games", False)
    assert dict(book.as_pairs())["video-games"] == "Video games"


# ================================================================================================
# structured output parsing
# ================================================================================================
def _res(structured=None, text="") -> R.CallResult:
    inv = R.Invocation(cmd=[], returncode=0, stdout="", stderr="", wall_ms=1)
    return R.CallResult(invocation=inv, result=None, status="ok", structured_output=structured, text=text)


PAYLOAD = {"topic": "rivers", "is_new_topic": True, "label": "Rivers",
           "one_line_summary": "A river.", "related_titles": ["Danube"]}


def test_parse_structured_prefers_the_validated_field():
    assert C.parse_structured(_res(structured=PAYLOAD))["topic"] == "rivers"
    assert C.parse_structured(_res(structured=json.dumps(PAYLOAD)))["topic"] == "rivers"


def test_parse_structured_falls_back_to_json_in_the_reply():
    res = _res(text="Here you go:\n" + json.dumps(PAYLOAD) + "\nthanks")
    assert C.parse_structured(res)["label"] == "Rivers"
    assert C.parse_structured(_res(text="no json here")) is None
    assert C.parse_structured(_res()) is None


# ================================================================================================
# the loop, with a fake CLI
# ================================================================================================
@pytest.fixture()
def corpus(tmp_path) -> Path:
    d = tmp_path / "corpus"
    for pid, title in TITLES.items():
        write_doc(d, Doc(id=pid, title=title, body=f"Body about {title}. " * 10))
    write_order(d / "orders" / "order-1.txt", list(TITLES))
    return d


class FakeC:
    def __init__(self, payloads):
        self.payloads = payloads
        self.systems: list[str] = []
        self.n = 0

    def __call__(self, cmd, stdin=None, timeout=300, cwd=None, env=None):
        self.systems.append(cmd[cmd.index("--system-prompt") + 1])
        payload = self.payloads[self.n]
        self.n += 1
        result = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
                  "duration_ms": 3000, "duration_api_ms": 2900, "total_cost_usd": 0.004,
                  "result": json.dumps(payload), "structured_output": payload,
                  "permission_denials": [],
                  "modelUsage": {"claude-sonnet-5": {"inputTokens": 900, "outputTokens": 120,
                                                     "cacheReadInputTokens": 2400 if self.n > 1 else 0,
                                                     "cacheCreationInputTokens": 2400 if self.n == 1 else 0,
                                                     "costUSD": 0.004}}}
        return R.Invocation(cmd=list(cmd), returncode=0, stdout=json.dumps(result), stderr="", wall_ms=3000)


def test_three_labels_produce_adapter_ready_rows(tmp_path, corpus, monkeypatch):
    payloads = [
        {"topic": "rivers", "is_new_topic": True, "label": "Rivers", "one_line_summary": "s",
         "related_titles": []},
        {"topic": "people", "is_new_topic": True, "label": "People", "one_line_summary": "s",
         "related_titles": ["Danube", "Marie Curie", "Atlantis"]},
        {"topic": "Rivers", "is_new_topic": True, "label": "Rivers", "one_line_summary": "s",
         "related_titles": ["Danube"]},
    ]
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    fake = FakeC(payloads)
    monkeypatch.setattr(R, "run_claude", fake)
    run = C.ArmCRun("c1", "test", n=3, corpus_dir=corpus, runs_root=tmp_path / "runs",
                    out_root=tmp_path / "out", log=lambda _m: None)
    summary = run.run()

    assert len(set(fake.systems)) == 1, "the cached prefix must not change between calls"
    rows = run.calls.rows()
    assert [r["status"] for r in rows] == ["ok", "ok", "ok"]
    assert rows[1]["validation"]["n_unresolved"] == 1        # "Atlantis"
    assert rows[1]["validation"]["n_self"] == 1              # "Marie Curie" is d2 itself
    assert rows[1]["validation"]["related"] == ["d1"]
    # "Rivers" matches "rivers" after casefolding, so the third document reuses the topic
    assert rows[2]["structured_output"]["topic"] == "rivers"
    assert rows[2]["structured_output"]["is_new_topic"] is False
    assert summary["n_topics"] == 2 and summary["n_is_new_disagreements"] == 1
    assert rows[1]["tokens"]["cache_read"] > 0

    from wikiorg.adapters.arm_c import from_jsonl
    state = from_jsonl(run.calls.path, TITLES, run_id="c1")
    assert state.pages["d1"].category == state.pages["d3"].category
    assert state.pages["d2"].related == ["d1"]
    assert state.pages["d4"].extra["status"] == "failed"     # never labelled
    assert len(state.categories) == 2


def test_arm_c_resumes_and_rebuilds_the_topic_list(tmp_path, corpus, monkeypatch):
    payloads = [{"topic": "rivers", "is_new_topic": True, "label": "Rivers",
                 "one_line_summary": "s", "related_titles": []}] * 5
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    monkeypatch.setattr(R, "run_claude", FakeC(payloads))
    kw = dict(corpus_dir=corpus, runs_root=tmp_path / "runs", out_root=tmp_path / "out",
              log=lambda _m: None)
    C.ArmCRun("c2", "test", n=4, stop_after=2, **kw).run()
    monkeypatch.setattr(R, "run_claude", FakeC(payloads))
    run2 = C.ArmCRun("c2", "test", n=4, **kw)
    run2.run()
    ok = [r for r in run2.calls.rows() if r["status"] == "ok"]
    assert [r["doc_id"] for r in ok] == ["d1", "d2", "d3", "d4"]
    # the resumed process must know the topic already exists
    assert ok[2]["structured_output"]["is_new_topic"] is False


def test_system_prompt_size_is_guarded(tmp_path, corpus, monkeypatch):
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    monkeypatch.setattr(C, "MAX_SYSTEM_BYTES", 100)
    with pytest.raises(R.HarnessError, match="argv"):
        C.ArmCRun("c3", "test", n=3, corpus_dir=corpus, runs_root=tmp_path / "runs")


def test_command_is_toolless_and_schema_bound(tmp_path, corpus, monkeypatch):
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    run = C.ArmCRun("c4", "test", n=3, corpus_dir=corpus, runs_root=tmp_path / "runs")
    cmd = run.cmd()
    assert cmd[:2] == ["claude", "-p"], "the prompt arrives on stdin"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert json.loads(cmd[cmd.index("--json-schema") + 1])["required"] == [
        "topic", "is_new_topic", "label", "one_line_summary", "related_titles"]
    assert cmd[cmd.index("--max-budget-usd") + 1] == "0.1"
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert cmd[cmd.index("--effort") + 1] == "high"


def test_disagreements_are_counted_from_the_rows_not_the_resumed_book(tmp_path, corpus, monkeypatch):
    """On resume the topic list is replayed with is_new=True; the live counter would
    then report a disagreement for every earlier document."""
    payloads = [{"topic": "rivers", "is_new_topic": True, "label": "Rivers",
                 "one_line_summary": "s", "related_titles": []}] * 5
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    kw = dict(corpus_dir=corpus, runs_root=tmp_path / "runs", out_root=tmp_path / "out",
              log=lambda _m: None)
    monkeypatch.setattr(R, "run_claude", FakeC(payloads))
    C.ArmCRun("c5", "test", n=4, stop_after=2, **kw).run()
    monkeypatch.setattr(R, "run_claude", FakeC(payloads))
    summary = C.ArmCRun("c5", "test", n=4, **kw).run()
    # documents 2, 3 and 4 all claim a new topic that already exists
    assert summary["n_is_new_disagreements"] == 3
    assert summary["n_topics"] == 1
