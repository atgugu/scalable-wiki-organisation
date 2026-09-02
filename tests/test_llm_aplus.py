"""Arm A+: the one-call-per-cluster namer (schema, <= 6 words, retry, fallback)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wikiorg.llm import arm_aplus as A
from wikiorg.llm import runner as R

pytestmark = pytest.mark.timeout(120)


def test_check_label_enforces_the_six_word_rule():
    assert A.check_label("Rivers of Europe")[0] is True
    assert A.check_label("Association football clubs of the world")[0] is True     # 6 words
    assert A.check_label("Association football clubs of the entire world")[0] is False
    assert A.check_label("")[0] is False
    assert A.check_label("Cluster of football pages")[0] is False                   # meta word
    assert A.check_label("Sports category")[0] is False
    assert A.check_label('"Video games"')[0] is True                                # quotes stripped


def test_cluster_prompt_caps_terms_and_titles():
    p = A.cluster_prompt([f"t{i}" for i in range(20)], [f"Title {i}" for i in range(20)])
    assert p.count(",") == 9 and p.count("- Title") == 5


def _res(structured=None, text="", status="ok") -> R.CallResult:
    inv = R.Invocation(cmd=[], returncode=0, stdout="", stderr="", wall_ms=1)
    return R.CallResult(invocation=inv, result=None, status=status,
                        structured_output=structured, text=text)


def test_parse_label_from_structured_output_or_text():
    assert A.parse_label(_res(structured={"label": "Rivers of Europe"})) == "Rivers of Europe"
    assert A.parse_label(_res(structured='{"label": "Politicians"}')) == "Politicians"
    assert A.parse_label(_res(text='{"label": "Video games"}')) == "Video games"
    assert A.parse_label(_res(text="Video games")) == "Video games"
    assert A.parse_label(_res()) is None


@pytest.fixture()
def arm_a_run(tmp_path) -> Path:
    d = tmp_path / "runA"
    d.mkdir()
    state = {"arm": "A",
             "pages": {"d1": {"title": "Danube", "category": "c00", "related": [], "related_raw": []},
                       "d2": {"title": "Rhine", "category": "c00", "related": [], "related_raw": []},
                       "d3": {"title": "Marie Curie", "category": "c01", "related": [], "related_raw": []}},
             "categories": {"c00": {"label": "river, danube, europe", "terms": ["river", "danube", "europe"],
                                    "parent": None},
                            "c01": {"label": "physicist, radium, nobel",
                                    "terms": ["physicist", "radium", "nobel"], "parent": None}},
             "extra_pages": []}
    (d / "wiki_state.json").write_text(json.dumps(state), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({"run_id": "A_test", "corpus": "test", "n": 3}),
                                     encoding="utf-8")
    return d


class FakeA:
    def __init__(self, labels):
        self.labels = list(labels)
        self.n = 0

    def __call__(self, cmd, stdin=None, timeout=180, cwd=None, env=None):
        label = self.labels[min(self.n, len(self.labels) - 1)]
        self.n += 1
        result = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
                  "duration_ms": 2000, "duration_api_ms": 1900, "total_cost_usd": 0.002,
                  "result": json.dumps({"label": label}), "structured_output": {"label": label},
                  "permission_denials": [],
                  "modelUsage": {"claude-sonnet-5": {"inputTokens": 200, "outputTokens": 20,
                                                     "cacheReadInputTokens": 100,
                                                     "cacheCreationInputTokens": 0, "costUSD": 0.002}}}
        return R.Invocation(cmd=list(cmd), returncode=0, stdout=json.dumps(result), stderr="", wall_ms=2000)


def test_naming_replaces_only_the_labels(tmp_path, arm_a_run, monkeypatch):
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    monkeypatch.setattr(R, "run_claude", FakeA(["Rivers of Europe", "Physicists"]))
    run = A.ArmAPlusRun(arm_a_run, runs_root=tmp_path / "runs", out_dir=tmp_path / "out",
                        concurrency=2, log=lambda _m: None)
    summary = run.run()
    state = json.loads((tmp_path / "out" / "wiki_state.json").read_text(encoding="utf-8"))
    assert state["arm"] == "A+"
    assert {c: v["label"] for c, v in state["categories"].items()} == {
        "c00": "Rivers of Europe", "c01": "Physicists"}
    assert state["pages"]["d1"]["category"] == "c00", "the partition is untouched"
    assert state["categories"]["c00"]["terms"] == ["river", "danube", "europe"]
    assert summary["n_clusters"] == 2 and summary["n_fallback"] == 0
    assert (tmp_path / "out" / "llm_calls.jsonl").exists()
    rows = run.calls.rows()
    assert len(rows) == 2 and all(r["kind"] == "name_cluster" for r in rows)


def test_a_too_long_label_is_retried_once_then_falls_back(tmp_path, arm_a_run, monkeypatch):
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    monkeypatch.setattr(R, "run_claude", FakeA(["A very long label that is well over six words"]))
    run = A.ArmAPlusRun(arm_a_run, runs_root=tmp_path / "runs", out_dir=tmp_path / "out",
                        concurrency=1, log=lambda _m: None)
    summary = run.run()
    assert summary["n_fallback"] == 2
    state = json.loads((tmp_path / "out" / "wiki_state.json").read_text(encoding="utf-8"))
    assert state["categories"]["c00"]["label"] == "river, danube, europe"
    rows = run.calls.rows()
    assert len(rows) == 4 and all(r["status"] == "invalid_label" for r in rows)
    assert rows[0]["validation"]["n_words"] > 6


def test_command_is_toolless_with_a_one_field_schema(tmp_path, arm_a_run, monkeypatch):
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")
    run = A.ArmAPlusRun(arm_a_run, runs_root=tmp_path / "runs", out_dir=tmp_path / "out",
                        log=lambda _m: None)
    cmd = run.cmd()
    assert cmd[cmd.index("--tools") + 1] == ""
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == {
        "type": "object", "additionalProperties": False,
        "properties": {"label": {"type": "string"}}, "required": ["label"]}
    assert cmd[cmd.index("--max-budget-usd") + 1] == "0.05"
    assert "at most 6 words" in cmd[cmd.index("--system-prompt") + 1]
