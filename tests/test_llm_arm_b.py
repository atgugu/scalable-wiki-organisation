"""Arm B: git store, resume, failure handling, budget cap and the stop rule.

Every test drives the real :class:`~wikiorg.llm.arm_b.ArmBRun` loop with the CLI
subprocess replaced by a fake that writes plausible wiki pages, so the git
bookkeeping, the ledger and the resume logic are exercised end to end without
network access or money.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from wikiorg.data.docs import Doc, write_doc, write_order
from wikiorg.llm import arm_b as B
from wikiorg.llm import runner as R

pytestmark = pytest.mark.timeout(120)

DOC_IDS = [f"dbp-{i:04d}" for i in range(1, 7)]


# ================================================================================================
# fixtures
# ================================================================================================
@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    for i, doc_id in enumerate(DOC_IDS):
        write_doc(d, Doc(id=doc_id, title=f"Document {i + 1}",
                         body=f"Body of document {i + 1}. " * 20))
    write_order(d / "orders" / "order-1.txt", DOC_IDS)
    (d / "manifest.json").write_text('{"corpus": "test"}', encoding="utf-8")
    return d


def result_json(cost: float = 0.10, turns: int = 11, files: int = 4, topic: str = "topic-a",
                new_topic: str = "yes", subtype: str = "success", denials=()) -> dict:
    return {
        "type": "result", "subtype": subtype, "is_error": subtype != "success",
        "duration_ms": 55000, "duration_api_ms": 50000, "num_turns": turns,
        "session_id": "s", "result": f"DONE topic={topic} new_topic={new_topic} files={files}",
        "total_cost_usd": cost, "permission_denials": list(denials),
        "modelUsage": {"claude-sonnet-5": {
            "inputTokens": 40, "outputTokens": 1200, "cacheReadInputTokens": 60000 + files * 500,
            "cacheCreationInputTokens": 9000, "costUSD": cost}},
    }


def stream(result: dict) -> str:
    lines = [json.dumps({"type": "system", "subtype": "init", "model": "claude-sonnet-5",
                         "tools": ["Read", "Write", "Edit", "Glob", "Grep"], "mcp_servers": []})]
    for name in ("Read", "Glob", "Write", "Edit"):
        lines.append(json.dumps({"type": "assistant",
                                 "message": {"content": [{"type": "tool_use", "name": name, "input": {}}]}}))
    lines.append(json.dumps(result))
    return "\n".join(lines) + "\n"


def doc_id_of(cmd) -> str | None:
    for i, part in enumerate(cmd):
        if part == "-p" and i + 1 < len(cmd):
            prompt = cmd[i + 1]
            if prompt.startswith("Ingest raw/"):
                return prompt[len("Ingest raw/"):].split(".md", 1)[0]
    return None


class FakeCLI:
    """Writes the pages a well-behaved agent would write, and reports it."""

    def __init__(self, plan: dict | None = None, cost: float = 0.10):
        self.plan = plan or {}            # doc_id -> "budget" | "turns" | "timeout" | "denial" | "tamper"
        self.cost = cost
        self.calls: list[dict] = []

    def __call__(self, cmd, stdin=None, timeout=1200, cwd=None, env=None):
        cwd = Path(cwd)
        doc_id = doc_id_of(cmd)
        n = len(self.calls) + 1
        self.calls.append({"doc_id": doc_id, "cmd": list(cmd), "timeout": timeout})
        mode = self.plan.get(doc_id)
        if isinstance(mode, list):                        # per-attempt behaviour
            seen = sum(1 for c in self.calls if c["doc_id"] == doc_id)
            mode = mode[min(seen - 1, len(mode) - 1)]
        wiki = cwd / "wiki"
        (wiki / "sources").mkdir(parents=True, exist_ok=True)
        (wiki / "topics").mkdir(parents=True, exist_ok=True)
        if doc_id is None:                                # a lint
            (wiki / "overview.md").write_text("# Overview\nlinted\n", encoding="utf-8")
            return R.Invocation(cmd=list(cmd), returncode=0,
                                stdout=stream(result_json(0.9, 30, 3, subtype="success")),
                                stderr="", wall_ms=9000)
        topic = "topic-a" if n % 2 else "topic-b"
        (wiki / "sources" / f"{doc_id}.md").write_text(
            f"---\ntitle: \"Document {doc_id}\"\ntype: source\nsource: raw/{doc_id}.md\n"
            f"topic: {topic}\nrelated: []\nentities: []\nconcepts: []\ningest: {n}\n---\n"
            f"# Document {doc_id}\nSummary text.\n", encoding="utf-8")
        (wiki / "topics" / f"{topic}.md").write_text(
            f"---\ntitle: \"Topic {topic}\"\ntype: topic\n---\n# Topic {topic}\nScope.\n\n"
            f"## Sources\n- [Document {doc_id}](../sources/{doc_id}.md)\n", encoding="utf-8")
        (wiki / "index.md").write_text("# Index\n\n## Topics\n\n## Sources\n", encoding="utf-8")
        (wiki / "log.md").write_text((wiki / "log.md").read_text(encoding="utf-8") if (wiki / "log.md").exists()
                                     else "" + f"## ingest {doc_id}\n", encoding="utf-8")
        if mode == "tamper":
            (cwd / "CLAUDE.md").write_text("# hijacked\n", encoding="utf-8")
        if mode == "timeout":
            return R.Invocation(cmd=list(cmd), returncode=-9, stdout="", stderr="", wall_ms=1000, timed_out=True)
        if mode in ("budget", "turns"):
            sub = "error_max_budget_usd" if mode == "budget" else "error_max_turns"
            return R.Invocation(cmd=list(cmd), returncode=0,
                                stdout=stream(result_json(self.cost, 80, 4, subtype=sub)),
                                stderr="", wall_ms=5000)
        if mode == "denial":
            return R.Invocation(cmd=list(cmd), returncode=0,
                                stdout=stream(result_json(self.cost, 3, 0, denials=[{"tool": "Bash"}])),
                                stderr="", wall_ms=5000)
        return R.Invocation(cmd=list(cmd), returncode=0,
                            stdout=stream(result_json(self.cost, 11, 4, topic=topic)),
                            stderr="", wall_ms=50000)


@pytest.fixture()
def make_run(tmp_path, corpus, monkeypatch):
    monkeypatch.setattr(R, "assert_cli_version", lambda *a, **k: "2.1.258")

    def _make(fake: FakeCLI, n: int = 3, **kw) -> B.ArmBRun:
        monkeypatch.setattr(R, "run_claude", fake)
        opts = {"lint_every": 0, "tag_every": 0, "snapshot_at": (), "apply_stop_rule": False}
        opts.update(kw)
        cfg = B.ArmBConfig(run_id="t1", corpus="test", n=n, corpus_dir=corpus,
                           runs_root=tmp_path / "runs", **opts)
        return B.ArmBRun(cfg, log=lambda _m: None)

    return _make


# ================================================================================================
# pure helpers
# ================================================================================================
def test_budget_cap_clamps_the_running_median():
    assert B.budget_cap([]) == 2.0                       # no history -> the default
    assert B.budget_cap([0.10] * 20) == 1.5              # 4*0.10 = 0.40 -> floor
    assert B.budget_cap([0.50] * 20) == pytest.approx(2.0)
    assert B.budget_cap([2.00] * 20) == 4.0              # 4*2.0 = 8.0 -> ceiling
    # only the last 20 successes count, and None/0 are ignored
    assert B.budget_cap([5.0] * 30 + [0.5] * 20) == pytest.approx(2.0)
    assert B.budget_cap([None, 0.0, 0.5, 0.5, 0.5]) == pytest.approx(2.0)


def test_parse_head_messages():
    assert B.parse_head("ingest 0042 dbp-x") == ("ingest", 42, "dbp-x")
    assert B.parse_head("lint 0050") == ("lint", 50, None)
    assert B.parse_head("init schema") == (None, None, None)
    assert B.parse_head(None) == (None, None, None)


def test_page_counts(tmp_path):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "topics").mkdir()
    (wiki / "sources" / "a.md").write_text("x")
    (wiki / "sources" / "b.md").write_text("x")
    (wiki / "topics" / "t.md").write_text("x")
    (wiki / "index.md").write_text("x")
    counts = B.page_counts(wiki)
    assert counts["sources"] == 2 and counts["topics"] == 1 and counts["root"] == 1
    assert counts["total"] == 4
    assert B.page_counts(tmp_path / "nope")["total"] == 0


# ================================================================================================
# stop rule
# ================================================================================================
def _rows(n, usd, pages_per_ingest=6.0, base_pages=0.0, waited_ms=0):
    return [{"kind": "ingest", "status": "ok", "ingest_index": i, "doc_id": f"d{i}",
             "cost_usd": usd(i), "wall_ms": 60000, "waited_ms": waited_ms,
             "wiki_pages_before": {"total": base_pages + pages_per_ingest * (i - 1)}}
            for i in range(1, n + 1)]


def test_stop_rule_continues_for_a_cheap_flat_run():
    now = datetime(2026, 9, 3, 12, 0).astimezone()
    started = (now - timedelta(hours=2)).isoformat()
    proj = B.project_run(_rows(20, lambda i: 0.11), n_target=300, started_at=started, now=now)
    assert proj["n_done"] == 20
    assert proj["projected_total_usd"] < 400
    assert proj["stop"] is False and proj["over_budget"] is False
    # 20 ingests in 2 h -> 280 left at ~6 min each is well inside the deadline
    assert proj["over_deadline"] is False


def test_stop_rule_fires_on_dollars_when_cost_grows_with_the_wiki():
    now = datetime(2026, 9, 3, 12, 0).astimezone()
    started = (now - timedelta(hours=2)).isoformat()
    proj = B.project_run(_rows(20, lambda i: 0.20 + 0.02 * (i - 1)), n_target=300,
                         started_at=started, now=now)
    assert proj["cost_slope_usd_per_page"] > 0
    assert proj["projected_total_usd"] > 400
    assert proj["over_budget"] is True and proj["stop"] is True and proj["stop_at"] == 200
    assert "projected $" in proj["reason"]


def test_stop_rule_fires_on_the_calendar_when_usage_limits_throttle_the_run():
    """10 days for 20 ingests, almost all of it spent waiting for the subscription."""
    now = datetime(2026, 10, 1, 12, 0).astimezone()
    started = (now - timedelta(days=10)).isoformat()
    rows = _rows(20, lambda i: 0.05, waited_ms=43140_000)   # 10 days / 20 ingests
    proj = B.project_run(rows, n_target=300, started_at=started, now=now)
    assert proj["over_budget"] is False
    assert proj["usage_limit_wait_hours"] > 200 and proj["active_hours"] > 200
    assert proj["over_deadline"] is True and proj["stop"] is True
    assert "projected finish" in proj["reason"]


def test_hours_the_operator_simply_had_the_run_stopped_do_not_trigger_the_rule():
    """A run paused for a week is not a slow run; only active time sets the pace."""
    now = datetime(2026, 10, 1, 12, 0).astimezone()
    started = (now - timedelta(days=10)).isoformat()
    proj = B.project_run(_rows(20, lambda i: 0.05), n_target=300, started_at=started, now=now)
    assert proj["idle_ratio"] > 100 and proj["active_hours"] < 1
    assert proj["over_deadline"] is False and proj["stop"] is False
    # the pessimistic wall-pace date is still reported for the operator to judge
    assert proj["projected_finish_at_elapsed_pace"] > proj["projected_finish"]


def test_a_negative_noise_slope_cannot_make_the_remaining_ingests_look_free():
    """With few points the fit is easily downward; the projection floors at the median."""
    now = datetime(2026, 9, 3, 12, 0).astimezone()
    rows = _rows(4, lambda i: [0.22, 0.16, 0.20, 0.18][i - 1], pages_per_ingest=7.0)
    proj = B.project_run(rows, n_target=300, started_at=(now - timedelta(hours=1)).isoformat(), now=now)
    assert proj["cost_slope_usd_per_page"] < 0, "this fixture reproduces the noisy negative slope"
    assert proj["projected_total_usd_unfloored"] < 40, "the raw fit collapses to nearly free"
    assert proj["projected_total_usd"] > 250 * proj["median_ingest_usd"]
    assert proj["floor_ingest_usd"] >= proj["median_ingest_usd"]


def test_projection_counts_the_remaining_lints_and_prints():
    now = datetime(2026, 9, 3, 12, 0).astimezone()
    proj = B.project_run(_rows(20, lambda i: 0.10), n_target=300,
                         started_at=(now - timedelta(hours=2)).isoformat(), now=now,
                         lint_every=50, lint_usd=6.0)
    assert proj["n_future_lints"] == 6                      # 50,100,150,200,250,300 -> 300 is the last
    assert proj["projected_total_usd"] >= 6 * 6.0
    text = B.format_projection(proj)
    assert "PROJECTED TOTAL" in text and "PROJECTED FINISH" in text and "DECISION" in text


def test_projection_survives_too_few_points():
    now = datetime(2026, 9, 3, 12, 0).astimezone()
    proj = B.project_run(_rows(2, lambda i: 0.10), n_target=300,
                         started_at=(now - timedelta(minutes=5)).isoformat(), now=now)
    assert proj["projected_total_usd"] > 0 and proj["n_remaining"] == 298


# ================================================================================================
# git store
# ================================================================================================
def test_git_store_hides_the_repository_from_the_work_tree(tmp_path):
    work, gitdir = tmp_path / "run", tmp_path / "run.git"
    store = B.GitStore(work, gitdir)
    store.init()
    assert not (work / ".git").exists(), "the agent must never see a .git entry"
    (work / "CLAUDE.md").write_text("schema", encoding="utf-8")
    store.add_all()
    sha = store.commit("init schema")
    assert sha and store.head_message() == "init schema"
    assert gitdir.is_dir()


def test_reset_keeps_the_new_source_on_retry_but_drops_it_on_resume(tmp_path):
    work, gitdir = tmp_path / "run", tmp_path / "run.git"
    store = B.GitStore(work, gitdir)
    store.init()
    (work / "raw").mkdir()
    (work / "wiki").mkdir()
    (work / "raw" / "old.md").write_text("old", encoding="utf-8")
    store.add_all()
    store.commit("ingest 0001 old")

    (work / "raw" / "new.md").write_text("new", encoding="utf-8")     # untracked, in flight
    (work / "wiki" / "half.md").write_text("half written", encoding="utf-8")
    store.reset_hard(clean_raw=False)
    assert (work / "raw" / "new.md").exists(), "a retry needs its source"
    assert not (work / "wiki" / "half.md").exists()
    assert (work / "raw" / "old.md").exists()

    store.reset_hard(clean_raw=True)
    assert not (work / "raw" / "new.md").exists()
    assert (work / "raw" / "old.md").exists(), "committed sources survive every reset"


def test_status_classification_and_protected_files(tmp_path):
    work, gitdir = tmp_path / "run", tmp_path / "run.git"
    run_store = B.GitStore(work, gitdir)
    run_store.init()
    (work / "raw").mkdir()
    (work / "wiki").mkdir()
    (work / "CLAUDE.md").write_text("schema", encoding="utf-8")
    (work / "raw" / "a.md").write_text("a", encoding="utf-8")
    run_store.add_all()
    run_store.commit("ingest 0001 a")

    (work / "raw" / "b.md").write_text("b", encoding="utf-8")     # the new source: allowed
    (work / "raw" / "a.md").write_text("tampered", encoding="utf-8")
    (work / "CLAUDE.md").write_text("hijacked", encoding="utf-8")
    (work / "wiki" / "index.md").write_text("i", encoding="utf-8")
    run_store.add_all()

    class Holder(B.ArmBRun):                      # reuse the classifier without a full run
        def __init__(self, store):
            self.git = store

    touched, violations = Holder(run_store)._classify_status("b")
    assert sorted(violations) == ["CLAUDE.md", "raw/a.md"]
    assert "wiki/index.md" in touched["created"] and touched["n_wiki"] == 1
    run_store.restore(violations)
    assert (work / "CLAUDE.md").read_text() == "schema"
    assert (work / "raw" / "a.md").read_text() == "a"


# ================================================================================================
# the ingest loop
# ================================================================================================
def test_three_ingests_write_files_commits_and_ledger_rows(make_run):
    fake = FakeCLI()
    run = make_run(fake, n=3)
    summary = run.run()
    assert summary["n_ingested"] == 3
    rows = run.calls.rows()
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 3
    assert [r["ingest_index"] for r in ok] == [1, 2, 3]
    assert [r["doc_id"] for r in ok] == DOC_IDS[:3]
    for r in ok:
        assert r["git_commit"] and r["files_touched"]["n_wiki"] >= 1
        assert r["tokens"]["cache_read"] > 0 and r["cost_usd"] > 0
        assert r["done_line"].startswith("DONE topic=")
        assert r["claude_md_sha256"] == run.claude_md_sha
        assert r["permission_denials"] == []
        assert (run.calls.stdout_dir / r["stdout_file"]).exists()
    # wiki size before ingest 1 is zero and grows
    assert ok[0]["wiki_pages_before"]["total"] == 0
    assert ok[2]["wiki_pages_before"]["total"] > 0
    log = run.git.git("log", "--pretty=%s").stdout.split("\n")
    assert log[0] == f"ingest 0003 {DOC_IDS[2]}" and log[-2] == "init schema"
    for doc_id in DOC_IDS[:3]:
        assert (run.raw / f"{doc_id}.md").exists()
    assert not (run.raw / f"{DOC_IDS[3]}.md").exists(), "un-ingested sources must stay invisible"


def test_raw_copy_is_byte_identical_to_what_arm_a_embeds(make_run, corpus):
    run = make_run(FakeCLI(), n=2)
    run.run()
    for doc_id in DOC_IDS[:2]:
        assert (run.raw / f"{doc_id}.md").read_bytes() == (corpus / "docs" / f"{doc_id}.md").read_bytes()


def test_resume_continues_at_the_next_position(make_run):
    fake = FakeCLI()
    run = make_run(fake, n=5, stop_after=2)
    run.run()
    assert len(run.calls.ok_rows("ingest")) == 2
    del run                                   # release the lock

    run2 = make_run(fake, n=5)
    state = run2.prepare()
    assert state["next_pos"] == 2 and state["next_doc"] == DOC_IDS[2]
    assert state["consistent"] and not state["orphan_commit"]
    run2.run()
    ok = run2.calls.ok_rows("ingest")
    assert [r["ingest_index"] for r in ok] == [1, 2, 3, 4, 5]
    assert [r["doc_id"] for r in ok] == DOC_IDS[:5]
    assert run2.git.head_message() == f"ingest 0005 {DOC_IDS[4]}"


def test_resume_removes_a_stray_half_ingested_source(make_run):
    run = make_run(FakeCLI(), n=3, stop_after=1)
    run.run()
    stray = run.raw / f"{DOC_IDS[1]}.md"
    stray.write_text("copied in just before the kill", encoding="utf-8")
    (run.wiki / "sources" / "junk.md").write_text("half written", encoding="utf-8")
    del run

    run2 = make_run(FakeCLI(), n=3)
    run2.prepare()
    assert not stray.exists()
    assert not (run2.wiki / "sources" / "junk.md").exists()


def test_resume_recovers_a_commit_whose_ledger_row_was_never_written(make_run):
    """Killed between ``git commit`` and the ledger append: the row is rebuilt."""
    run = make_run(FakeCLI(), n=3, stop_after=2)
    run.run()
    rows = R.read_jsonl(run.calls.path)
    assert rows[-1]["ingest_index"] == 2
    run.calls.path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8")
    del run

    run2 = make_run(FakeCLI(), n=3)
    state = run2.prepare()
    assert state["next_pos"] == 2, "the recovered row counts as done"
    recovered = [r for r in run2.calls.rows() if r.get("recovered")]
    assert len(recovered) == 1 and recovered[0]["ingest_index"] == 2
    assert recovered[0]["status"] == "ok" and recovered[0]["cost_usd"] > 0


def test_resume_drops_an_orphan_commit_with_no_archived_stdout(make_run):
    run = make_run(FakeCLI(), n=3, stop_after=2)
    run.run()
    rows = R.read_jsonl(run.calls.path)
    run.calls.path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8")
    for p in run.calls.stdout_dir.glob("ingest-0002-*"):
        p.unlink()
    del run

    run2 = make_run(FakeCLI(), n=3)
    state = run2.prepare()
    assert state["head_index"] == 1 and state["next_pos"] == 1


def test_resume_aborts_on_an_unexplainable_mismatch(make_run):
    run = make_run(FakeCLI(), n=3, stop_after=2)
    run.run()
    run.git.git("reset", "-q", "--hard", "HEAD~2")            # HEAD far behind the ledger
    del run
    run2 = make_run(FakeCLI(), n=3)
    with pytest.raises(R.HarnessError, match="cannot resume"):
        run2.prepare()


def test_a_second_process_cannot_take_the_lock(make_run):
    run = make_run(FakeCLI(), n=1)
    run.prepare()
    run2 = make_run(FakeCLI(), n=1)
    with pytest.raises(R.HarnessError, match="another wikiorg run-b process"):
        run2.prepare()


# ================================================================================================
# failure handling
# ================================================================================================
@pytest.mark.parametrize("mode", ["budget", "turns", "timeout"])
def test_budget_turns_timeout_retry_once_at_1_5x_then_fail_and_continue(make_run, mode):
    fake = FakeCLI(plan={DOC_IDS[1]: mode})
    run = make_run(fake, n=3)
    run.run()
    rows = run.calls.rows()
    attempts = [r for r in rows if r["doc_id"] == DOC_IDS[1]]
    assert len(attempts) == 3, "two attempts plus the final failed row"
    assert attempts[0]["max_turns"] == 80 and attempts[1]["max_turns"] == 120
    assert attempts[1]["budget_usd"] == pytest.approx(attempts[0]["budget_usd"] * 1.5)
    assert attempts[1]["timeout_s"] == pytest.approx(attempts[0]["timeout_s"] * 1.5)
    assert attempts[-1]["status"] == "failed"
    assert not (run.raw / f"{DOC_IDS[1]}.md").exists(), "a failed source is removed again"
    assert not (run.wiki / "sources" / f"{DOC_IDS[1]}.md").exists()
    ok = run.calls.ok_rows("ingest")
    assert [r["doc_id"] for r in ok] == [DOC_IDS[0], DOC_IDS[2]], "the run continues"
    # every attempt is billed, so the ledger keeps them all
    assert sum(1 for r in rows if r["kind"] == "ingest" and r["status"] in ("budget", "turns", "timeout")) == 2


def test_retry_succeeds_on_the_second_attempt(make_run):
    fake = FakeCLI(plan={DOC_IDS[0]: ["budget", "ok"]})
    run = make_run(fake, n=2)
    run.run()
    rows = [r for r in run.calls.rows() if r["doc_id"] == DOC_IDS[0]]
    assert [r["status"] for r in rows] == ["budget", "ok"]
    assert rows[1]["git_commit"]
    log = run.git.git("log", "--pretty=%s").stdout.split()
    assert "init" in log and run.git.head_message() == f"ingest 0002 {DOC_IDS[1]}"


def test_permission_denials_abort_the_run(make_run):
    run = make_run(FakeCLI(plan={DOC_IDS[0]: "denial"}), n=3)
    with pytest.raises(R.HarnessError, match="permission_denials"):
        run.run()
    assert run.calls.rows()[-1]["status"] == "aborted_permission_denial"


def test_an_agent_editing_the_schema_is_reverted_and_flagged(make_run):
    run = make_run(FakeCLI(plan={DOC_IDS[0]: "tamper"}), n=1)
    run.run()
    row = run.calls.ok_rows("ingest")[0]
    assert row["protected_files_touched"] == ["CLAUDE.md"]
    assert (run.workdir / "CLAUDE.md").read_text(encoding="utf-8").startswith("# Wiki maintainer schema")


# ================================================================================================
# lints, tags and snapshots
# ================================================================================================
def test_lints_tags_and_snapshots_fire_on_schedule(make_run):
    run = make_run(FakeCLI(), n=4, lint_every=2, tag_every=2, snapshot_at=(4,))
    run.run()
    rows = run.calls.rows()
    lints = [r for r in rows if r["kind"] == "lint"]
    assert [r["ingest_index"] for r in lints] == [2, 4]
    assert lints[0]["budget_usd"] == B.LINT_BUDGET and lints[0]["max_turns"] == B.LINT_MAX_TURNS
    assert lints[0]["timeout_s"] == B.LINT_TIMEOUT
    tags = run.git.git("tag").stdout.split()
    assert tags == ["ingest-0002", "ingest-0004"]
    snaps = list((run.meta / "snapshots").glob("*.tar.gz"))
    assert len(snaps) == 1 and "N0004" in snaps[0].name
    assert run.git.head_message() == "lint 0004"


def test_stop_rule_shortens_the_run_in_the_loop(make_run, monkeypatch):
    fake = FakeCLI(cost=5.0)                                # deliberately over budget
    run = make_run(fake, n=6, apply_stop_rule=True, stop_rule_after=2, usd_limit=10.0, stop_at=3)
    run.run()
    assert len(run.calls.ok_rows("ingest")) == 3
    decisions = R.read_jsonl(run.calls.decisions_path)
    assert any(d["decision"] == "stop_rule_triggered" for d in decisions)
    triggered = [d for d in decisions if d["decision"] == "stop_rule_triggered"][0]
    assert triggered["n_target"] == 3 and triggered["over_budget"] is True


# ================================================================================================
# the exact command
# ================================================================================================
def test_ingest_command_matches_appendix_c(make_run):
    run = make_run(FakeCLI(), n=1)
    cmd = run.ingest_cmd("dbp-x", 2.0, 80)
    assert cmd[:3] == ["claude", "-p", "Ingest raw/dbp-x.md following CLAUDE.md."]
    joined = " ".join(cmd)
    for flag in ("--output-format stream-json", "--verbose", "--permission-mode acceptEdits",
                 "--setting-sources project", "--strict-mcp-config", "--disable-slash-commands",
                 "--no-session-persistence", "--max-budget-usd 2", "--max-turns 80",
                 "--model claude-sonnet-5", "--effort high"):
        assert flag in joined, flag
    assert cmd[cmd.index("--tools") + 1] == "Read,Write,Edit,Glob,Grep"
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Write,Edit,Glob,Grep"
    assert "MultiEdit" not in joined and "--fallback-model" not in joined
    lint = run.lint_cmd()
    assert lint[2] == "Run the lint procedure in CLAUDE.md."
    assert lint[lint.index("--max-budget-usd") + 1] == "6"
    assert lint[lint.index("--max-turns") + 1] == "200"


def test_a_kill_inside_a_milestone_lint_is_repaired_on_resume(make_run):
    """The N=100 snapshot is the pre-registered fallback artefact; a lint runs for
    up to 1800 s six times over days, so being killed inside one is expected."""
    # the state a kill inside lint(2) leaves behind: ingest 2 committed and logged,
    # no lint row, no tag, no snapshot
    run = make_run(FakeCLI(), n=4, lint_every=0, tag_every=0, snapshot_at=(), stop_after=2)
    run.run()
    assert run.git.head_message() == f"ingest 0002 {DOC_IDS[1]}"
    assert not [r for r in run.calls.rows() if r["kind"] == "lint"]
    assert not list((run.meta / "snapshots").glob("*.tar.gz"))
    del run

    run2 = make_run(FakeCLI(), n=4, lint_every=2, tag_every=2, snapshot_at=(2,))
    run2.run()
    lints = [r for r in run2.calls.rows() if r["kind"] == "lint"]
    assert [r["ingest_index"] for r in lints] == [2, 4], "the missed lint at 2 is run on resume"
    assert len(list((run2.meta / "snapshots").glob("*N0002*"))) == 1
    assert run2.git.git("tag").stdout.split() == ["ingest-0002", "ingest-0004"]


def test_rerunning_a_finished_run_does_not_lint_or_snapshot_twice(make_run):
    run = make_run(FakeCLI(), n=2, lint_every=2, tag_every=2, snapshot_at=(2,))
    run.run()
    n_lints = len([r for r in run.calls.rows() if r["kind"] == "lint"])
    head = run.git.head_sha()
    del run
    run2 = make_run(FakeCLI(), n=2, lint_every=2, tag_every=2, snapshot_at=(2,))
    run2.run()
    assert len([r for r in run2.calls.rows() if r["kind"] == "lint"]) == n_lints == 1
    assert run2.git.head_sha() == head, "no new commits on a finished run"


def test_a_lint_commit_with_no_ledger_row_is_recovered(make_run):
    run = make_run(FakeCLI(), n=2, lint_every=2, tag_every=0, snapshot_at=())
    run.run()
    rows = R.read_jsonl(run.calls.path)
    assert rows[-1]["kind"] == "lint"
    run.calls.path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8")
    del run

    run2 = make_run(FakeCLI(), n=2, lint_every=2, tag_every=0, snapshot_at=())
    state = run2.prepare()
    assert state["consistent"] and not state["orphan_commit"]
    recovered = [r for r in run2.calls.rows() if r.get("recovered")]
    assert len(recovered) == 1 and recovered[0]["kind"] == "lint"
    assert recovered[0]["cost_usd"] > 0, "a lint is worth up to $6; it must stay in the ledger"


def test_a_triggered_stop_rule_still_takes_its_snapshot(make_run):
    run = make_run(FakeCLI(cost=5.0), n=6, lint_every=0, tag_every=0, snapshot_at=(),
                   apply_stop_rule=True, stop_rule_after=2, usd_limit=10.0, stop_at=3)
    run.run()
    snaps = list((run.meta / "snapshots").glob("*.tar.gz"))
    assert len(snaps) == 1 and "N0003" in snaps[0].name


def test_run_claude_kills_the_child_when_the_parent_is_interrupted(tmp_path):
    """A surviving `claude` would keep writing into the work tree during a resume."""
    marker = tmp_path / "still-alive"
    script = tmp_path / "slow.sh"
    script.write_text(f"#!/bin/sh\nsleep 5\ntouch {marker}\n", encoding="utf-8")
    script.chmod(0o755)
    import signal
    import time as _t

    def _boom(_s, _f):
        raise KeyboardInterrupt("SIGALRM stands in for Ctrl-C / SIGTERM")

    old = signal.signal(signal.SIGALRM, _boom)
    signal.setitimer(signal.ITIMER_REAL, 0.4)
    try:
        with pytest.raises(KeyboardInterrupt):
            R.run_claude([str(script)], timeout=30)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    _t.sleep(6)
    assert not marker.exists(), "the child outlived the interrupted parent"
