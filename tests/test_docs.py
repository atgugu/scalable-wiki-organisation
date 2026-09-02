"""Document format: round trip, frontmatter restricted to id/title, no gold leakage into docs."""
import json
import re
from pathlib import Path

import pytest

from wikiorg.data.docs import Doc, doc_markdown, parse_doc, read_docs, read_gold, read_order, sha256_docs

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smoke20"
_FM_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_FORBIDDEN_KEY_RE = re.compile(r"^(l1|l2|l3|class|class_id|class_name|label|labels|links|category|topic|gold)\s*:", re.M | re.I)


def _corpus_dirs():
    dirs = [FIXTURE]
    for c in ("dbpedia", "wikics"):
        for d in sorted((ROOT / "data" / c).glob("*")):
            if d.is_dir() and (d / "docs").is_dir() and (d / "gold.jsonl").exists():
                dirs.append(d)
    return dirs


def test_round_trip_and_quoting():
    for title in ('Plain', 'Colon: here', 'Quote "x" and \'y\'', 'Ünïcödé — dash', '[bracket] #hash', '  padded '):
        doc = Doc(id="abc-1", title=title, body="line one\n\n## heading\n\nline two: with colon")
        text = doc_markdown(doc)
        assert text.startswith('---\nid: "abc-1"\ntitle: ')
        back = parse_doc(text)
        assert (back.id, back.title, back.body) == (doc.id, doc.title, doc.body)


def test_invalid_ids_and_titles_rejected():
    with pytest.raises(ValueError):
        Doc(id="has space", title="t", body="b")
    with pytest.raises(ValueError):
        Doc(id="ok", title="   ", body="b")


def test_parse_rejects_extra_frontmatter_keys():
    with pytest.raises(ValueError):
        parse_doc('---\nid: "x"\ntitle: "t"\nl2: "Leak"\n---\n\nbody\n')


@pytest.mark.parametrize("corpus_dir", _corpus_dirs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_docs_contain_only_id_and_title_and_no_gold(corpus_dir):
    docs = read_docs(corpus_dir)
    gold = read_gold(corpus_dir)
    assert docs and {d.id for d in docs} == set(gold)
    gold_names = set()
    for row in gold.values():
        for k, v in row.items():
            if k in ("l1", "l2", "l3", "class") and isinstance(v, str):
                gold_names.add(v)
    for doc in docs:
        raw = (corpus_dir / "docs" / f"{doc.id}.md").read_text(encoding="utf-8")
        m = _FM_RE.match(raw)
        assert m, f"{doc.id}: no frontmatter"
        keys = [ln.split(":", 1)[0].strip() for ln in m.group(1).splitlines() if ln.strip()]
        assert keys == ["id", "title"], f"{doc.id}: frontmatter keys {keys}"
        assert not _FORBIDDEN_KEY_RE.search(m.group(1)), f"{doc.id}: gold-like key in frontmatter"
        # multi-word CamelCase gold names (e.g. FootballClub, RouteOfTransportation) must not occur anywhere
        for name in gold_names:
            if re.search(r"[a-z][A-Z]", name):
                assert name not in raw, f"{doc.id}: gold class name {name!r} found in doc"
    for order in (corpus_dir / "orders").glob("*.txt"):
        ids = read_order(order)
        assert set(ids) == set(gold) and len(ids) == len(set(ids)), f"{order}: not a permutation of the corpus"


def test_fixture_manifest_matches_docs():
    docs = read_docs(FIXTURE)
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    assert manifest["n"] == len(docs) == 20 and manifest["oracle_k"] == 4
    assert manifest["sha256"]["docs"] == sha256_docs(docs)
