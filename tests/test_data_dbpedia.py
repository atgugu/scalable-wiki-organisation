"""DBPedia corpus builder: titles, escapes, allocation and nested sampling on a synthetic pool."""
import collections
import hashlib

import pytest

from wikiorg.data.dbpedia import (allocate_2000, apportion, clean_text, derive_title, first_sentence, sample_nested,
                                  select_classes)


@pytest.mark.parametrize("text,title", [
    ("John Smith (born 1950) is an American actor.", "John Smith"),
    ("The Marquam Bridge /ˈmɑːrkəm/ is a double-deck bridge.", "The Marquam Bridge"),
    ("Toros II the Great (Armenian: Թորոս Բ), also Thoros II, (unknown – 1169) was king of Cilicia.", "Toros II the Great"),
    ("Springfield, Illinois is the capital of Illinois.", "Springfield, Illinois"),
    ("Osama bin Laden, the founder of Al-Qaeda, was born in Riyadh.", "Osama bin Laden"),
    ('\\"My Plague\\" is a song by Slipknot.', "My Plague"),
    ("Dr. John A. Smith Jr. was a physician. He was born in 1900.", "Dr. John A. Smith Jr"),
    ("Secular music; Secular means being separate from religion. It is often contrasted with sacred music.", "Secular music"),
    ("The members of the 17th General Assembly of Newfoundland were elected in 1889.", "The members of the 17th General Assembly of Newfoundland"),
])
def test_derive_title(text, title):
    assert derive_title(text) == title


def test_title_cut_at_80_and_fallback():
    long = "The " + "very " * 40 + "long name is a thing."
    t = derive_title(long)
    assert len(t) <= 80 and t.startswith("The very")
    fb = derive_title("Lake Hartridge, with somewhat of a figure eight shape, has a surface area of 200 acres.")
    assert fb == "Lake Hartridge" and derive_title.last_fallback is True


def test_first_sentence_respects_abbreviations():
    assert first_sentence("Mr. Brown went home. He slept.") == "Mr. Brown went home"
    assert first_sentence("It was built in 1901. It burned.") == "It was built in 1901"
    assert first_sentence("A. B. Smith is here. Then.") == "A. B. Smith is here"


def test_clean_text_unescapes_csv_export():
    assert clean_text('four \\"great windows\\", one') == 'four "great windows", one'
    assert clean_text("Commanding: \\n* Lt.-Col. Bell") == "Commanding:\n* Lt.-Col. Bell"
    assert clean_text("Kuma\\\\War") == "Kuma\\War"
    assert clean_text("a \\t b") == "a b"
    assert clean_text('plain "quoted" text') == 'plain "quoted" text'


def test_apportion_and_allocate():
    caps = {"a": 100, "b": 100, "c": 4, "d": 100}
    alloc = apportion(50, {"a": 60.0, "b": 30.0, "c": 5.0, "d": 5.0}, floor=3, caps=caps)
    assert sum(alloc.values()) == 50 and alloc["c"] == 4 and min(alloc.values()) >= 3
    assert alloc["a"] > alloc["b"] > alloc["d"]
    pool = {f"c{i:02d}": n for i, n in enumerate([6000, 4000, 2000] + [300] * 30 + [40] * 30 + [10] * 7)}
    a2000 = allocate_2000(pool)
    assert sum(a2000.values()) == 2000 and max(a2000.values()) == 120 and min(a2000.values()) == 5
    assert a2000["c00"] == 120 and a2000["c69"] == 5


def _synthetic_pool(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    per_l1 = [30, 16, 8, 6, 5, 2, 1, 1, 1]  # mimics DBPedia: three l1 classes with a single l2
    rows = []
    c = 0
    for li, n_l2 in enumerate(per_l1):
        for j in range(n_l2):
            size = int(rng.integers(8, 260))
            for k in range(size):
                text = f"Doc {c}-{k} is about class {c}. " + "x " * 20
                rows.append({"id": "dbp-" + hashlib.sha1(text.encode()).hexdigest()[:10], "title": f"Doc {c}-{k}",
                             "text": text, "l1": f"L1_{li}", "l2": f"L2_{c:02d}", "l3": f"L3_{c:02d}"})
            c += 1
    return rows


def test_sample_nested_properties():
    rows = _synthetic_pool()
    out = sample_nested(rows, seed=20260902, sizes=(100, 300, 1000, 2000), log=lambda *_: None)
    sets, orders, classes = out["sets"], out["orders"], out["classes"]
    for n in (100, 300, 1000, 2000):
        assert len(sets[n]) == n and len({r["id"] for r in sets[n]}) == n
    assert len(classes[100]) == 18 and len(classes[300]) == 40 and len(classes[1000]) == 70 == len(classes[2000])
    ids = {n: {r["id"] for r in sets[n]} for n in sets}
    assert ids[100] <= ids[300] <= ids[1000] <= ids[2000]
    counts = {n: collections.Counter(r["l2"] for r in sets[n]) for n in sets}
    for c in classes[100]:
        assert 3 <= counts[100][c] <= counts[300][c] <= counts[1000][c] <= counts[2000][c] <= 120
    assert all(len({r["l1"] for r in sets[n]}) == 9 for n in sets)
    # prefix-consistent order-1, order-2 = permutation of the first 100 then identical
    assert orders[300]["order-1"][:100] == orders[100]["order-1"]
    assert orders[2000]["order-1"][:1000] == orders[1000]["order-1"]
    assert set(orders[100]["order-2"]) == set(orders[100]["order-1"]) and orders[100]["order-2"] != orders[100]["order-1"]
    assert orders[300]["order-2"][100:] == orders[300]["order-1"][100:]
    assert set(orders[2000]["order-1"]) == ids[2000]


def test_select_classes_fill_rule():
    l1l2 = {"A": {"a1": 100, "a2": 90, "a3": 80, "a4": 70, "a5": 60, "a6": 50},
            "B": {"b1": 40, "b2": 30, "b3": 20}, "C": {"c1": 10}}
    assert select_classes(l1l2, 5, 2) == ["a1", "a2", "b1", "b2", "c1"]
    assert select_classes(l1l2, 7, 2) == ["a1", "a2", "a3", "a4", "b1", "b2", "c1"]
    assert select_classes(l1l2, 4, 2, restrict_to={"a1", "a2", "b1", "c1"}) == ["a1", "a2", "b1", "c1"]
