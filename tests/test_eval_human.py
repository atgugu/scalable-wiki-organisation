"""Human rating study: sheet construction, blinding, scoring, judge agreement."""
import csv

import numpy as np
import pytest

from wikiorg.eval.human import (ARMS, CONTROL_ARM, build_items, human_table_rows, read_key, read_ratings, score,
                                score_with_and_without, select_classes, write_study)
from wikiorg.eval.judge import judge_agreement, read_judge_csv
from wikiorg.eval.state import Category, Page, WikiState, gold_from_rows

CLASSES = ["Athlete", "Politician", "River", "Company", "Plant", "Album", "Film", "Village", "Animal", "Software", "Artist", "Bridge"]


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    rows = [{"id": f"d{i}", "title": f"Doc {i}", "l1": "x", "l2": CLASSES[i % 12]} for i in range(240)]
    gold = gold_from_rows(rows)

    def mk(arm, noise):
        pages = {}
        for i in range(240):
            c = i % 12 if rng.random() > noise else int(rng.integers(12))
            pages[f"d{i}"] = Page(title=f"Doc {i}", category=f"c{c}")
        return WikiState(arm, pages, {f"c{j}": Category(label=f"{arm} label {j}", terms=[f"t{j}"]) for j in range(12)})

    states = {"A": mk("A", 0.1), "A+": mk("A+", 0.1), "B": mk("B", 0.2), "C": mk("C", 0.2)}
    return gold, states


def test_build_items_and_blinding(setup):
    gold, states = setup
    assert len(select_classes(gold, "l2", 10)) == 10
    items, codes = build_items(states, gold, n_classes=10, n_controls=5, seed=1)
    assert len(items) == 45
    assert sum(1 for it in items if it["arm"] == CONTROL_ARM) == 5
    for arm in ARMS:
        assert sum(1 for it in items if it["arm"] == arm) == 10
    assert len({it["item_id"] for it in items}) == 45
    assert len(set(codes.values())) == 5 and set(codes) == set(ARMS) | {CONTROL_ARM}
    assert all(it["label"] and it["title_1"] and it["title_2"] and it["title_3"] for it in items)
    controls = [it for it in items if it["arm"] == CONTROL_ARM]
    assert all(it["label"] == it["gold_class"].lower() for it in controls)  # de-CamelCased gold name
    assert all(it["majority_matches"] for it in items)
    items2, _ = build_items(states, gold, n_classes=10, n_controls=5, seed=1)
    assert [it["item_id"] for it in items2] == [it["item_id"] for it in items]  # deterministic


def test_write_read_score(setup, tmp_path):
    gold, states = setup
    items, codes = build_items(states, gold, n_classes=10, n_controls=5, seed=2)
    paths = write_study(items, codes, ["student", "peer1", "peer2"], tmp_path, seed=2)
    sheet = list(csv.DictReader(open(paths["sheet:student"])))
    assert len(sheet) == 45 and all(r["rating"] == "" for r in sheet)
    assert "arm" not in sheet[0] and "gold_class" not in sheet[0]  # blinded
    order_s = [r["item_id"] for r in sheet]
    order_p = [r["item_id"] for r in csv.DictReader(open(paths["sheet:peer1"]))]
    assert set(order_s) == set(order_p) and order_s != order_p  # per-rater randomised order
    assert "1:" in paths["instructions"].read_text()
    key = read_key(paths["key"])
    assert len(key) == 45 and key[items[0]["item_id"]]["arm"] == items[0]["arm"]
    rng = np.random.default_rng(3)
    base = {"A": 3.0, "A+": 4.0, "B": 4.2, "C": 3.6, CONTROL_ARM: 4.8}
    done = []
    for rater in ("student", "peer1", "peer2"):
        rows = list(csv.DictReader(open(paths[f"sheet:{rater}"])))
        for r in rows:
            r["rating"] = int(np.clip(round(base[key[r["item_id"]]["arm"]] + rng.normal(0, 0.6)), 1, 5))
        rows[0]["rating"] = ""  # one blank rating survives reading
        out = tmp_path / f"{rater}_done.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        done.append(out)
    ratings = read_ratings(done)
    assert len(ratings) == 3 * 45 - 3
    res = score_with_and_without(ratings, key, "student", n_resamples=200, seed=0)
    allr, wo = res["all_raters"], res["without_student"]
    assert allr["n_raters"] == 3 and wo["n_raters"] == 2 and wo["excluded_raters"] == ["student"]
    assert set(allr["arms"]) == {"A", "A+", "B", "C", CONTROL_ARM}
    assert allr["arms"][CONTROL_ARM]["mean"] > allr["arms"]["A"]["mean"]
    for v in allr["arms"].values():
        assert v["ci_low"] <= v["mean"] <= v["ci_high"] and v["n_items"] in (5, 10)
    assert -1 <= allr["krippendorff_alpha_interval"] <= 1 and len(allr["spearman"]) == 3
    assert len(allr["cliffs_delta"]) == 10
    rows = human_table_rows(res)
    assert len(rows) == 10 and {r["variant"] for r in rows} == {"all_raters", "without_student"}
    # judge agreement: a judge equal to the human mean correlates perfectly
    judge_csv = tmp_path / "judge.csv"
    with open(judge_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "rating", "model"])
        for it, val in allr["item_means"].items():
            w.writerow([it, val, "sonnet"])
    judge = read_judge_csv(judge_csv)
    ja = judge_agreement(judge, ratings, key)
    assert ja["n_common_items"] == 45 and ja["spearman_judge_vs_human_mean"] == pytest.approx(1.0)
    assert ja["mean_abs_diff"] == pytest.approx(0.0) and ja["alpha_judge_vs_human_consensus"] == pytest.approx(1.0)
    assert set(ja["arms"]) == {"A", "A+", "B", "C", CONTROL_ARM}


def test_score_handles_single_rater(setup):
    gold, states = setup
    items, codes = build_items(states, gold, n_classes=3, n_controls=1, seed=0)
    key = {it["item_id"]: it for it in items}
    ratings = [{"rater": "solo", "item_id": it["item_id"], "rating": 3.0} for it in items]
    res = score(ratings, key, n_resamples=50)
    assert res["n_raters"] == 1 and np.isnan(res["krippendorff_alpha_interval"]) and res["spearman"] == []
