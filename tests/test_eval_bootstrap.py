"""Bootstrap, subsampling, effect sizes, Holm and the pre-registered decision."""
import numpy as np
import pytest

from wikiorg.eval.bootstrap import (bca_bound, cliffs_delta, cohens_d, cohens_dz, decide, effect_sizes, holm,
                                    holm_non_inferiority, mean_ci, non_inferiority, non_inferiority_partition,
                                    paired_mean_diff_ci, prereg_thresholds, pvalue_greater, subsample_diff_ci,
                                    subsample_metric_ci)

FLOORS_OK = {"nmi_vs_l2_at_oracle_k": 0.70, "p1_at_oracle_k": 0.75}


def test_bootstrap_reproducible_with_fixed_seed():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.5, 0.2, 120), rng.normal(0.45, 0.2, 120)
    r1 = paired_mean_diff_ci(a, b, n_resamples=500, seed=42)
    r2 = paired_mean_diff_ci(a, b, n_resamples=500, seed=42)
    assert (r1.low, r1.high) == (r2.low, r2.high) and np.array_equal(r1.dist, r2.dist)
    r3 = paired_mean_diff_ci(a, b, n_resamples=500, seed=7)
    assert (r3.low, r3.high) != (r1.low, r1.high)
    assert r1.method == "BCa" and r1.low < r1.estimate < r1.high
    s1 = subsample_diff_ci([1, 1, 2, 2, 3, 3] * 20, [1, 2, 1, 2, 3, 3] * 20, [1, 1, 2, 2, 3, 3] * 20, n_resamples=100, seed=3)
    s2 = subsample_diff_ci([1, 1, 2, 2, 3, 3] * 20, [1, 2, 1, 2, 3, 3] * 20, [1, 1, 2, 2, 3, 3] * 20, n_resamples=100, seed=3)
    assert np.array_equal(s1.dist, s2.dist) and s1.extra["subsample_size"] == 96


def test_bca_inversion_matches_scipy_bound():
    rng = np.random.default_rng(1)
    a, b = rng.beta(2, 3, 200), rng.beta(2, 3, 200) - 0.03
    r = paired_mean_diff_ci(a, b, n_resamples=2000, seed=0)
    assert pvalue_greater(r, r.low) == pytest.approx(0.025, abs=0.006)
    assert bca_bound(r, 0.025, "lower") == pytest.approx(r.low, abs=2e-3)
    assert pvalue_greater(r, r.high) == pytest.approx(0.975, abs=0.006)
    assert pvalue_greater(r, -10.0) == 0.0 and pvalue_greater(r, 10.0) == 1.0


def test_non_inferiority_known_answers():
    rng = np.random.default_rng(2)
    b = rng.uniform(0.3, 0.9, 300)
    equal = non_inferiority(b + rng.normal(0, 0.05, 300), b, 0.10, "eq", n_resamples=1000)
    assert equal["non_inferior"] and not equal["falsified"] and equal["p_one_sided"] < 0.001
    worse = non_inferiority(b - 0.30 + rng.normal(0, 0.05, 300), b, 0.10, "worse", n_resamples=1000)
    assert worse["falsified"] and not worse["non_inferior"] and worse["p_one_sided"] > 0.99
    border = non_inferiority(b[:100] - 0.10 + rng.normal(0, 0.3, 100), b[:100], 0.10, "border", n_resamples=1000)
    assert not border["non_inferior"] and not border["falsified"]
    assert equal["n_paired"] == 300 and "cohen_dz" in equal["effect"]
    assert equal["ci_reading"].startswith("lower bound of the two-sided 95% CI")
    one_sided = non_inferiority(b, b - 0.02, 0.10, "os", n_resamples=500, alternative="greater")
    assert one_sided["ci_high"] == np.inf and one_sided["non_inferior"]


def test_degenerate_cases():
    const = non_inferiority(np.ones(50), np.ones(50), 0.1, "c")
    assert const["method"] == "constant" and const["non_inferior"] and const["p_one_sided"] == 0.0
    r = mean_ci(np.r_[np.ones(49), 0.0], n_resamples=300, seed=0)
    assert r.low <= r.estimate <= r.high


def test_partition_subsampling():
    rng = np.random.default_rng(3)
    gold = [i % 5 for i in range(300)]
    a = [g if rng.random() > 0.05 else int(rng.integers(5)) for g in gold]
    b = [g if rng.random() > 0.30 else int(rng.integers(5)) for g in gold]
    res = non_inferiority_partition(a, b, gold, 0.10, metric="nmi", n_resamples=300, seed=0)
    assert res["method"].startswith("subsampling") and res["subsample_size"] == 240
    assert res["estimate"] == pytest.approx(res["estimate_a"] - res["estimate_b"])
    assert res["non_inferior"] and res["ci_low"] > 0 and res["ci_low_raw"] < res["ci_high_raw"]
    single = subsample_metric_ci(a, gold, "p1", n_resamples=200, seed=0)
    assert 0 < single.low <= single.estimate <= single.high <= 1.0 + 1e-9
    same = subsample_diff_ci(a, a, gold, metric="ari", n_resamples=50)
    assert same.estimate == 0.0 and same.degenerate


def test_effect_sizes_and_holm():
    assert cliffs_delta([1, 2, 3], [0, 1, 2]) == pytest.approx(5 / 9)
    assert cliffs_delta([1, 1], [1, 1]) == 0.0
    assert cohens_d([0, 0, 1, 1], [1, 1, 2, 2]) == pytest.approx(-1 / np.sqrt(1 / 3))
    assert cohens_dz([1, 2, 3], [0, 1, 2]) == np.inf
    es = effect_sizes([1, 2, 3, 4], [0, 1, 2, 3])
    assert es["mean_diff"] == 1.0 and es["cliffs_delta"] == pytest.approx(7 / 16)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.2, 0.001]) == pytest.approx([0.2, 0.002])
    assert holm([]) == []


def _endpoints(rng, shift_e3=0.0):
    b = rng.uniform(0.3, 0.9, 300)
    gold = [i % 6 for i in range(300)]
    a_part = [g if rng.random() > 0.05 else int(rng.integers(6)) for g in gold]
    b_part = [g if rng.random() > 0.15 else int(rng.integers(6)) for g in gold]
    return {"E1": non_inferiority_partition(a_part, b_part, gold, 0.10, n_resamples=200, seed=0),
            "E2": non_inferiority(b + rng.normal(0, 0.05, 300), b, 0.10, "label_sim", n_resamples=300, seed=0),
            "E3": non_inferiority(b + shift_e3 + rng.normal(0, 0.05, 300), b, 0.10, "p_at_5", n_resamples=300, seed=0)}


def test_decide_rule():
    th = prereg_thresholds()
    assert th["deltas"] == {"E1": 0.1, "E2": 0.1, "E3": 0.1} and th["cost_ratio_required"] == 20
    assert th["floors"] == {"nmi_vs_l2_at_oracle_k": 0.55, "p1_at_oracle_k": 0.6} and th["alpha_one_sided"] == 0.025
    rng = np.random.default_rng(4)
    ok = decide(_endpoints(rng), cost_ratio=120, floors=FLOORS_OK)
    assert ok["verdict"] == "supported" and ok["all_non_inferior_holm"] and ok["cost"]["met"] and ok["floors_met"]
    assert all(e["non_inferior_holm"] for e in ok["endpoints"].values())
    assert decide(_endpoints(rng), cost_ratio=5, floors=FLOORS_OK)["verdict"] == "inconclusive"
    assert decide(_endpoints(rng), cost_ratio=120, floors={"nmi_vs_l2_at_oracle_k": 0.50, "p1_at_oracle_k": 0.75})["verdict"] == "falsified"
    bad = decide(_endpoints(rng, shift_e3=-0.35), cost_ratio=120, floors=FLOORS_OK)
    assert bad["verdict"] == "falsified" and bad["endpoints"]["E3"]["falsified"]
    partial = decide({k: v for k, v in _endpoints(rng).items() if k != "E3"}, cost_ratio=120, floors=FLOORS_OK)
    assert partial["verdict"] == "inconclusive" and partial["missing_endpoints"] == ["E3"]
    wrong = _endpoints(rng)
    wrong["E1"]["delta"] = 0.2
    with pytest.raises(ValueError):
        decide(wrong, cost_ratio=120, floors=FLOORS_OK)
    adj = holm_non_inferiority(_endpoints(rng), alpha=0.025)
    assert [e["holm_level"] for e in sorted(adj.values(), key=lambda e: e["p_one_sided"])] == pytest.approx([0.025 / 3, 0.025 / 2, 0.025])
    assert all(e["ci_low_holm"] <= e["ci_low"] + 1e-9 for e in adj.values())
