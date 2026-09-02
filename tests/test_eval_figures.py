"""Figures render from CSVs at 300 dpi and 15 cm width."""
import numpy as np
import pandas as pd
import pytest

from wikiorg.eval import figures
from wikiorg.eval.figures import DPI, WIDTH_CM, pareto_frontier

EXPECTED_W = int(WIDTH_CM / 2.54 * DPI)  # 1771 px


def _png_shape(path):
    import matplotlib.image as mpimg
    return mpimg.imread(path).shape


def test_pareto_frontier():
    cost = [1, 2, 3, 4, 0.5]
    qual = [0.5, 0.7, 0.6, 0.9, 0.4]
    assert pareto_frontier(cost, qual) == [4, 0, 1, 3]


def test_all_figures_render(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame({"arm": ["A", "A+", "B", "C"], "corpus": "dbpedia", "n_docs": 300, "k_variant": ["k_matched", "k_matched", "native", "native"],
                  "variant": ["", "", "", ""], "usd": [0.012, 0.8, 20.0, 5.0], "nmi_l2": [0.72, 0.72, 0.74, 0.70],
                  "nmi_l2_ci_low": [0.69, 0.69, 0.70, 0.66], "nmi_l2_ci_high": [0.75, 0.75, 0.78, 0.74]}).to_csv(tables / "cost_quality.csv", index=False)
    pd.DataFrame({"arm": "A", "corpus": "dbpedia", "n_docs": [300, 300, 300, 1000, 1000, 1000], "mmr_lambda": [1.0, 0.7, 0.5] * 2,
                  "same_l2_p_at_5": [0.80, 0.78, 0.74, 0.82, 0.80, 0.76], "same_l2_p_at_5_ci_low": [0.77, 0.75, 0.71, 0.80, 0.78, 0.74],
                  "same_l2_p_at_5_ci_high": [0.83, 0.81, 0.77, 0.84, 0.82, 0.78], "ild": [0.30, 0.36, 0.42, 0.31, 0.37, 0.43],
                  "unique_category_frac": [0.3, 0.35, 0.4, 0.3, 0.35, 0.4], "hyperlink_p_at_5": [np.nan] * 6}).to_csv(tables / "ablation_mmr.csv", index=False)
    rng = np.random.default_rng(0)
    x = np.arange(300) * 4
    pd.DataFrame({"arm": "B", "corpus": "dbpedia", "n_docs": 300, "replicate": "order-1", "ingest_index": np.arange(300), "wiki_pages_before": x,
                  "usd": 0.03 + 2e-5 * x + rng.normal(0, 0.004, 300), "fit_slope": 2e-5, "fit_intercept": 0.03, "fit_slope_ci_low": 1.5e-5,
                  "fit_slope_ci_high": 2.5e-5}).to_csv(tables / "cost_per_ingest.csv", index=False)
    pd.DataFrame({"arm": "A", "corpus": "dbpedia", "n_docs": 300, "k": [10, 20, 40, 60, 80], "nmi_l2": [0.6, 0.68, 0.72, 0.71, 0.69],
                  "nmi_l2_sd": [0.01] * 5, "p1_l2": [0.55, 0.62, 0.66, 0.65, 0.63], "p1_l2_sd": [0.01] * 5}).to_csv(tables / "k_sensitivity.csv", index=False)
    out = figures.make_all(tables, tmp_path / "figs", reference_k={"k*": 40, "k_B": 40, "k_C": 52})
    assert set(out) == {"F2_cost_quality", "F3_ablation_mmr", "F4_cost_per_ingest", "k_sensitivity"}
    for path in out.values():
        h, w, _ = _png_shape(path)
        assert abs(w - EXPECTED_W) <= 1 and h > 300, (path, w, h)
    # F2 ignores ablation/k-grid variants and still renders with a corpus filter
    p = figures.fig_cost_quality(tables / "cost_quality.csv", tmp_path / "f2b.png", corpus="dbpedia", quality="nmi_l2")
    assert p.exists()
    with pytest.raises(ValueError):
        figures.fig_k_sensitivity(tables / "cost_quality.csv", tmp_path / "bad.png")
    assert figures.make_all(tmp_path / "nowhere", tmp_path / "figs2") == {}
