"""Thesis figures (matplotlib, print): F2 cost-vs-quality Pareto, F3 MMR lambda
ablation, F4 arm-B marginal cost vs wiki size, and the k-sensitivity curve.

Every figure reads a CSV written by ``report.write_tables`` (never a number
typed by hand) and is saved as a 300-dpi PNG 15 cm wide into ``figures/``.

Visual rules follow the ``dataviz`` skill: one series per arm with a fixed
hue (validated all-pairs with the skill's validator: blue ``#2a78d6`` = A,
hollow blue = A+, orange ``#eb6834`` = B, aqua ``#1baf7a`` = C; aqua is below
3:1 on the light surface, so every point carries a direct label), reference
lines in muted ink, hairline solid gridlines, 2 px lines, >= 8 px markers with
a surface ring, one y-axis per panel (two measures = two panels, never a dual
axis), legend whenever two or more series are drawn, text in ink tokens only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "figures"
TABLES_DIR = REPO_ROOT / "results" / "tables"
WIDTH_CM = 15.0
DPI = 300

SERIES = {"A": "#2a78d6", "A+": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
HOLLOW = {"A+"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb", "wash_alpha": 0.10}
ARM_ORDER = ("A", "A+", "B", "C")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK["axis"], "axes.linewidth": 0.6,
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "axes.labelcolor": INK["secondary"],
        "text.color": INK["primary"], "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"], "legend.frameon": False,
    })
    return plt


def _style(ax) -> None:
    ax.grid(True, color=INK["grid"], linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, width=0.5)


def _figsize(height_cm: float) -> tuple[float, float]:
    return WIDTH_CM / 2.54, height_cm / 2.54


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)  # exact 15 cm width (constrained layout, no tight bbox)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path


def _marker_kw(arm: str) -> dict[str, Any]:
    color = SERIES.get(arm, INK["muted"])
    if arm in HOLLOW:
        return {"marker": "o", "s": 36, "facecolors": INK["surface"], "edgecolors": color, "linewidths": 1.4, "zorder": 4}
    return {"marker": "o", "s": 36, "facecolors": color, "edgecolors": INK["surface"], "linewidths": 1.2, "zorder": 4}


def _legend_handles(arms: Sequence[str]):
    from matplotlib.lines import Line2D
    handles = []
    for arm in arms:
        color = SERIES.get(arm, INK["muted"])
        if arm in HOLLOW:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK["surface"], markeredgecolor=color,
                                  markeredgewidth=1.4, markersize=5, label=arm))
        else:
            handles.append(Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5, label=arm))
    return handles


def pareto_frontier(cost: Sequence[float], quality: Sequence[float]) -> list[int]:
    """Indices of the points not dominated (lower cost and higher quality)."""
    pts = sorted(range(len(cost)), key=lambda i: (cost[i], -quality[i]))
    best = -np.inf
    keep = []
    for i in pts:
        if quality[i] > best:
            keep.append(i)
            best = quality[i]
    return keep


def _read(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _first_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# --------------------------------------------------------------------------
# F2 cost vs quality
# --------------------------------------------------------------------------
def fig_cost_quality(csv_path: str | Path, out_path: str | Path, quality: str | None = None,
                     corpus: str | None = None, min_usd: float = 1e-4, title: str | None = None,
                     k_variants: Sequence[str] | None = None) -> Path:
    """Headline configurations only (ablation / k-grid variants are dropped); ``k_variants``
    restricts the clustering variants shown (e.g. ``("k_matched", "native")`` for the thesis
    headline); by default every variant is drawn and the point label names it when an arm
    has more than one."""
    plt = _plt()
    df = _read(csv_path)
    if corpus is not None and "corpus" in df:
        df = df[df["corpus"] == corpus]
    qcol = quality or _first_col(df, ["nmi_l2", "nmi_class", "nmi_l2_mean", "label_sim"])
    if qcol is None:
        raise ValueError("no quality column in " + str(csv_path))
    lo, hi = qcol + "_ci_low", qcol + "_ci_high"
    usd_col = _first_col(df, ["usd", "usd_mean"]) or "usd"
    if "variant" in df:  # headline configurations only: no ablation / k-grid variants
        df = df[df["variant"].isna() | (df["variant"].astype(str).str.strip() == "")]
    if k_variants is not None and "k_variant" in df:
        df = df[df["k_variant"].isna() | df["k_variant"].astype(str).isin(list(k_variants))]
    df = df[df[qcol].notna() & df[usd_col].notna()].copy()
    if df.empty:
        raise ValueError("nothing to plot in " + str(csv_path))
    df["x"] = df[usd_col].clip(lower=min_usd)
    multi = {a: df.loc[df["arm"] == a, "k_variant"].astype(str).nunique() > 1 for a in df["arm"].unique()} if "k_variant" in df else {}
    fig, ax = plt.subplots(figsize=_figsize(9.5), layout="constrained")
    _style(ax)
    ax.set_xscale("log")
    arms_present = [a for a in ARM_ORDER if a in set(df["arm"])] + sorted(set(df["arm"]) - set(ARM_ORDER))
    for arm in arms_present:
        sub = df[df["arm"] == arm]
        if lo in sub and hi in sub:
            ok = sub[lo].notna() & sub[hi].notna()
            if ok.any():
                ax.errorbar(sub.loc[ok, "x"], sub.loc[ok, qcol], yerr=[sub.loc[ok, qcol] - sub.loc[ok, lo], sub.loc[ok, hi] - sub.loc[ok, qcol]],
                            fmt="none", ecolor=SERIES.get(arm, INK["muted"]), elinewidth=0.8, capsize=2, capthick=0.8, zorder=3)
        ax.scatter(sub["x"], sub[qcol], **_marker_kw(arm), label=arm)
        for _, r in sub.iterrows():
            n = f"@{int(r['n_docs'])}" if "n_docs" in r and pd.notna(r["n_docs"]) else ""
            kv = f" ({r['k_variant']})" if multi.get(arm) and pd.notna(r.get("k_variant")) else ""
            ax.annotate(f"{arm}{n}{kv}", (r["x"], r[qcol]), xytext=(4, 4), textcoords="offset points", fontsize=6.5,
                        color=INK["secondary"], zorder=5)
    idx = pareto_frontier(df["x"].tolist(), df[qcol].tolist())
    if len(idx) > 1:
        fr = df.iloc[idx].sort_values("x")
        ax.step(fr["x"], fr[qcol], where="post", color=INK["muted"], linewidth=1.0, zorder=2, label="Pareto frontier")
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    if xmax / xmin < 10:  # keep a readable decade span on the log axis (single point or near-identical costs)
        ax.set_xlim(xmin / 10 ** 0.75, xmax * 10 ** 0.75)
    ys = pd.concat([df[qcol], df[lo] if lo in df else df[qcol], df[hi] if hi in df else df[qcol]])
    pad = max(0.02, 0.15 * float(ys.max() - ys.min()))
    ax.set_ylim(max(0.0, float(ys.min()) - pad), min(1.0, float(ys.max()) + pad))
    ax.set_xlabel("Cost per run, USD (LLM arms: list price with caching; arm A: cloud-equivalent; log scale)")
    ax.set_ylabel({"nmi_l2": "NMI vs gold l2", "label_sim": "Label similarity", "nmi_class": "NMI vs class"}.get(qcol, qcol))
    ax.set_title(title or "Cost vs quality per arm (95% CI whiskers)", loc="left", color=INK["primary"])
    if len(arms_present) >= 2:  # a single series needs no legend box (the title names it)
        ax.legend(handles=_legend_handles(arms_present) + ([plt.Line2D([0], [0], color=INK["muted"], linewidth=1.0, label="Pareto frontier")] if len(idx) > 1 else []),
                  loc="best", handletextpad=0.4)
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# F3 MMR lambda ablation
# --------------------------------------------------------------------------
def fig_mmr_ablation(csv_path: str | Path, out_path: str | Path, corpus: str | None = None,
                     k_variant: str | None = None) -> Path:
    """One curve per N within a single clustering variant (``k_matched`` when present, else
    ``oracle``, else the first available) -- never averaged across variants."""
    plt = _plt()
    df = _read(csv_path)
    if corpus is not None and "corpus" in df:
        df = df[df["corpus"] == corpus]
    if "k_variant" in df.columns and df["k_variant"].notna().any():
        present = [v for v in df["k_variant"].dropna().astype(str).unique()]
        chosen = k_variant or next((v for v in ("k_matched", "oracle") if v in present), present[0])
        df = df[df["k_variant"].astype(str) == chosen]
    else:
        chosen = None
    lam = "mmr_lambda"
    panels = [(c, lab) for c, lab in (("same_l2_p_at_5", "Same-l2 precision@5"), ("hyperlink_p_at_5", "Hyperlink precision@5"),
                                       ("ild", "Intra-list diversity (1 - cos)"), ("unique_category_frac", "Unique-category fraction"))
              if c in df.columns and df[c].notna().any()]
    if not panels:
        raise ValueError("no ablation columns in " + str(csv_path))
    fig, axes = plt.subplots(1, len(panels), figsize=_figsize(6.5), sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    groups = sorted(df["n_docs"].dropna().unique()) if "n_docs" in df else [None]
    for ax, (col, label) in zip(axes, panels):
        _style(ax)
        for gi, n in enumerate(groups):
            sub = df if n is None else df[df["n_docs"] == n]
            sub = sub.sort_values(lam)
            color = SERIES["A"]
            alpha = 1.0 if gi == len(groups) - 1 else 0.45
            ax.plot(sub[lam], sub[col], color=color, linewidth=2, alpha=alpha, solid_joinstyle="round", zorder=3,
                    label=(f"N={int(n)}" if n is not None else "A"))
            ax.scatter(sub[lam], sub[col], s=30, color=color, edgecolors=INK["surface"], linewidths=1.2, alpha=alpha, zorder=4)
            lo, hi = col + "_ci_low", col + "_ci_high"
            if lo in sub and hi in sub and sub[lo].notna().any():
                ax.fill_between(sub[lam], sub[lo], sub[hi], color=color, alpha=INK["wash_alpha"], linewidth=0, zorder=2)
        ax.set_xlabel("MMR lambda (1.0 = pure kNN)")
        ax.set_title(label, loc="left")
    axes[0].invert_xaxis()  # shared axis: invert once (1.0 = pure kNN on the left)
    if len(groups) > 1:
        axes[0].legend(loc="best")
    fig.suptitle("Arm A related-link ablation over the MMR trade-off" + (f" ({chosen} clustering)" if chosen else ""),
                 x=0.01, ha="left", fontsize=9, color=INK["primary"])
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# F4 marginal cost per ingest vs wiki size (arm B)
# --------------------------------------------------------------------------
def fig_cost_per_ingest(csv_path: str | Path, out_path: str | Path, replicate: Any | None = None) -> Path:
    plt = _plt()
    df = _read(csv_path)
    if replicate is not None and "replicate" in df:
        df = df[df["replicate"].astype(str) == str(replicate)]
    df = df[df["usd"].notna() & df["wiki_pages_before"].notna()]
    fig, ax = plt.subplots(figsize=_figsize(8.5), layout="constrained")
    _style(ax)
    color = SERIES["B"]
    x, y = df["wiki_pages_before"].to_numpy(float), df["usd"].to_numpy(float)
    ax.scatter(x, y, s=18, color=color, edgecolors=INK["surface"], linewidths=0.8, alpha=0.9, zorder=3)
    slope = df["fit_slope"].dropna().iloc[0] if "fit_slope" in df and df["fit_slope"].notna().any() else None
    if slope is not None:
        intercept = df["fit_intercept"].dropna().iloc[0]
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color=INK["secondary"], linewidth=2, zorder=4)
        lo = df["fit_slope_ci_low"].dropna().iloc[0] if "fit_slope_ci_low" in df and df["fit_slope_ci_low"].notna().any() else None
        hi = df["fit_slope_ci_high"].dropna().iloc[0] if "fit_slope_ci_high" in df and df["fit_slope_ci_high"].notna().any() else None
        if lo is not None and hi is not None:
            xm, ym = x.mean(), y.mean()
            band_lo = np.minimum(ym + lo * (xs - xm), ym + hi * (xs - xm))
            band_hi = np.maximum(ym + lo * (xs - xm), ym + hi * (xs - xm))
            ax.fill_between(xs, band_lo, band_hi, color=color, alpha=INK["wash_alpha"], linewidth=0, zorder=2)
            txt = f"OLS slope: {slope * 100:+.3f} USD per 100 pages\n95% CI [{lo * 100:+.3f}, {hi * 100:+.3f}]"
        else:
            txt = f"OLS slope: {slope * 100:+.3f} USD per 100 pages"
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK["secondary"], zorder=5)
    ax.set_xlabel("Wiki pages before the ingest")
    ax.set_ylabel("Cost of the ingest, USD (list price)")
    ax.set_ylim(bottom=0)
    ax.set_title("Arm B: marginal cost of one ingest vs wiki size", loc="left", color=INK["primary"])
    return _save(fig, out_path)


# --------------------------------------------------------------------------
# k-sensitivity
# --------------------------------------------------------------------------
def fig_k_sensitivity(csv_path: str | Path, out_path: str | Path, reference_k: Mapping[str, float] | None = None,
                      corpus: str | None = None, n_docs: int | None = None) -> Path:
    plt = _plt()
    df = _read(csv_path)
    if corpus is not None and "corpus" in df:
        df = df[df["corpus"] == corpus]
    if n_docs is not None and "n_docs" in df:
        df = df[df["n_docs"] == n_docs]
    metrics = [(c, lab) for c, lab in (("nmi_l2", "NMI vs gold l2"), ("p1_l2", "P1 vs gold l2"), ("ari_l2", "ARI vs gold l2"))
               if c in df.columns and df[c].notna().any()]
    if not metrics or "k" not in df.columns:
        raise ValueError("no k-sensitivity columns (k + metric) in " + str(csv_path))
    metrics = metrics[:2]
    fig, axes = plt.subplots(1, len(metrics), figsize=_figsize(6.5), sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    df = df.groupby("k", as_index=False).mean(numeric_only=True).sort_values("k") if "k_variant" in df.columns else df.sort_values("k")
    for ax, (col, label) in zip(axes, metrics):
        _style(ax)
        ax.plot(df["k"], df[col], color=SERIES["A"], linewidth=2, solid_joinstyle="round", zorder=3)
        ax.scatter(df["k"], df[col], s=26, color=SERIES["A"], edgecolors=INK["surface"], linewidths=1.2, zorder=4)
        sd = col + "_sd"
        if sd in df and df[sd].notna().any():
            ax.fill_between(df["k"], df[col] - df[sd], df[col] + df[sd], color=SERIES["A"], alpha=INK["wash_alpha"], linewidth=0, zorder=2)
        by_k: dict[float, list[str]] = {}
        for name, kv in (reference_k or {}).items():
            if kv is None or not np.isfinite(kv):
                continue
            by_k.setdefault(float(kv), []).append(name)
        for kv, names in by_k.items():
            ax.axvline(kv, color=INK["muted"], linewidth=0.8, zorder=1)
            ax.annotate(", ".join(names), xy=(kv, ax.get_ylim()[1]), xytext=(3, -2), textcoords="offset points",
                        fontsize=6.5, color=INK["secondary"], rotation=90, va="top", ha="left")
        ax.set_xlabel("k (number of clusters)")
        ax.set_title(label, loc="left")
    fig.suptitle("Arm A sensitivity to k (mean over seeds; band = +/- 1 sd)", x=0.01, ha="left", fontsize=9, color=INK["primary"])
    return _save(fig, out_path)


def make_all(tables_dir: str | Path = TABLES_DIR, figures_dir: str | Path = FIGURES_DIR,
             reference_k: Mapping[str, float] | None = None) -> dict[str, Path]:
    """Render every figure whose source CSV exists."""
    tables_dir, figures_dir = Path(tables_dir), Path(figures_dir)
    out: dict[str, Path] = {}
    jobs = (("F2_cost_quality", "cost_quality.csv", lambda p, o: fig_cost_quality(p, o)),
            ("F3_ablation_mmr", "ablation_mmr.csv", lambda p, o: fig_mmr_ablation(p, o)),
            ("F4_cost_per_ingest", "cost_per_ingest.csv", lambda p, o: fig_cost_per_ingest(p, o)),
            ("k_sensitivity", "k_sensitivity.csv", lambda p, o: fig_k_sensitivity(p, o, reference_k=reference_k)))
    for name, csv, fn in jobs:
        src = tables_dir / csv
        if src.exists():
            try:
                out[name] = fn(src, figures_dir / f"{name}.png")
            except ValueError:
                continue
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``wikiorg figures [--tables DIR] [--out DIR] [--reference-k 'k*=40,k_B=38']``."""
    import argparse

    p = argparse.ArgumentParser(prog="wikiorg figures", description="render F2-F4 and the k-sensitivity curve from results/tables")
    p.add_argument("--tables", default=str(TABLES_DIR))
    p.add_argument("--out", default=str(FIGURES_DIR))
    p.add_argument("--reference-k", default="", help="comma-separated name=k pairs drawn as reference lines")
    argv = list(argv or [])
    while argv and argv[0] == "--":
        argv.pop(0)
    args = p.parse_args(argv)
    ref: dict[str, float] = {}
    for item in [x for x in args.reference_k.split(",") if x.strip()]:
        name, _, val = item.partition("=")
        try:
            ref[name.strip()] = float(val)
        except ValueError:
            continue
    out = make_all(args.tables, args.out, reference_k=ref or None)
    for name, path in out.items():
        print(f"{name}: {path}")
    if not out:
        print(f"no source CSVs found in {args.tables}")
    return 0
