"""The experiment ledger and the thesis tables.

``results/ledger.jsonl`` is **append-only**: one JSON line per evaluated run,
unique per ``(arm, corpus, n_docs, replicate, k_variant, variant, config_hash)`` where
``replicate`` is the seed (arm A), the ingest order (arm B) or the run index
(arm C).  Every row carries the git hash (and dirty flag), the config hash,
the LLM model id, CLI version, ``claude_md_sha256``, package versions, seeds,
the price-table date, an ``eval_code_hash`` of this package, all metrics and
the cost block.  Rows of other kinds (``stability``, ``human``, ``judge``,
``decision``, ``compare``) live in the same file, so that *every number the
thesis prints comes from the ledger*.

A second row with the same key and identical content is a no-op; one with
different content is refused unless ``allow_supersede=True``, in which case
it is appended with ``supersedes = <row_id>`` and readers keep the latest per
key -- the file itself is never rewritten.

``write_tables`` renders ``results/tables/*.csv`` (T1-T6 of Appendix D and the
figure sources F2-F4 + k-sensitivity) from ledger rows only.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .structure import kamps_koolen_row

REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"
TABLES_DIR = REPO_ROOT / "results" / "tables"
KEY_FIELDS = ("kind", "arm", "corpus", "n_docs", "replicate", "k_variant", "variant", "config_hash")
GROUP_FIELDS = ["arm", "corpus", "n_docs", "k_variant", "variant"]
PACKAGES = ("numpy", "scipy", "sklearn", "pandas", "networkx", "torch", "sentence_transformers", "krippendorff",
            "matplotlib", "wikiorg")


class LedgerConflict(RuntimeError):
    pass


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------
def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    return sha256_bytes(p.read_bytes()) if p.exists() else None


def config_hash(config: Mapping[str, Any] | str | Path | None) -> str | None:
    if config is None:
        return None
    if isinstance(config, (str, Path)):
        return file_sha256(config)
    return sha256_bytes(_canonical(dict(config)).encode())


def git_info(repo: str | Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_hash": head, "git_dirty": (bool(status.strip()) if status is not None else None),
            "git_tag": run("describe", "--tags", "--exact-match") if head else None}


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    from importlib import metadata
    out: dict[str, str | None] = {}
    for n in names:
        dist = {"sklearn": "scikit-learn", "sentence_transformers": "sentence-transformers"}.get(n, n)
        try:
            out[n] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            try:
                out[n] = getattr(__import__(n), "__version__", None)
            except Exception:
                out[n] = None
    return out


def eval_code_hash() -> str:
    """sha256 over the sources of ``wikiorg.eval`` and ``wikiorg.adapters`` (analysis-code identity)."""
    h = hashlib.sha256()
    base = Path(__file__).resolve().parent.parent
    for sub in ("eval", "adapters"):
        for f in sorted((base / sub).glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def gpu_name() -> str | None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------
def make_row(*, kind: str = "run", arm: str, corpus: str, n_docs: int, replicate: Any, config: Mapping[str, Any] | str | Path | None = None,
             metrics: Mapping[str, Any] | None = None, cost: Mapping[str, Any] | None = None, variant: str = "",
             k_variant: str = "", k: int | None = None, seeds: Mapping[str, Any] | None = None, model_id: str | None = None,
             cli_version: str | None = None, claude_md_sha256: str | None = None, effort: str | None = None,
             corpus_manifest_sha256: str | None = None, price_table_date: str | None = None, run_dir: str | Path | None = None,
             extra: Mapping[str, Any] | None = None, repo: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Assemble a ledger row with full provenance."""
    cfg_hash = config_hash(config)
    row: dict[str, Any] = {
        "kind": kind, "arm": arm, "corpus": corpus, "n_docs": int(n_docs), "replicate": replicate,
        "variant": variant or "", "k_variant": k_variant or "", "k": k, "config_hash": cfg_hash,
        "config": dict(config) if isinstance(config, Mapping) else (str(config) if config else None),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id, "cli_version": cli_version, "claude_md_sha256": claude_md_sha256, "effort": effort,
        "corpus_manifest_sha256": corpus_manifest_sha256, "price_table_date": price_table_date,
        "seeds": dict(seeds or {}), "run_dir": str(run_dir) if run_dir else None,
        "python": sys.version.split()[0], "platform": platform.platform(), "host": socket.gethostname(),
        "gpu": gpu_name(), "package_versions": package_versions(), "eval_code_hash": eval_code_hash(),
        "metrics": json.loads(_canonical(metrics or {})), "cost": json.loads(_canonical(cost or {})),
    }
    row.update(git_info(repo))
    if extra:
        row["extra"] = json.loads(_canonical(dict(extra)))
    row["row_id"] = sha256_bytes(_canonical({k: row.get(k) for k in KEY_FIELDS} | {"t": row["timestamp"], "m": row["metrics"], "c": row["cost"]}).encode())[:16]
    return row


def row_key(row: Mapping[str, Any]) -> tuple:
    return tuple(_canonical(row.get(k)) for k in KEY_FIELDS)


def _content(row: Mapping[str, Any]) -> str:
    return _canonical({k: row.get(k) for k in ("metrics", "cost", "extra", "k", "k_variant", "seeds")})


def read_ledger(path: str | Path = LEDGER_PATH, latest_only: bool = True, kind: str | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if latest_only:
        latest: dict[tuple, dict[str, Any]] = {}
        for r in rows:  # later lines supersede earlier ones with the same key
            latest[row_key(r)] = r
        rows = list(latest.values())
    if kind is not None:
        rows = [r for r in rows if r.get("kind") == kind]
    return rows


def _append_ledger_unlocked(row: Mapping[str, Any], path: str | Path = LEDGER_PATH, allow_supersede: bool = False) -> dict[str, Any]:
    """Append one row; refuse to change history.

    Returns ``{"status": "appended" | "duplicate" | "superseded", "row_id": ...}``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    key = row_key(row)
    existing = read_ledger(p, latest_only=True)
    prev = next((r for r in existing if row_key(r) == key), None)
    if prev is not None:
        if _content(prev) == _content(row):
            return {"status": "duplicate", "row_id": prev.get("row_id"), "path": str(p)}
        if not allow_supersede:
            raise LedgerConflict(f"ledger already has a different row for key {dict(zip(KEY_FIELDS, key))}; "
                                 f"pass allow_supersede=True to append a superseding row")
        row["supersedes"] = prev.get("row_id")
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(_canonical(row) + "\n")
        fh.flush()
    return {"status": "superseded" if prev is not None else "appended", "row_id": row.get("row_id"), "path": str(p)}



def append_ledger(row: Mapping[str, Any], path: str | Path = LEDGER_PATH, allow_supersede: bool = False) -> dict[str, Any]:
    """Append one ledger row under an exclusive inter-process lock.

    Ledger rows run to ~12 KB, well past what a single write(2) is guaranteed to append atomically,
    and the uniqueness/supersede check is a read-then-append. Two evaluations appending concurrently
    (e.g. DBPedia and WikiCS in parallel) could therefore interleave a line or both pass the check.
    A sidecar ``<ledger>.lock`` held with ``fcntl.flock`` serialises the whole critical section."""
    import fcntl
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_name(p.name + ".lock")
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _append_ledger_unlocked(row, p, allow_supersede)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def verify_ledger(path: str | Path = LEDGER_PATH) -> dict[str, Any]:
    """Check that every run row carries the provenance the pre-registration requires."""
    required = ("git_hash", "config_hash", "eval_code_hash", "package_versions", "timestamp")
    llm_required = ("model_id", "cli_version", "claude_md_sha256")
    problems = []
    rows = read_ledger(path, latest_only=False)
    for r in rows:
        if r.get("kind") != "run":
            continue
        for f in required:
            if r.get(f) in (None, "", {}):
                problems.append({"row_id": r.get("row_id"), "missing": f})
        if str(r.get("arm", "")).upper() in ("B", "C", "A+"):
            for f in llm_required:
                if r.get(f) in (None, ""):
                    problems.append({"row_id": r.get("row_id"), "missing": f})
    return {"n_rows": len(rows), "n_problems": len(problems), "problems": problems, "ok": not problems}


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def flatten(d: Mapping[str, Any], prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            flatten(v, key + ".", out)
        else:
            out[key] = v
    return out


def rows_to_frame(rows: Iterable[Mapping[str, Any]], kind: str | None = "run") -> pd.DataFrame:
    flat = [flatten(r) for r in rows if kind is None or r.get("kind") == kind]
    if not flat:
        return pd.DataFrame()
    df = pd.DataFrame(flat)
    for c in ("n_docs", "k"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _col(df: pd.DataFrame, path: str) -> pd.Series:
    return pd.to_numeric(df[path], errors="coerce") if path in df else pd.Series([np.nan] * len(df), index=df.index)


def _select(df: pd.DataFrame, columns: Mapping[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name in ("arm", "corpus", "n_docs", "k_variant", "variant", "replicate"):
        out[name] = df[name] if name in df else ""
    for name, path in columns.items():
        out[name] = _col(df, path)
    return out


def aggregate(per_run: pd.DataFrame, group: Sequence[str] = GROUP_FIELDS) -> pd.DataFrame:
    """Mean / sd over replicates for every numeric column (``n_replicates`` added)."""
    if per_run.empty:
        return per_run
    num = [c for c in per_run.columns if c not in group and c != "replicate" and pd.api.types.is_numeric_dtype(per_run[c])]
    g = per_run.groupby(list(group), dropna=False, sort=True)
    mean = g[num].mean()
    sd = g[num].std(ddof=1)
    n = g.size().rename("n_replicates")
    out = pd.concat([n, mean, sd.add_suffix("_sd")], axis=1).reset_index()
    return out


def _level(df: pd.DataFrame) -> str:
    if "metrics.primary_level" in df and df["metrics.primary_level"].notna().any():
        return str(df["metrics.primary_level"].dropna().iloc[0])
    return "l2"


def table_category_metrics(df: pd.DataFrame) -> pd.DataFrame:  # T1
    lvl = _level(df)
    cols = {"k": "metrics.categories.k_produced", "n_uncategorised": "metrics.categories.n_uncategorised"}
    for m in ("nmi", "ari", "ami", "p1", "purity", "inverse_purity", "homogeneity", "completeness"):
        cols[f"{m}_{lvl}"] = f"metrics.categories.levels.{lvl}.{m}"
    for m in ("nmi", "ari", "p1"):
        cols[f"{m}_{lvl}_random"] = f"metrics.categories.levels.{lvl}.random_baseline.{m}"
        cols[f"{m}_{lvl}_ci_low"] = f"metrics.ci.{m}_{lvl}.ci_low"
        cols[f"{m}_{lvl}_ci_high"] = f"metrics.ci.{m}_{lvl}.ci_high"
    for m in ("nmi", "ari", "p1"):
        cols[f"{m}_l1"] = f"metrics.categories.levels.l1.{m}"
    return _select(df, cols)


def table_link_metrics(df: pd.DataFrame) -> pd.DataFrame:  # T2
    lvl = _level(df)
    k = 5
    cols = {f"hyperlink_p_at_{k}": "metrics.links.hyperlinks.precision_at_k",
            f"hyperlink_r_at_{k}": "metrics.links.hyperlinks.recall_at_k", "hyperlink_map": "metrics.links.hyperlinks.map",
            f"hyperlink_oracle_p_at_{k}": "metrics.links.hyperlinks.oracle_precision_at_k",
            f"hyperlink_random_p_at_{k}": "metrics.links.hyperlinks.random_precision_at_k",
            "hyperlink_random_map": "metrics.links.hyperlinks.random_map",
            f"hyperlink_p_at_{k}_ci_low": f"metrics.ci.hyperlink_p_at_{k}.ci_low",
            f"hyperlink_p_at_{k}_ci_high": f"metrics.ci.hyperlink_p_at_{k}.ci_high",
            f"same_{lvl}_p_at_{k}": f"metrics.links.same_class.{lvl}.precision_at_k",
            f"same_{lvl}_chance": f"metrics.links.same_class.{lvl}.chance",
            f"same_{lvl}_p_at_{k}_ci_low": f"metrics.ci.same_{lvl}_p_at_{k}.ci_low",
            f"same_{lvl}_p_at_{k}_ci_high": f"metrics.ci.same_{lvl}_p_at_{k}.ci_high",
            f"same_l1_p_at_{k}": "metrics.links.same_class.l1.precision_at_k", "same_l1_chance": "metrics.links.same_class.l1.chance",
            "ild": "metrics.links.diversity.ild", "same_category_frac": "metrics.links.category_mix.same_category_frac",
            "unique_category_frac": "metrics.links.category_mix.unique_category_frac",
            "mean_links_per_doc": "metrics.links.category_mix.mean_links_per_doc",
            "links_resolved_frac": "metrics.links.validity.resolved_frac", "links_near_miss_frac": "metrics.links.validity.near_miss_frac",
            "links_dangling_frac": "metrics.links.validity.dangling_frac"}
    return _select(df, cols)


def table_label_metrics(df: pd.DataFrame, human_rows: Sequence[Mapping[str, Any]] = ()) -> pd.DataFrame:  # T3
    cols = {"label_sim": "metrics.labels.weighted_mean", "label_sim_unweighted": "metrics.labels.unweighted_mean",
            "label_sim_ci_low": "metrics.ci.label_sim.ci_low", "label_sim_ci_high": "metrics.ci.label_sim.ci_high",
            "label_sim_floor_random": "metrics.labels.floor_random_weighted",
            "label_sim_floor_all_other": "metrics.labels.floor_all_other_unweighted",
            "jaccard": "metrics.labels.jaccard_weighted", "npmi": "metrics.labels.npmi.mean",
            "npmi_weighted": "metrics.labels.npmi.weighted_mean", "n_clusters": "metrics.labels.n_clusters"}
    out = _select(df, cols)
    human = {}
    for r in human_rows:
        if r.get("variant", "all_raters") == "all_raters":
            human[str(r.get("arm"))] = (r.get("mean"), r.get("ci_low"), r.get("ci_high"))
    out["human_mean"] = [human.get(str(a), (np.nan,) * 3)[0] for a in out["arm"]]
    out["human_ci_low"] = [human.get(str(a), (np.nan,) * 3)[1] for a in out["arm"]]
    out["human_ci_high"] = [human.get(str(a), (np.nan,) * 3)[2] for a in out["arm"]]
    return out


STRUCTURE_COLS = ("n_pages", "n_categories", "depth", "docs_per_category_gini", "docs_per_category_entropy_norm",
                  "orphan_rate", "reciprocity_raw", "reciprocity_gl", "density", "avg_clustering", "pagerank_top5_mass",
                  "pagerank_gini", "largest_wcc_frac", "median_in_degree", "median_out_degree", "n_extra_pages",
                  "extra_pages_words")


def table_structure_metrics(df: pd.DataFrame) -> pd.DataFrame:  # T4
    cols = {c: f"metrics.structure.{c}" for c in STRUCTURE_COLS}
    out = _select(df, cols)
    agg = aggregate(out)
    ref = kamps_koolen_row()
    ref_row = {c: ref.get(c) for c in agg.columns}
    ref_row.update({"arm": ref["arm"], "corpus": "Wikipedia", "n_docs": np.nan, "k_variant": "", "variant": "", "n_replicates": np.nan})
    return pd.concat([agg, pd.DataFrame([ref_row])], ignore_index=True)


def table_cost_quality(df: pd.DataFrame) -> pd.DataFrame:  # F2 source
    lvl = _level(df)
    cols = {"usd": "cost.total_usd", "usd_uncached": "cost.uncached_usd", "usd_reported": "cost.reported_usd",
            "wall_hours": "cost.wall_hours", "num_turns": "cost.num_turns", "n_llm_calls": "cost.n_calls",
            "gpu_seconds": "cost.gpu_seconds", "cpu_seconds": "cost.cpu_seconds",
            f"nmi_{lvl}": f"metrics.categories.levels.{lvl}.nmi", f"nmi_{lvl}_ci_low": f"metrics.ci.nmi_{lvl}.ci_low",
            f"nmi_{lvl}_ci_high": f"metrics.ci.nmi_{lvl}.ci_high", f"p1_{lvl}": f"metrics.categories.levels.{lvl}.p1",
            "label_sim": "metrics.labels.weighted_mean", "label_sim_ci_low": "metrics.ci.label_sim.ci_low",
            "label_sim_ci_high": "metrics.ci.label_sim.ci_high",
            f"same_{lvl}_p_at_5": f"metrics.links.same_class.{lvl}.precision_at_k",
            f"same_{lvl}_p_at_5_ci_low": f"metrics.ci.same_{lvl}_p_at_5.ci_low",
            f"same_{lvl}_p_at_5_ci_high": f"metrics.ci.same_{lvl}_p_at_5.ci_high",
            "hyperlink_p_at_5": "metrics.links.hyperlinks.precision_at_k"}
    return _select(df, cols)


def table_ablation_mmr(df: pd.DataFrame) -> pd.DataFrame:  # F3 source
    lvl = _level(df)
    lam = None
    for c in ("config.mmr_lambda", "config.links.mmr_lambda", "extra.mmr_lambda", "config.lambda"):
        if c in df:
            lam = pd.to_numeric(df[c], errors="coerce")
            break
    if lam is None:
        lam = df["variant"].astype(str).str.extract(r"mmr[_-]?(\d*\.?\d+)")[0].astype(float) if "variant" in df else pd.Series(np.nan, index=df.index)
    cols = {f"same_{lvl}_p_at_5": f"metrics.links.same_class.{lvl}.precision_at_k",
            f"same_{lvl}_p_at_5_ci_low": f"metrics.ci.same_{lvl}_p_at_5.ci_low",
            f"same_{lvl}_p_at_5_ci_high": f"metrics.ci.same_{lvl}_p_at_5.ci_high",
            "hyperlink_p_at_5": "metrics.links.hyperlinks.precision_at_k", "hyperlink_map": "metrics.links.hyperlinks.map",
            "ild": "metrics.links.diversity.ild", "unique_category_frac": "metrics.links.category_mix.unique_category_frac"}
    out = _select(df, cols)
    out.insert(0, "mmr_lambda", lam.values)
    out = out[out["mmr_lambda"].notna()]
    return out[out["arm"].astype(str).isin(["A", "A+"])]  # the MMR ablation is an arm-A sweep


def table_cost_per_ingest(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:  # F4 source
    recs = []
    for r in rows:
        if r.get("kind") != "run":
            continue
        pts = (r.get("cost") or {}).get("per_ingest") or []
        fit = (r.get("cost") or {}).get("marginal_fit") or {}
        for p in pts:
            recs.append({"arm": r.get("arm"), "corpus": r.get("corpus"), "n_docs": r.get("n_docs"), "replicate": r.get("replicate"),
                         "ingest_index": p.get("ingest_index"), "wiki_pages_before": p.get("wiki_pages_before"),
                         "usd": p.get("usd"), "usd_uncached": p.get("usd_uncached"), "tokens": p.get("tokens"),
                         "num_turns": p.get("num_turns"), "wall_ms": p.get("wall_ms"),
                         "fit_slope": fit.get("slope"), "fit_intercept": fit.get("intercept"),
                         "fit_slope_ci_low": fit.get("slope_ci_low"), "fit_slope_ci_high": fit.get("slope_ci_high")})
    return pd.DataFrame(recs)


def table_k_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """NMI / ARI / P1 vs k for arm A.  Uses a dedicated ``k_grid`` / ``k_sensitivity``
    variant when the ledger has one; otherwise every headline-lambda arm-A run
    (oracle / auto / two-level / k-matched each contribute their own k)."""
    lvl = _level(df)
    cols = {f"nmi_{lvl}": f"metrics.categories.levels.{lvl}.nmi", f"ari_{lvl}": f"metrics.categories.levels.{lvl}.ari",
            f"p1_{lvl}": f"metrics.categories.levels.{lvl}.p1", "label_sim": "metrics.labels.weighted_mean"}
    if df.empty or "k_variant" not in df:
        return pd.DataFrame()
    grid = df[df["k_variant"].astype(str).isin(["k_grid", "k_sensitivity"])]
    if grid.empty:
        headline = df["variant"].isna() | (df["variant"].astype(str).str.strip() == "") if "variant" in df else pd.Series(True, index=df.index)
        grid = df[headline & df["arm"].astype(str).isin(["A"])]
    if grid.empty:
        return pd.DataFrame()
    out = _select(grid, cols)
    out.insert(0, "k", _col(grid, "k").fillna(_col(grid, "metrics.categories.k_produced")).values)
    return out


def table_generic(rows: Iterable[Mapping[str, Any]], kind: str) -> pd.DataFrame:
    recs = []
    for r in rows:
        if r.get("kind") != kind:
            continue
        base = {k: r.get(k) for k in ("arm", "corpus", "n_docs", "replicate", "variant", "row_id", "timestamp")}
        payload = r.get("metrics") or {}
        table_rows = payload.get("rows")
        if isinstance(table_rows, list):
            for tr in table_rows:
                recs.append({**base, **flatten(tr)})
        else:
            recs.append({**base, **flatten(payload)})
    return pd.DataFrame(recs)


def headline_replicate(rows: Iterable[Mapping[str, Any]], corpus: str, n_docs: int, arm: str = "A",
                       k_variant: str = "k_matched", variant: str = "") -> dict[str, Any] | None:
    """The arm-A replicate with the median NMI (pre-registered seed selection)."""
    cands = [r for r in rows if r.get("kind") == "run" and r.get("arm") == arm and r.get("corpus") == corpus
             and int(r.get("n_docs", -1)) == int(n_docs) and r.get("k_variant", "") == k_variant and (r.get("variant") or "") == variant]
    if not cands:
        return None
    lvl = cands[0].get("metrics", {}).get("primary_level", "l2")
    cands.sort(key=lambda r: (r["metrics"]["categories"]["levels"][lvl]["nmi"], str(r.get("replicate"))))
    return cands[(len(cands) - 1) // 2]


def write_tables(rows: Sequence[Mapping[str, Any]] | None = None, tables_dir: str | Path = TABLES_DIR,
                 ledger_path: str | Path = LEDGER_PATH) -> dict[str, Path]:
    """Render every CSV of Appendix D from the ledger."""
    rows = list(rows) if rows is not None else read_ledger(ledger_path)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = rows_to_frame(rows, kind="run")
    written: dict[str, Path] = {}

    def save(name: str, table: pd.DataFrame) -> None:
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path

    human_rows = table_generic(rows, "human").to_dict("records")
    if not df.empty:
        t1 = table_category_metrics(df); save("category_metrics_runs", t1); save("category_metrics", aggregate(t1))
        t2 = table_link_metrics(df); save("link_metrics_runs", t2); save("link_metrics", aggregate(t2))
        t3 = table_label_metrics(df, human_rows); save("label_metrics_runs", t3); save("label_metrics", aggregate(t3))
        save("structure_metrics", table_structure_metrics(df))
        cq = table_cost_quality(df); save("cost_quality_runs", cq); save("cost_quality", aggregate(cq))
        ab = table_ablation_mmr(df)
        if not ab.empty:  # one curve per clustering variant: never average oracle/auto/two-level runs into one point
            save("ablation_mmr", aggregate(ab, ["arm", "corpus", "n_docs", "k_variant", "mmr_lambda"]))
        ks = table_k_sensitivity(df)
        if not ks.empty:
            save("k_sensitivity", aggregate(ks, ["arm", "corpus", "n_docs", "k_variant", "k"]))
    cpi = table_cost_per_ingest(rows)
    if not cpi.empty:
        save("cost_per_ingest", cpi)
    for kind, name in (("human", "human_eval"), ("stability", "stability"), ("judge", "judge_agreement"),
                       ("decision", "decision"), ("compare", "comparisons")):
        t = table_generic(rows, kind)
        if not t.empty:
            save(name, t)
    manifest = {"written": {k: str(v) for k, v in written.items()}, "n_ledger_rows": len(rows),
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "eval_code_hash": eval_code_hash()}
    (tables_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return written


# --------------------------------------------------------------------------
# Run-directory evaluation (the ``wikiorg eval`` command)
# --------------------------------------------------------------------------
HEADLINE_LAMBDA = 0.7
LLM_ARMS = ("B", "C", "A+")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml  # type: ignore
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_corpus_dir(manifest: Mapping[str, Any], corpus_dir: str | Path | None = None,
                       repo: str | Path = REPO_ROOT) -> Path:
    """``data/<corpus>/<N>`` from the run manifest (``tests/fixtures/<corpus>`` for fixtures)."""
    if corpus_dir is not None:
        return Path(corpus_dir)
    for key in ("corpus_dir", "corpus_path"):
        if manifest.get(key):
            return Path(manifest[key])
    corpus = str(manifest.get("corpus") or "")
    n = manifest.get("n") or manifest.get("n_docs")
    repo = Path(repo)
    candidates = [repo / "data" / corpus / str(n), repo / "tests" / "fixtures" / corpus, repo / "data" / corpus]
    for c in candidates:
        if (c / "gold.jsonl").exists():
            return c
    raise FileNotFoundError(f"no corpus directory with gold.jsonl for corpus={corpus!r} n={n!r} (tried {[str(c) for c in candidates]})")


def load_corpus(corpus_dir: str | Path) -> tuple[Any, dict[str, str]]:
    """(Gold, texts) from a corpus directory; titles come from the documents when the gold lacks them."""
    from wikiorg.adapters.common import split_frontmatter

    from .state import read_gold as _read_gold

    corpus_dir = Path(corpus_dir)
    gold = _read_gold(corpus_dir / "gold.jsonl")
    texts: dict[str, str] = {}
    docs_dir = corpus_dir / "docs"
    if docs_dir.is_dir():
        for p in sorted(docs_dir.glob("*.md")):
            fields, body = split_frontmatter(p.read_text(encoding="utf-8"))
            pid = str(fields.get("id") or p.stem)
            texts[pid] = body.strip()
            if not gold.titles.get(pid) and fields.get("title"):
                gold.titles[pid] = str(fields["title"])
    gold.texts = texts
    return gold, texts


def make_judge(judge: str = "auto") -> tuple[Any, dict[str, Any]]:
    """The judge embedder: mpnet when requested/available, the hashing stand-in otherwise."""
    from .labels import JUDGE_MODEL, HashingEmbedder, JudgeEmbedder

    info: dict[str, Any] = {"requested": judge}
    if judge == "none":
        return None, info | {"model": None, "stand_in": False}
    if judge in ("mpnet", "auto"):
        available = True
        if judge == "auto":
            try:
                from huggingface_hub import try_to_load_from_cache  # type: ignore
                available = try_to_load_from_cache(JUDGE_MODEL, "config.json") is not None
            except Exception:
                available = False
        if available:
            return JudgeEmbedder(), info | {"model": JUDGE_MODEL, "stand_in": False}
        if judge == "mpnet":
            raise RuntimeError(f"judge model {JUDGE_MODEL} is not available in the local HF cache")
    return HashingEmbedder(), info | {"model": "HashingEmbedder (stand-in, NOT the thesis judge)", "stand_in": True}


JUDGE_CACHE_DIR = REPO_ROOT / "results" / "judge_cache"
JUDGE_DOC_CHARS = 2000  # mpnet truncates at 384 word pieces; ~2000 characters cover that


def judge_doc_embeddings(gold: Any, texts: Mapping[str, str], embed_fn: Any, judge_info: Mapping[str, Any],
                         cache_dir: str | Path | None = JUDGE_CACHE_DIR) -> dict[str, np.ndarray]:
    """Judge-model embeddings of the documents (for intra-list diversity), cached
    on disk per (model, document set, text hash) so a corpus is embedded once."""
    ids = [i for i in gold.ids() if i in texts]
    inputs = [f"{gold.titles.get(i, '')}\n{texts[i][:JUDGE_DOC_CHARS]}" for i in ids]
    key = sha256_bytes(_canonical({"model": judge_info.get("model"), "ids": ids,
                                   "texts": [sha256_bytes(t.encode()) for t in inputs]}).encode())[:24]
    cache = Path(cache_dir) / f"{key}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == ids:
            return {i: z["E"][n] for n, i in enumerate(ids)}
    E = np.asarray(embed_fn(inputs), dtype=np.float32)
    if cache is not None and not judge_info.get("stand_in"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(ids), E=E)
    return {i: E[n] for n, i in enumerate(ids)}


def _find_calls_file(run_dir: Path, manifest: Mapping[str, Any]) -> Path | None:
    for key in ("calls_file", "llm_calls", "calls_jsonl"):
        if manifest.get(key) and Path(manifest[key]).exists():
            return Path(manifest[key])
    for name in ("llm_calls.jsonl", "calls.jsonl"):
        if (run_dir / name).exists():
            return run_dir / name
    return None


def load_state(run_dir: Path, arm: str, gold: Any, manifest: Mapping[str, Any]) -> Any:
    """``wiki_state.json`` when present; otherwise build it through the arm's adapter."""
    from .state import WikiState

    path = run_dir / "wiki_state.json"
    if path.exists():
        return WikiState.load(path)
    if arm.upper().startswith("B"):
        from wikiorg.adapters.arm_b import from_extract_csv, from_wiki_dir
        extract = sorted(run_dir.glob("extract*.csv")) + sorted((run_dir / "extract").glob("*.csv")) if (run_dir / "extract").is_dir() else sorted(run_dir.glob("extract*.csv"))
        if (run_dir / "wiki").is_dir():
            st = from_wiki_dir(run_dir, corpus_ids=gold.ids(), corpus_titles=gold.titles, arm=arm)
        elif extract:
            st = from_extract_csv(extract[-1], corpus_ids=gold.ids(), corpus_titles=gold.titles, arm=arm)
        else:
            raise FileNotFoundError(f"{run_dir}: no wiki_state.json, wiki/ tree or extract CSV for arm B")
    elif arm.upper().startswith("C"):
        from wikiorg.adapters.arm_c import from_jsonl
        calls = _find_calls_file(run_dir, manifest)
        if calls is None:
            raise FileNotFoundError(f"{run_dir}: no wiki_state.json or calls file for arm C")
        st = from_jsonl(calls, gold.titles, arm=arm, run_id=manifest.get("run_id"))
    else:
        raise FileNotFoundError(f"{run_dir}: no wiki_state.json")
    st.save(path)
    return st


def run_cost(run_dir: Path, arm: str, manifest: Mapping[str, Any], n_resamples: int = 2000) -> dict[str, Any]:
    from .cost import aggregate_calls, arm_a_cost, arm_b_cost, load_prices

    prices = load_prices()
    timing = _read_json(run_dir / "timing.json")
    calls = _find_calls_file(run_dir, manifest)
    rows = []
    if calls is not None:
        from wikiorg.adapters.arm_c import read_jsonl
        rows = read_jsonl(calls)
        run_id = manifest.get("run_id")
        if run_id and any("run_id" in r for r in rows):
            rows = [r for r in rows if r.get("run_id", run_id) == run_id]
    if arm.upper() in ("A", "A+"):
        cost = arm_a_cost(timing, prices) if timing else {"total_usd": 0.0, "uncached_usd": 0.0, "n_calls": 0, "num_turns": 0,
                                                             "gpu_seconds": 0.0, "cpu_seconds": 0.0, "wall_hours": 0.0}
        cost["pipeline_usd"] = cost["total_usd"]
        if rows:  # A+: the naming calls on top of the pipeline
            llm = aggregate_calls(rows, prices)
            cost["llm"] = {k: v for k, v in llm.items() if k != "flagged_calls"}
            cost["llm_usd"] = llm["total_usd"]
            cost["total_usd"] = cost["pipeline_usd"] + llm["total_usd"]
            cost["uncached_usd"] = cost["pipeline_usd"] + llm["uncached_usd"]
            cost["n_calls"], cost["num_turns"], cost["tokens"] = llm["n_calls"], llm["num_turns"], llm["tokens"]
            cost["reported_usd"] = llm["reported_usd"]
        return cost
    if not rows:
        return {"total_usd": float("nan"), "uncached_usd": float("nan"), "n_calls": 0, "note": "no calls file found"}
    cost = arm_b_cost(rows, prices, n_resamples=n_resamples)
    cost["calls_file"] = str(calls)
    if timing:
        cost["wall_seconds_timing"] = timing.get("total_wall_s")
    return cost


def evaluate_run(run_dir: str | Path, corpus_dir: str | Path | None = None, *, ledger_path: str | Path = LEDGER_PATH,
                 n_resamples: int = 2000, judge: str = "auto", k: int = 5, seed: int = 0, n_random: int = 100,
                 append: bool | None = None, write_metrics: bool = True, allow_supersede: bool = False,
                 headline_lambda: float = HEADLINE_LAMBDA) -> dict[str, Any]:
    """Evaluate one run directory end to end and record it in the ledger.

    Reads ``wiki_state.json`` (or builds it through the arm's adapter), the
    run's ``manifest.json`` / ``config.yaml`` / ``timing.json`` / calls file,
    and the corpus (gold + documents).  Writes ``metrics.json`` into the run
    directory and appends the ledger row (unless the corpus is a test fixture
    and ``append`` is not given).  Returns a short summary.
    """
    from .suite import evaluate_run as _evaluate, strip_bulky

    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    config = _read_yaml(run_dir / "config.yaml")
    corpus = resolve_corpus_dir(manifest, corpus_dir)
    if append is None:  # fixture corpora (tests/fixtures/*) never enter the real ledger by default
        append = "fixtures" not in corpus.resolve().parts
    gold, texts = load_corpus(corpus)
    arm = str(manifest.get("arm") or "")
    state_path = run_dir / "wiki_state.json"
    if state_path.exists() and not arm:
        arm = str(json.loads(state_path.read_text(encoding="utf-8")).get("arm") or "A")
    arm = arm or "A"
    state = load_state(run_dir, arm, gold, manifest)
    arm = state.arm or arm
    embed_fn, judge_info = make_judge(judge)
    judge_embeddings = judge_doc_embeddings(gold, texts, embed_fn, judge_info) if (embed_fn is not None and texts) else None
    metrics = _evaluate(state, gold, texts=texts or None, embed_fn=embed_fn, judge_embeddings=judge_embeddings, k=k,
                        seed=seed, n_random=n_random, n_resamples=n_resamples)
    metrics["judge"] = judge_info
    cost = run_cost(run_dir, arm, manifest, n_resamples=n_resamples)
    lam = manifest.get("lambda", (config.get("run") or {}).get("lambda"))
    is_llm = arm.upper() in LLM_ARMS
    if is_llm:
        replicate = (manifest.get("replicate") or manifest.get("run_id") or manifest.get("order")
                     or manifest.get("order_file") or manifest.get("run") or run_dir.name)   # explicit label first: replicates of one order must not collide
        k_variant = manifest.get("variant") or "native"
    else:
        replicate = manifest.get("seed", run_dir.name)
        k_variant = manifest.get("variant") or (config.get("run") or {}).get("variant") or ""
    variant = ""
    if lam is not None and abs(float(lam) - float(headline_lambda)) > 1e-9:
        variant = f"mmr_{float(lam):g}"
    extra = {"run_id": manifest.get("run_id"), "lambda": lam, "k_star": manifest.get("k_star"), "k1": manifest.get("k1"),
             "gold_level": manifest.get("gold_level"), "embedding": manifest.get("embedding"), "sanity": manifest.get("sanity"),
             "corpus_dir": str(corpus), "mmr_lambda": lam}
    row = make_row(arm=arm, corpus=str(manifest.get("corpus") or corpus.parent.name), n_docs=metrics["n_docs"], replicate=replicate,
                   config=config or None, metrics=strip_bulky(metrics), cost=strip_bulky(cost), variant=variant, k_variant=str(k_variant),
                   k=manifest.get("k", metrics["categories"]["k_produced"]), seeds={"kmeans": manifest.get("seed"), "eval": seed},
                   model_id=manifest.get("llm_model") or manifest.get("model"), cli_version=manifest.get("cli_version"),
                   claude_md_sha256=manifest.get("claude_md_sha256"), effort=manifest.get("effort"),
                   corpus_manifest_sha256=manifest.get("corpus_manifest_sha256"),
                   price_table_date=manifest.get("price_table_fetched") or cost.get("price_table_date"), run_dir=run_dir, extra=extra)
    if manifest.get("config_hash"):
        row["config_hash"] = manifest["config_hash"]  # the producer's hash of the effective config
    if manifest.get("git_hash") and not row.get("git_hash"):
        row["git_hash"] = manifest["git_hash"]
    row["run_git_hash"] = manifest.get("git_hash")
    status = {"status": "not appended", "row_id": row["row_id"]}
    if append:
        status = append_ledger(row, ledger_path, allow_supersede=allow_supersede)
    if write_metrics:
        (run_dir / "metrics.json").write_text(json.dumps({"row_id": status.get("row_id"), "ledger": str(ledger_path), "arm": arm,
                                                          "metrics": row["metrics"], "cost": row["cost"], "judge": judge_info,
                                                          "timestamp": row["timestamp"]}, indent=1, default=_json_default))
    return {"run_dir": str(run_dir), "arm": arm, "corpus": row["corpus"], "n_docs": row["n_docs"], "replicate": replicate,
            "k_variant": k_variant, "variant": variant, "primary": metrics["primary"], "cost_usd": cost.get("total_usd"),
            "judge": judge_info.get("model"), "ledger": status}


def decide_runs(run_a: str | Path, run_b: str | Path, run_oracle: str | Path | None = None, corpus_dir: str | Path | None = None, *,
                ledger_path: str | Path = LEDGER_PATH, n_resamples: int = 2000, judge: str = "auto", k: int = 5, seed: int = 0,
                append: bool = True, cost_a: float | None = None, cost_b: float | None = None) -> dict[str, Any]:
    """The headline comparison: arm A run (k-matched, median seed) vs arm B run
    on the same corpus, floors from the oracle-k run, cost ratio from the runs'
    ``metrics.json`` (or the given USD).  Appends ``compare`` and ``decision`` rows."""
    from .bootstrap import decide
    from .cost import cost_ratio
    from .suite import compare_arms, evaluate_run as _evaluate, floors_from_metrics, strip_bulky

    run_a, run_b = Path(run_a), Path(run_b)
    man_a, man_b = _read_json(run_a / "manifest.json"), _read_json(run_b / "manifest.json")
    corpus = resolve_corpus_dir(man_a, corpus_dir)
    gold, texts = load_corpus(corpus)
    state_a = load_state(run_a, str(man_a.get("arm") or "A"), gold, man_a)
    state_b = load_state(run_b, str(man_b.get("arm") or "B"), gold, man_b)
    embed_fn, judge_info = make_judge(judge)
    cmp = compare_arms(state_a, state_b, gold, embed_fn=embed_fn, k=k, seed=seed, n_resamples=n_resamples)

    def usd(run: Path, given: float | None) -> float:
        if given is not None:
            return float(given)
        m = _read_json(run / "metrics.json")
        return float((m.get("cost") or {}).get("total_usd", float("nan")))

    ua, ub = usd(run_a, cost_a), usd(run_b, cost_b)
    ratio = cost_ratio(ua, ub)
    if run_oracle is not None:
        m_or = _read_json(Path(run_oracle) / "metrics.json").get("metrics")
        if not m_or:
            st_or = load_state(Path(run_oracle), "A", gold, _read_json(Path(run_oracle) / "manifest.json"))
            m_or = _evaluate(st_or, gold, embed_fn=embed_fn, with_ci=False, n_random=0)
        floors = floors_from_metrics(m_or)
    else:
        floors = {}
    dec = decide({e: cmp[e] for e in ("E1", "E2", "E3")}, ratio, floors)
    dec["cost_usd"] = {"a": ua, "b": ub}
    dec["runs"] = {"a": str(run_a), "b": str(run_b), "oracle": str(run_oracle) if run_oracle else None}
    dec["judge"] = judge_info
    out = {"decision": dec, "compare": strip_bulky({k_: v for k_, v in cmp.items()}), "ledger": {}}
    if append:
        corpus_name = str(man_a.get("corpus") or corpus.parent.name)
        n_docs = cmp["n_paired"]
        cmp_row = make_row(kind="compare", arm=f"{state_a.arm}-vs-{state_b.arm}", corpus=corpus_name, n_docs=n_docs,
                           replicate=f"{run_a.name}|{run_b.name}", metrics=out["compare"], extra={"judge": judge_info})
        dec_row = make_row(kind="decision", arm=f"{state_a.arm}-vs-{state_b.arm}", corpus=corpus_name, n_docs=n_docs,
                           replicate=f"{run_a.name}|{run_b.name}", metrics=strip_bulky(dec))
        out["ledger"] = {"compare": append_ledger(cmp_row, ledger_path, allow_supersede=True),
                         "decision": append_ledger(dec_row, ledger_path, allow_supersede=True)}
    return out


def _expand_runs(runs: Sequence[str], globs: Sequence[str]) -> list[Path]:
    import glob as _glob
    out: list[Path] = [Path(r) for r in runs]
    for g in globs:
        out += [Path(p) for p in sorted(_glob.glob(g))]
    return [p for p in out if p.is_dir()]


def main(argv: Sequence[str] | None = None) -> int:
    """``wikiorg eval --run DIR ...`` / ``wikiorg tables`` / ``wikiorg eval decide --a DIR --b DIR``.

    The CLI forwards everything after the sub-command, so the first token may
    name the action (``eval`` | ``tables`` | ``decide``); without it, ``--run``
    or ``--glob`` selects evaluation and anything else renders the tables.
    """
    import argparse

    argv = list(argv or [])
    while argv and argv[0] == "--":  # ``wikiorg eval -- --run DIR`` (argparse REMAINDER quirk in the CLI wrapper)
        argv.pop(0)
    action = None
    if argv and argv[0] in ("eval", "tables", "decide", "verify"):
        action = argv.pop(0)
    while argv and argv[0] == "--":
        argv.pop(0)
    p = argparse.ArgumentParser(prog="wikiorg eval", description="W3 evaluation: run metrics, ledger, tables, decision")
    p.add_argument("--run", action="append", default=[], help="run directory (repeatable)")
    p.add_argument("--glob", action="append", default=[], help="glob of run directories (repeatable)")
    p.add_argument("--corpus-dir", help="override the corpus directory (data/<corpus>/<N>)")
    p.add_argument("--ledger", default=str(LEDGER_PATH))
    p.add_argument("--out", default=str(TABLES_DIR), help="tables directory")
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--judge", default="auto", choices=["auto", "mpnet", "hash", "none"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-append", action="store_true", help="evaluate without touching the ledger")
    p.add_argument("--supersede", action="store_true", help="allow a changed row to supersede an existing key")
    p.add_argument("--a", help="decide: arm A run directory (k-matched, median seed)")
    p.add_argument("--b", help="decide: arm B run directory")
    p.add_argument("--oracle", help="decide: arm A run at oracle k for the floors")
    p.add_argument("--cost-a", type=float)
    p.add_argument("--cost-b", type=float)
    args = p.parse_args(argv)
    if action is None:
        action = "decide" if args.a and args.b else ("eval" if (args.run or args.glob) else "tables")
    if action == "verify":
        v = verify_ledger(args.ledger)
        print(json.dumps(v, indent=1))
        return 0 if v["ok"] else 1
    if action == "decide":
        res = decide_runs(args.a, args.b, args.oracle, args.corpus_dir, ledger_path=args.ledger, n_resamples=args.n_resamples,
                          judge=args.judge, seed=args.seed, append=not args.no_append, cost_a=args.cost_a, cost_b=args.cost_b)
        d = res["decision"]
        print(f"VERDICT: {d['verdict']}  cost ratio {d['cost']['ratio']:.1f} (>= {d['cost']['required']:g}: {d['cost']['met']})")
        for name, e in d["endpoints"].items():
            print(f"  {name} {e['metric']}: A-B = {e['estimate']:+.4f}  CI [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]  "
                  f"p={e['p_one_sided']:.4f} holm={e['p_holm']:.4f}  non-inferior={e['non_inferior_holm']} falsified={e['falsified']}")
        for name, f in d["floors"].items():
            print(f"  floor {name}: {f['value']} >= {f['required']} -> {f['met']}")
        return 0
    if action == "eval":
        runs = _expand_runs(args.run, args.glob)
        if not runs:
            print("no run directories given (--run / --glob)", file=sys.stderr)
            return 2
        rc = 0
        for run in runs:
            try:
                res = evaluate_run(run, args.corpus_dir, ledger_path=args.ledger, n_resamples=args.n_resamples, judge=args.judge,
                                   seed=args.seed, append=not args.no_append, allow_supersede=args.supersede)
            except Exception as exc:  # keep going over a grid; report at the end
                print(f"{run}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
                rc = 1
                continue
            prim = " ".join(f"{k_}={v:.3f}" for k_, v in res["primary"].items() if isinstance(v, float))
            print(f"{run}: [{res['ledger']['status']}] arm={res['arm']} {res['k_variant']} {res['variant'] or 'headline'} {prim} usd={res['cost_usd']}")
        return rc
    written = write_tables(None, args.out, args.ledger)
    for name, path in written.items():
        print(f"{name}: {path}")
    if not written:
        print(f"ledger {args.ledger} has no rows; nothing written")
    return 0
