"""Wiki structure metrics (networkx) with the Kamps & Koolen (2009) Wikipedia
reference row.

All graph statistics are computed on the **document-page related-link graph**
only (nodes = document pages, directed edge i -> j when j is in i's ``related``
list, self-links and duplicates removed).  Category pages are deliberately not
nodes: every document has an inbound link from its category page by
construction, which would make the orphan rate zero for every arm.

Metrics
-------
* ``n_pages``, ``n_categories`` (non-empty produced categories), ``depth``
  (number of link hops from ``index.md`` to a document page: index -> category
  -> page = 2 for a flat taxonomy; each ``parent`` level adds one).
* Docs-per-category **Gini** coefficient (0 = perfectly even sizes) and
  **normalised entropy** H / log k (1 = perfectly even; undefined for k = 1).
* **Orphan rate**: fraction of document pages with no inbound related link.
* **Reciprocity**: raw fraction of reciprocated links r = L^<->/L and the
  Garlaschelli & Loffredo (2004, PRL 93:268701) coefficient
  rho = (r - a) / (1 - a) with a = L / (N (N-1)) the density, which is 0 for a
  random graph of the same density, 1 for a fully reciprocal one and negative
  for anti-reciprocal graphs.
* **Density**, **average clustering coefficient** (Watts & Strogatz 1998, on the
  undirected simple graph), **PageRank** (Brin & Page 1998) top-5 % mass and
  Gini, **largest weakly-connected-component** fraction, median in/out degree.
* Kamps & Koolen (2009), *Is Wikipedia link structure different?*, WSDM: the
  English Wikipedia's median in-degree 4 and median out-degree 12 (comparison
  row; their other statistics are not directly comparable to a 300-page wiki).
* Arm-B extras from ``extra_pages``: counts by page type and total words.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import networkx as nx
import numpy as np

from .state import WikiState, coerce_state

KAMPS_KOOLEN_2009: dict[str, Any] = {
    "source": "Kamps & Koolen (2009) WSDM, English Wikipedia",
    "median_in_degree": 4,
    "median_out_degree": 12,
}


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of non-negative values (0 = all equal)."""
    x = np.sort(np.asarray(list(values), dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def normalised_entropy(counts: Sequence[float]) -> float:
    """Shannon entropy of the size distribution divided by log(k); NaN for k <= 1."""
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    k = len(c)
    if k <= 1:
        return float("nan")
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(k))


def doc_graph(state: WikiState | Any, ids: Sequence[str] | None = None) -> nx.DiGraph:
    st = coerce_state(state)
    ids = list(ids) if ids is not None else st.ids()
    id_set = set(ids)
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    for pid, lst in st.links().items():
        if pid not in id_set:
            continue
        for r in lst:
            if r in id_set:
                G.add_edge(pid, r)
    return G


def hierarchy_depth(state: WikiState | Any) -> int:
    st = coerce_state(state)
    depth_cache: dict[str, int] = {}

    def depth(cid: str, seen: set[str]) -> int:
        if cid in depth_cache:
            return depth_cache[cid]
        cat = st.categories.get(cid)
        parent = cat.parent if cat else None
        if parent is None or parent not in st.categories or parent in seen:
            d = 1
        else:
            d = 1 + depth(parent, seen | {cid})
        depth_cache[cid] = d
        return d

    used = {p.category for p in st.pages.values() if p.category is not None}
    if not used:
        return 1
    return 1 + max(depth(c, {c}) for c in used)


def reciprocity(G: nx.DiGraph) -> dict[str, float]:
    n, L = G.number_of_nodes(), G.number_of_edges()
    if L == 0 or n < 2:
        return {"reciprocity_raw": float("nan"), "reciprocity_gl": float("nan"), "density": 0.0 if n > 1 else float("nan")}
    r = float(nx.overall_reciprocity(G))
    a = L / (n * (n - 1))
    rho = (r - a) / (1 - a) if a < 1 else float("nan")
    return {"reciprocity_raw": r, "reciprocity_gl": float(rho), "density": float(a)}


def pagerank_stats(G: nx.DiGraph, top_frac: float = 0.05) -> dict[str, float]:
    n = G.number_of_nodes()
    if n == 0:
        return {"pagerank_top5_mass": float("nan"), "pagerank_gini": float("nan")}
    pr = nx.pagerank(G) if G.number_of_edges() else {v: 1.0 / n for v in G}
    vals = np.sort(np.array(list(pr.values())))[::-1]
    top = max(1, math.ceil(top_frac * n))
    return {"pagerank_top5_mass": float(vals[:top].sum()), "pagerank_gini": gini(vals)}


def extra_page_stats(extra_pages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_type: Counter = Counter()
    words = 0
    for ep in extra_pages:
        t = ep.get("type") or ep.get("kind") or ep.get("page_type") or "other"
        by_type[str(t)] += 1
        w = ep.get("words", ep.get("n_words", 0)) or 0
        try:
            words += int(w)
        except (TypeError, ValueError):
            pass
    return {"n_extra_pages": int(sum(by_type.values())), "extra_pages_by_type": dict(by_type),
            "extra_pages_words": int(words)}


def structure_metrics(state: WikiState | Any, ids: Sequence[str] | None = None) -> dict[str, Any]:
    st = coerce_state(state)
    ids = list(ids) if ids is not None else st.ids()
    G = doc_graph(st, ids)
    n = G.number_of_nodes()
    sizes = [len(m) for cid, m in st.members().items() if m]
    in_deg = np.array([d for _, d in G.in_degree()]) if n else np.array([])
    out_deg = np.array([d for _, d in G.out_degree()]) if n else np.array([])
    out: dict[str, Any] = {
        "n_pages": n,
        "n_categories": len(sizes),
        "n_uncategorised": sum(1 for i in ids if (st.pages.get(i) is None or st.pages[i].category is None)),
        "depth": hierarchy_depth(st),
        "docs_per_category_mean": float(np.mean(sizes)) if sizes else float("nan"),
        "docs_per_category_median": float(np.median(sizes)) if sizes else float("nan"),
        "docs_per_category_min": int(min(sizes)) if sizes else 0,
        "docs_per_category_max": int(max(sizes)) if sizes else 0,
        "docs_per_category_gini": gini(sizes),
        "docs_per_category_entropy_norm": normalised_entropy(sizes),
        "n_edges": G.number_of_edges(),
        "mean_out_degree": float(out_deg.mean()) if n else float("nan"),
        "median_out_degree": float(np.median(out_deg)) if n else float("nan"),
        "median_in_degree": float(np.median(in_deg)) if n else float("nan"),
        "max_in_degree": int(in_deg.max()) if n else 0,
        "orphan_rate": float(np.mean(in_deg == 0)) if n else float("nan"),
        "avg_clustering": float(nx.average_clustering(G.to_undirected())) if n else float("nan"),
        "largest_wcc_frac": float(max(len(c) for c in nx.weakly_connected_components(G)) / n) if n else float("nan"),
    }
    out.update(reciprocity(G))
    out.update(pagerank_stats(G))
    out.update(extra_page_stats(st.extra_pages))
    return out


def kamps_koolen_row() -> dict[str, Any]:
    """Reference row for T4 (missing statistics are None)."""
    row = {k: None for k in ("n_pages", "n_categories", "depth", "docs_per_category_gini",
                             "docs_per_category_entropy_norm", "orphan_rate", "reciprocity_raw",
                             "reciprocity_gl", "density", "avg_clustering", "pagerank_top5_mass",
                             "pagerank_gini", "largest_wcc_frac")}
    row.update(KAMPS_KOOLEN_2009)
    row["arm"] = "Wikipedia (Kamps & Koolen 2009)"
    return row
