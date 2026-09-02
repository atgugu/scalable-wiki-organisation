"""Related-link quality.

* **Hyperlink agreement (WikiCS).**  Gold = the induced *undirected union* of the
  real Wikipedia hyperlinks among the corpus documents.  Precision@k, recall@k
  and MAP (mean average precision truncated at k; AP normalised by
  ``min(|gold_i|, k)`` so that a perfect list scores 1) over the documents that
  have at least one gold link (Manning, Raghavan & Schütze 2008, ch. 8).
  The *oracle ceiling* is ``mean_i min(deg_i, k) / k`` -- the best precision any
  k-list can reach when a document has fewer than k gold links; the random
  baseline draws k random other documents (Monte-Carlo, seeded) and also has the
  closed form ``mean_i deg_i / (N-1)`` for precision.
* **Same-class precision (DBPedia).**  Fraction of a document's links that share
  its gold ``l2`` (or ``l1``) class; the pre-registered endpoint E3 is the
  same-l2 precision@5.  Chance level is the class collision probability
  ``sum_c p_c^2`` (pre-registration) and its finite-population form
  ``sum_c p_c (n_c-1)/(N-1)``.  A document with an empty link list scores 0
  (no links produced = no correct links); ``n_without_links`` is reported.
* **Intra-list diversity** (Ziegler et al. 2005; Smyth & McClave 2001):
  mean pairwise ``1 - cosine`` among the linked documents, in judge (mpnet)
  embedding space.
* **Category mix**: fraction of links inside the source's produced category and
  the fraction of distinct categories among a document's links.
* **Link validity** for the LLM arms: each raw link target written by the model
  is *resolved* (exact match on id or normalised title), a *near miss* (difflib
  ratio >= 0.9 to a known title, never turned into a link) or *dangling*.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

from wikiorg.adapters.common import TitleIndex, norm_title

from .state import Gold, WikiState, coerce_state, common_ids


def produced_links(state: WikiState | Any, ids: Iterable[str] | None = None, k: int | None = 5) -> dict[str, list[str]]:
    """Clean link lists (no self, no duplicates, only known pages), truncated to k."""
    st = coerce_state(state)
    links = st.links()
    ids = list(ids) if ids is not None else st.ids()
    out = {}
    for pid in ids:
        lst = links.get(pid, [])
        out[pid] = lst[:k] if k is not None else list(lst)
    return out


# --------------------------------------------------------------------------
# Ranked-list metrics
# --------------------------------------------------------------------------
def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    top = list(ranked)[:k]
    return sum(1 for r in top if r in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top = list(ranked)[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """AP@k = (1 / min(|relevant|, k)) * sum_{i<=k} P@i * rel_i."""
    if not relevant:
        return float("nan")
    top = list(ranked)[:k]
    hits, s = 0, 0.0
    for i, r in enumerate(top, start=1):
        if r in relevant:
            hits += 1
            s += hits / i
    return s / min(len(relevant), k)


def hyperlink_metrics(state: WikiState | Any, gold: Gold, k: int = 5, ids: Sequence[str] | None = None,
                      n_random: int = 100, seed: int = 0) -> dict[str, Any]:
    """P@k / R@k / MAP against gold hyperlinks with oracle and random baselines."""
    st = coerce_state(state)
    if not gold.has_links():
        raise ValueError("gold has no links")
    ids = list(ids) if ids is not None else common_ids(st, gold)
    universe = [i for i in ids if i in gold.links]  # type: ignore[operator]
    evaluated = [i for i in universe if len(gold.links[i]) > 0]  # type: ignore[index]
    links = produced_links(st, evaluated, k)
    P, R, AP, deg = [], [], [], []
    for pid in evaluated:
        rel = gold.links[pid]  # type: ignore[index]
        P.append(precision_at_k(links[pid], rel, k))
        R.append(recall_at_k(links[pid], rel, k))
        AP.append(average_precision_at_k(links[pid], rel, k))
        deg.append(len(rel))
    P, R, AP, deg_a = map(np.asarray, (P, R, AP, deg))
    N = len(universe)
    out: dict[str, Any] = {
        "k": k, "n_evaluated": len(evaluated), "n_universe": N,
        "n_without_links_produced": int(sum(1 for i in evaluated if not links[i])),
        "precision_at_k": float(P.mean()) if len(P) else float("nan"),
        "recall_at_k": float(R.mean()) if len(R) else float("nan"),
        "map": float(AP.mean()) if len(AP) else float("nan"),
        "oracle_precision_at_k": float(np.mean(np.minimum(deg_a, k) / k)) if len(deg_a) else float("nan"),
        "oracle_recall_at_k": float(np.mean(np.minimum(deg_a, k) / deg_a)) if len(deg_a) else float("nan"),
        "random_precision_closed_form": float(np.mean(deg_a / max(N - 1, 1))) if len(deg_a) else float("nan"),
        "median_gold_degree": float(np.median(deg_a)) if len(deg_a) else float("nan"),
        "per_document": {"precision_at_k": dict(zip(evaluated, P.tolist())),
                          "recall_at_k": dict(zip(evaluated, R.tolist())),
                          "ap_at_k": dict(zip(evaluated, AP.tolist()))},
    }
    if n_random > 0 and len(evaluated):
        rng = np.random.default_rng(seed)
        idx = {i: n for n, i in enumerate(universe)}
        rp, rr, rap = [], [], []
        for _ in range(n_random):
            p_, r_, a_ = [], [], []
            for pid in evaluated:
                cand = np.array([j for j in range(N) if j != idx[pid]])
                draw = rng.choice(cand, size=min(k, len(cand)), replace=False)
                ranked = [universe[j] for j in draw]
                rel = gold.links[pid]  # type: ignore[index]
                p_.append(precision_at_k(ranked, rel, k))
                r_.append(recall_at_k(ranked, rel, k))
                a_.append(average_precision_at_k(ranked, rel, k))
            rp.append(np.mean(p_)); rr.append(np.mean(r_)); rap.append(np.mean(a_))
        out["random_precision_at_k"] = float(np.mean(rp))
        out["random_recall_at_k"] = float(np.mean(rr))
        out["random_map"] = float(np.mean(rap))
        out["n_random_draws"] = n_random
    return out


# --------------------------------------------------------------------------
# Same-class precision (DBPedia)
# --------------------------------------------------------------------------
def same_class_precision(state: WikiState | Any, gold: Gold, level: str = "l2", k: int = 5,
                         ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Fraction of each document's (<= k) links that share its gold class."""
    st = coerce_state(state)
    ids = list(ids) if ids is not None else common_ids(st, gold, level)
    gl = gold.labels[level]
    links = produced_links(st, ids, k)
    vals = []
    n_without = 0
    for pid in ids:
        lst = [r for r in links[pid] if r in gl]
        if not lst:
            n_without += 1
            vals.append(0.0)
            continue
        vals.append(sum(1 for r in lst if gl[r] == gl[pid]) / len(lst))
    counts = Counter(gl[i] for i in ids)
    N = len(ids)
    p = np.array([c / N for c in counts.values()]) if N else np.array([])
    chance = float(np.sum(p ** 2)) if N else float("nan")
    chance_fp = float(sum((c / N) * ((c - 1) / max(N - 1, 1)) for c in counts.values())) if N else float("nan")
    v = np.asarray(vals)
    return {"level": level, "k": k, "n_docs": N, "n_without_links": n_without,
            "precision_at_k": float(v.mean()) if N else float("nan"),
            "precision_at_k_linked_only": float(v[[bool(links[i]) for i in ids]].mean()) if N - n_without else float("nan"),
            "chance": chance, "chance_finite_population": chance_fp,
            "per_document": dict(zip(ids, v.tolist()))}


# --------------------------------------------------------------------------
# Diversity and category mix
# --------------------------------------------------------------------------
def intra_list_diversity(state: WikiState | Any, embeddings: dict[str, np.ndarray],
                         ids: Sequence[str] | None = None, k: int = 5) -> dict[str, Any]:
    """Mean pairwise (1 - cosine) among each document's linked pages."""
    st = coerce_state(state)
    ids = list(ids) if ids is not None else st.ids()
    links = produced_links(st, ids, k)
    vals: dict[str, float] = {}
    for pid in ids:
        lst = [r for r in links[pid] if r in embeddings]
        if len(lst) < 2:
            continue
        E = np.vstack([embeddings[r] for r in lst]).astype(float)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        S = E @ E.T
        iu = np.triu_indices(len(lst), k=1)
        vals[pid] = float(np.mean(1.0 - S[iu]))
    v = np.array(list(vals.values()))
    return {"ild": float(v.mean()) if len(v) else float("nan"), "n_docs": len(vals),
            "per_document": vals}


def category_mix(state: WikiState | Any, ids: Sequence[str] | None = None, k: int = 5) -> dict[str, Any]:
    """Share of links inside the source's own produced category, and the share
    of distinct categories among a document's links (uncategorised = own id)."""
    st = coerce_state(state)
    ids = list(ids) if ids is not None else st.ids()
    part = st.partition(st.ids(), singleton_uncategorised=True)
    links = produced_links(st, ids, k)
    same, uniq = [], []
    for pid in ids:
        lst = links[pid]
        if not lst:
            continue
        cats = [part.get(r, "__singleton__" + r) for r in lst]
        same.append(sum(1 for c in cats if c == part.get(pid)) / len(cats))
        uniq.append(len(set(cats)) / len(cats))
    return {"same_category_frac": float(np.mean(same)) if same else float("nan"),
            "unique_category_frac": float(np.mean(uniq)) if uniq else float("nan"),
            "n_docs_with_links": len(same),
            "mean_links_per_doc": float(np.mean([len(links[i]) for i in ids])) if ids else float("nan")}


# --------------------------------------------------------------------------
# Link validity (LLM arms)
# --------------------------------------------------------------------------
def link_validity(state: WikiState | Any, corpus_titles: dict[str, str] | None = None,
                  ids: Sequence[str] | None = None, fuzzy_cutoff: float = 0.9) -> dict[str, Any]:
    """Classify every raw link target the arm wrote as resolved / near-miss / dangling.

    Resolution targets are the ids and titles of the pages that exist in the
    produced wiki (pages whose ``status`` is ``"missing"`` -- e.g. a failed
    arm-B ingest -- are not targets), or ``corpus_titles`` when given.  Self
    references and repeated targets are counted separately and excluded from
    the three classes.
    """
    st = coerce_state(state)
    ids = list(ids) if ids is not None else st.ids()
    if corpus_titles is not None:
        titles = dict(corpus_titles)
    else:  # pages that exist in the produced wiki (a failed arm-B ingest has no page)
        titles = {i: p.title for i, p in st.pages.items() if p.extra.get("status") != "missing"}
    index = TitleIndex(titles, cutoff=fuzzy_cutoff)
    id_norm = {norm_title(i): i for i in titles}
    counts = Counter()
    per_doc: dict[str, dict[str, int]] = {}
    for pid in ids:
        page = st.pages.get(pid)
        raw = list(page.related_raw) if page is not None else []
        if page is not None and not raw and page.related:
            raw = list(page.related)
        seen: set[str] = set()
        d = Counter()
        for r in raw:
            key = norm_title(r)
            if r == pid or key == norm_title(pid) or key == norm_title(titles.get(pid, "")):
                d["self"] += 1
                continue
            if key in seen:
                d["duplicate"] += 1
                continue
            seen.add(key)
            if r in titles or key in id_norm:
                d["resolved"] += 1
                continue
            cls, _ = index.classify(r)
            d[cls] += 1
        d["total"] = len(raw)
        per_doc[pid] = dict(d)
        counts.update(d)
    n_targets = counts["resolved"] + counts["near_miss"] + counts["dangling"]
    out: dict[str, Any] = {k: int(counts[k]) for k in ("resolved", "near_miss", "dangling", "self", "duplicate", "total")}
    for k in ("resolved", "near_miss", "dangling"):
        out[k + "_frac"] = counts[k] / n_targets if n_targets else float("nan")
    out["n_docs"] = len(ids)
    out["n_docs_without_links"] = sum(1 for pid in ids if per_doc[pid].get("total", 0) == 0)
    out["fuzzy_cutoff"] = fuzzy_cutoff
    out["per_document"] = per_doc
    return out


def evaluate_links(state: WikiState | Any, gold: Gold, k: int = 5, ids: Sequence[str] | None = None,
                   embeddings: dict[str, np.ndarray] | None = None, corpus_titles: dict[str, str] | None = None,
                   seed: int = 0, n_random: int = 100) -> dict[str, Any]:
    """Every link metric applicable to the gold at hand."""
    st = coerce_state(state)
    out: dict[str, Any] = {"k": k}
    if gold.has_links():
        out["hyperlinks"] = hyperlink_metrics(st, gold, k=k, ids=ids, n_random=n_random, seed=seed)
    for lvl in gold.levels:
        out["same_class"] = out.get("same_class", {})
        out["same_class"][lvl] = same_class_precision(st, gold, level=lvl, k=k, ids=ids)
    if embeddings:
        out["diversity"] = intra_list_diversity(st, embeddings, ids=ids, k=k)
    out["category_mix"] = category_mix(st, ids=ids, k=k)
    out["validity"] = link_validity(st, corpus_titles=corpus_titles, ids=ids)
    return out
