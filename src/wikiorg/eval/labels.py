"""Label quality: judge-embedding similarity to the gold class name, token
Jaccard, calibration floor, and NPMI topic coherence of c-TF-IDF terms.

Judge model
-----------
``label_sim`` embeds the produced label and the *de-CamelCased gold name of the
cluster's majority class* (e.g. ``EducationalInstitution`` -> "educational
institution") with ``sentence-transformers/all-mpnet-base-v2`` (Reimers &
Gurevych 2019, *Sentence-BERT*; MPNet: Song et al. 2020).  This is deliberately
a **different model family** from the pipeline's ``nomic-embed-text-v1.5``: the
same encoder that produced the clusters must not also be the judge of their
names, otherwise a label that merely sits close to the centroid in nomic space
would be rewarded for circular reasons.  The judge is fixed in
``configs/prereg.yaml`` (``judge_embedding_model``).

Scores
------
* ``label_sim`` per cluster = cosine(mpnet(label), mpnet(gold name of majority class)).
  Reported size-weighted (= mean over documents, the pre-registered E2 unit) and
  unweighted (mean over clusters).  Documents without a category (singletons,
  no label) contribute 0 to the document-level mean; ``n_uncategorised`` says
  how many.
* Calibration floor: the same cosine against a *random other* gold class name
  (mean of ``n_floor_draws`` seeded draws per cluster) and against the mean over
  all other class names.  A label that is only as similar to the majority class
  as to a random class is uninformative.
* Token Jaccard between the label's word set and the gold name's word set (a
  lexical, embedding-free counterpart).
* NPMI coherence (Bouma 2009, *Normalized (pointwise) mutual information in
  collocation extraction*; Röder, Both & Hinneburg 2015, *Exploring the space
  of topic coherence measures*, WSDM) of the top-10 c-TF-IDF terms per category
  over a boolean sliding window of 10 tokens on the evaluation corpus.  Arms
  A/A+ only (the LLM arms produce no term lists).  A multi-word term counts as
  present in a window when all of its tokens occur in that window.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .categories import majority_class
from .state import Gold, WikiState, coerce_state, common_ids

JUDGE_MODEL = "sentence-transformers/all-mpnet-base-v2"
EmbedFn = Callable[[Sequence[str]], np.ndarray]

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # CountVectorizer's default token pattern
_CAMEL_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def de_camel(name: str) -> str:
    """``EducationalInstitution`` -> ``educational institution``; ``USPresident``
    -> ``us president``; underscores/hyphens become spaces."""
    s = str(name or "").replace("_", " ").replace("-", " ")
    s = _CAMEL_1.sub(" ", s)
    s = _CAMEL_2.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def label_tokens(s: str) -> set[str]:
    return set(tokenize(s))


def token_jaccard(a: str, b: str) -> float:
    ta, tb = label_tokens(a), label_tokens(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --------------------------------------------------------------------------
# Judge embedder
# --------------------------------------------------------------------------
class JudgeEmbedder:
    """Lazy ``SentenceTransformer`` wrapper returning unit-norm vectors, with a
    per-text cache so labels and class names are only encoded once."""

    def __init__(self, model_name: str = JUDGE_MODEL, device: str | None = None,
                 batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self):
        if self._model is None:
            import os
            try:
                from sentence_transformers import SentenceTransformer  # lazy: heavy import
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("sentence-transformers is required for the judge embedder") from e
            device = self.device or os.environ.get("WIKIORG_JUDGE_DEVICE") or None
            try:
                self._model = SentenceTransformer(self.model_name, device=device)
            except Exception as e:  # CUDA OOM on the shared 4 GB GPU -> CPU (results are device-independent)
                if device == "cpu" or "out of memory" not in str(e).lower():
                    raise
                self._model = SentenceTransformer(self.model_name, device="cpu")
                self.device = "cpu"
        return self._model

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        try:
            return self._encode(texts)
        except Exception as e:
            if self.device == "cpu" or "out of memory" not in str(e).lower():
                raise
            self._model = None
            self.device = "cpu"
            return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        todo = sorted({t for t in texts if t not in self._cache})
        if todo:
            model = self._load()
            vecs = model.encode(todo, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
            for t, v in zip(todo, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float64)
        if not texts:
            return np.zeros((0, 0))
        return np.vstack([self._cache[t] for t in texts])


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (unit norm).  A stand-in for
    tests and offline runs -- NOT the judge used in the thesis."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = _stable_hash(tok, self.seed)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def _stable_hash(tok: str, seed: int) -> int:
    h = 2166136261 ^ seed
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", unit(a), unit(b))


# --------------------------------------------------------------------------
# Label similarity
# --------------------------------------------------------------------------
@dataclass
class ClusterLabelRow:
    cid: str
    label: str
    size: int
    majority_class: str | None
    majority_count: int
    gold_name: str
    members: list[str]


def cluster_label_table(state: WikiState | Any, gold: Gold, level: str = "l2",
                        ids: Sequence[str] | None = None) -> list[ClusterLabelRow]:
    """One row per non-empty produced category (singletons excluded)."""
    st = coerce_state(state)
    ids = list(ids) if ids is not None else common_ids(st, gold, level)
    id_set = set(ids)
    gl = gold.labels[level]
    rows: list[ClusterLabelRow] = []
    for cid, members in st.members(include_singletons=False).items():
        members = [m for m in members if m in id_set]
        if not members:
            continue
        cls, n, _ = majority_class(members, gl)
        cat = st.categories.get(cid)
        label = cat.label if cat is not None else ""
        if not label and cat is not None and cat.terms:
            label = ", ".join(cat.terms[:3])
        rows.append(ClusterLabelRow(cid=cid, label=label or "", size=len(members),
                                    majority_class=cls, majority_count=n,
                                    gold_name=de_camel(cls) if cls else "", members=members))
    return rows


def label_similarity(state: WikiState | Any, gold: Gold, level: str = "l2",
                     embed_fn: EmbedFn | None = None, ids: Sequence[str] | None = None,
                     seed: int = 0, n_floor_draws: int = 20) -> dict[str, Any]:
    """Judge-embedding similarity of produced labels to gold class names.

    Returns weighted/unweighted means, the calibration floors, token Jaccard,
    per-cluster rows and the per-document vector (``per_document``: id -> value;
    uncategorised documents get 0).
    """
    st = coerce_state(state)
    ids = list(ids) if ids is not None else common_ids(st, gold, level)
    embed = embed_fn or JudgeEmbedder()
    rows = cluster_label_table(st, gold, level, ids)
    classes = sorted(set(gold.classes(level, ids)))
    class_names = [de_camel(c) for c in classes]
    class_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)

    result: dict[str, Any] = {
        "judge_model": getattr(embed, "model_name", type(embed).__name__),
        "level": level, "n_clusters": len(rows), "n_classes": len(classes),
        "per_cluster": [], "per_document": {},
    }
    if not rows:
        for k in ("weighted_mean", "unweighted_mean", "floor_random_weighted",
                  "floor_random_unweighted", "floor_all_other_unweighted",
                  "jaccard_weighted", "jaccard_unweighted"):
            result[k] = float("nan")
        result["n_uncategorised"] = len(ids)
        result["per_document"] = {i: 0.0 for i in ids}
        return result

    E_lab = unit(embed([r.label for r in rows]))
    E_cls = unit(embed(class_names)) if class_names else np.zeros((0, E_lab.shape[1]))
    S = E_lab @ E_cls.T  # clusters x classes
    sims, floors_r, floors_all, jacc, sizes = [], [], [], [], []
    for i, r in enumerate(rows):
        j = class_idx.get(r.majority_class) if r.majority_class is not None else None
        sim = float(S[i, j]) if j is not None else 0.0
        others = [c for c in range(len(classes)) if c != j]
        if others:
            draws = rng.choice(others, size=n_floor_draws, replace=True)
            floor_r = float(np.mean(S[i, draws]))
            floor_all = float(np.mean(S[i, others]))
        else:
            floor_r = floor_all = float("nan")
        jc = token_jaccard(r.label, r.gold_name)
        sims.append(sim)
        floors_r.append(floor_r)
        floors_all.append(floor_all)
        jacc.append(jc)
        sizes.append(r.size)
        result["per_cluster"].append({
            "cid": r.cid, "label": r.label, "size": r.size, "majority_class": r.majority_class,
            "majority_count": r.majority_count, "gold_name": r.gold_name, "label_sim": sim,
            "floor_random": floor_r, "floor_all_other": floor_all, "jaccard": jc,
        })
        for m in r.members:
            result["per_document"][m] = sim
    w = np.asarray(sizes, dtype=float)
    sims_a, fr_a, fa_a, jc_a = map(np.asarray, (sims, floors_r, floors_all, jacc))
    n_uncat = sum(1 for i in ids if i not in result["per_document"])
    for i in ids:
        result["per_document"].setdefault(i, 0.0)
    doc_vals = np.array([result["per_document"][i] for i in ids])
    result.update({
        "weighted_mean": float(doc_vals.mean()),           # = size-weighted incl. uncategorised as 0
        "weighted_mean_categorised": float(np.average(sims_a, weights=w)),
        "unweighted_mean": float(sims_a.mean()),
        "floor_random_weighted": float(np.average(np.nan_to_num(fr_a), weights=w)),
        "floor_random_unweighted": float(np.nanmean(fr_a)),
        "floor_all_other_unweighted": float(np.nanmean(fa_a)),
        "jaccard_weighted": float(np.average(jc_a, weights=w)),
        "jaccard_unweighted": float(jc_a.mean()),
        "n_uncategorised": n_uncat,
        "n_docs": len(ids),
    })
    return result


# --------------------------------------------------------------------------
# NPMI coherence
# --------------------------------------------------------------------------
def _term_windows(tokens: list[str], term_tokens: dict[str, list[str]], window: int) -> dict[str, np.ndarray]:
    """Boolean window-membership vector for each term present in ``tokens``."""
    L = len(tokens)
    n_win = max(1, L - window + 1)
    positions: dict[str, list[int]] = defaultdict(list)
    needed = {t for toks in term_tokens.values() for t in toks}
    for i, tok in enumerate(tokens):
        if tok in needed:
            positions[tok].append(i)
    uni: dict[str, np.ndarray] = {}
    for tok, pos in positions.items():
        v = np.zeros(n_win, dtype=bool)
        for p in pos:
            lo = max(0, p - window + 1)
            hi = min(p, n_win - 1)
            v[lo:hi + 1] = True
        uni[tok] = v
    out: dict[str, np.ndarray] = {}
    for term, toks in term_tokens.items():
        if all(t in uni for t in toks):
            v = uni[toks[0]].copy()
            for t in toks[1:]:
                v &= uni[t]
            if v.any():
                out[term] = v
    return out


def npmi_coherence(term_lists: dict[str, Sequence[str]], texts: Iterable[str], window: int = 10,
                   top_n: int = 10, sizes: dict[str, int] | None = None,
                   eps: float = 1e-12) -> dict[str, Any]:
    """Mean NPMI over all term pairs of each category's top-``top_n`` terms.

    ``texts`` are the corpus documents (any iterable of strings); co-occurrence
    is counted over boolean sliding windows of ``window`` tokens (a document
    shorter than the window is one window).  Pairs that never co-occur score -1.
    Returns per-category means, the unweighted mean over categories and, if
    ``sizes`` (category -> n docs) is given, the size-weighted mean.
    """
    lists = {cid: [str(t) for t in terms][:top_n] for cid, terms in term_lists.items() if terms}
    if not lists:
        return {"mean": float("nan"), "weighted_mean": float("nan"), "per_category": {},
                "window": window, "top_n": top_n, "n_windows": 0}
    term_tokens = {t: tokenize(t) for terms in lists.values() for t in terms}
    term_tokens = {t: toks for t, toks in term_tokens.items() if toks}
    pairs: set[tuple[str, str]] = set()
    for terms in lists.values():
        ts = [t for t in terms if t in term_tokens]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                pairs.add((ts[i], ts[j]) if ts[i] <= ts[j] else (ts[j], ts[i]))
    pair_by_term: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for a, b in pairs:
        pair_by_term[a].append((a, b))
    n_windows = 0
    c_term: Counter = Counter()
    c_pair: Counter = Counter()
    for text in texts:
        toks = tokenize(text)
        n_win = max(1, len(toks) - window + 1)
        n_windows += n_win
        present = _term_windows(toks, term_tokens, window)
        for t, v in present.items():
            c_term[t] += int(v.sum())
        for a in present:
            for (x, y) in pair_by_term.get(a, ()):
                if y in present:
                    c_pair[(x, y)] += int((present[x] & present[y]).sum())
    W = max(n_windows, 1)

    def npmi(a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        cab = c_pair.get(key, 0)
        if cab == 0:
            return -1.0
        if cab >= W:  # the pair is in every window: perfect association (0/0 limit)
            return 1.0
        pa, pb, pab = c_term[a] / W, c_term[b] / W, cab / W
        return float(np.log(pab / (pa * pb) + eps) / (-np.log(pab + eps)))

    per_cat: dict[str, float] = {}
    for cid, terms in lists.items():
        ts = [t for t in terms if t in term_tokens]
        vals = [npmi(ts[i], ts[j]) for i in range(len(ts)) for j in range(i + 1, len(ts))]
        per_cat[cid] = float(np.mean(vals)) if vals else float("nan")
    vals = np.array([v for v in per_cat.values() if not np.isnan(v)])
    out: dict[str, Any] = {"per_category": per_cat, "window": window, "top_n": top_n,
                           "n_windows": int(W), "n_categories": len(per_cat),
                           "mean": float(vals.mean()) if len(vals) else float("nan")}
    if sizes:
        ws = np.array([sizes.get(c, 0) for c, v in per_cat.items() if not np.isnan(v)], dtype=float)
        out["weighted_mean"] = float(np.average(vals, weights=ws)) if len(vals) and ws.sum() > 0 else float("nan")
    else:
        out["weighted_mean"] = out["mean"]
    return out


def evaluate_labels(state: WikiState | Any, gold: Gold, level: str = "l2", texts: dict[str, str] | None = None,
                    embed_fn: EmbedFn | None = None, ids: Sequence[str] | None = None,
                    seed: int = 0, window: int = 10) -> dict[str, Any]:
    """Full label block: similarity + Jaccard + floors, and NPMI when the state
    carries c-TF-IDF terms (arms A/A+) and corpus texts are supplied."""
    st = coerce_state(state)
    out = label_similarity(st, gold, level=level, embed_fn=embed_fn, ids=ids, seed=seed)
    term_lists = {cid: c.terms for cid, c in st.categories.items() if c.terms}
    if term_lists and texts:
        sizes = {cid: len(m) for cid, m in st.members().items()}
        out["npmi"] = npmi_coherence(term_lists, texts.values(), window=window, sizes=sizes)
    else:
        out["npmi"] = None
    return out
