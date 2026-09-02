"""Class-based TF-IDF (c-TF-IDF) category labels, per Grootendorst (2022, arXiv:2203.05794).

Weighting exactly as BERTopic's ``ClassTfidfTransformer`` (no BM25, no frequent-word reduction):

    W[t, c] = tf[t, c] · log(1 + A / f_t)

with ``tf[t, c]`` the L1-normalised count of term ``t`` in the concatenation of the documents of
class ``c``, ``f_t`` the raw count of ``t`` over all classes, and ``A = int(mean words per class)``
(integer truncation as in the reference implementation).

Vocabulary: ``CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2)``.  The
vocabulary is fitted on the *individual documents* (so ``min_df=2`` means "in at least two
documents") and then applied to the class-concatenated documents.  Fitting on the concatenated
class documents, as BERTopic's code path does, would make ``min_df=2`` mean "in at least two
classes" and delete exactly the class-specific terms.  The paper defines only the weighting, so
this reading is the one that preserves discriminative terms.

Label = the top-3 terms joined by ", "; the top-10 terms are kept for the wiki and for NPMI.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

TOP_TERMS = 10
LABEL_TERMS = 3


def ctfidf_matrix(docs: list[str], labels, ngram_range=(1, 2), stop_words="english", min_df=2,
                  vectorizer: CountVectorizer | None = None) -> tuple[list, np.ndarray, sparse.csr_matrix, dict]:
    """Returns ``(classes, terms, W, info)`` with ``W`` of shape (n_classes, n_terms)."""
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    vec = vectorizer or CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, min_df=min_df)
    vec.fit(docs)
    class_docs = [" ".join(d for d, lab in zip(docs, labels) if lab == c) for c in classes]
    X = vec.transform(class_docs).astype(np.float64)              # raw counts per class
    f_t = np.asarray(X.sum(axis=0)).ravel()                        # term frequency over all classes
    words_per_class = np.asarray(X.sum(axis=1)).ravel()
    A = int(words_per_class.mean()) if X.shape[0] else 0            # int truncation as in BERTopic
    with np.errstate(divide="ignore"):
        idf = np.log(1.0 + A / np.where(f_t > 0, f_t, np.inf))
    tf = normalize(X, norm="l1", axis=1)
    W = sparse.csr_matrix(tf.multiply(idf))
    terms = np.asarray(vec.get_feature_names_out())
    info = {"n_terms": int(len(terms)), "A": A, "ngram_range": list(ngram_range), "stop_words": stop_words,
            "min_df": min_df}
    return classes, terms, W, info


def top_terms(classes: list, terms: np.ndarray, W: sparse.csr_matrix, top_n: int = TOP_TERMS) -> dict:
    """{class: [(term, weight), ...]} by decreasing weight (ties: alphabetical)."""
    out = {}
    for row, c in enumerate(classes):
        r = W.getrow(row)
        if r.nnz == 0:
            out[c] = []
            continue
        idx = r.indices
        w = r.data
        order = sorted(range(len(idx)), key=lambda j: (-w[j], str(terms[idx[j]])))[:top_n]
        out[c] = [(str(terms[idx[j]]), float(w[j])) for j in order]
    return out


def ctfidf_terms(docs: list[str], labels, top_n: int = TOP_TERMS, **kwargs) -> tuple[dict, dict]:
    classes, terms, W, info = ctfidf_matrix(docs, labels, **kwargs)
    return top_terms(classes, terms, W, top_n=top_n), info


def make_label(terms: list, n: int = LABEL_TERMS, fallback: str = "untitled") -> str:
    words = [t[0] if isinstance(t, (tuple, list)) else str(t) for t in terms[:n]]
    return ", ".join(words) if words else fallback
