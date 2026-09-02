"""c-TF-IDF on a hand-computed toy (Grootendorst 2022 weighting, BERTopic details)."""
import math

import numpy as np

from wikiorg.label import ctfidf_matrix, ctfidf_terms, make_label, top_terms

DOCS = ["apple banana apple", "apple cherry", "banana cherry cherry", "cherry dog"]
LABELS = [0, 0, 1, 1]


def test_hand_computed_weights_min_df_1():
    classes, terms, W, info = ctfidf_matrix(DOCS, LABELS, ngram_range=(1, 1), stop_words=None, min_df=1)
    assert classes == [0, 1]
    assert list(terms) == ["apple", "banana", "cherry", "dog"]
    # class counts: c0 = [3,1,1,0] (5 words), c1 = [0,1,3,1] (5 words); f_t = [3,2,4,1]; A = int(5) = 5
    assert info["A"] == 5
    idf = [math.log(1 + 5 / 3), math.log(1 + 5 / 2), math.log(1 + 5 / 4), math.log(1 + 5 / 1)]
    expected = np.array([[0.6 * idf[0], 0.2 * idf[1], 0.2 * idf[2], 0.0],
                         [0.0, 0.2 * idf[1], 0.6 * idf[2], 0.2 * idf[3]]])
    np.testing.assert_allclose(W.toarray(), expected, atol=1e-12)
    tt = top_terms(classes, terms, W, top_n=10)
    assert [t for t, _ in tt[0]] == ["apple", "banana", "cherry"]
    assert [t for t, _ in tt[1]] == ["cherry", "dog", "banana"]


def test_min_df_counts_documents_and_A_truncates():
    # min_df=2 over the individual documents drops "dog" (1 doc); apple/banana/cherry stay.
    classes, terms, W, info = ctfidf_matrix(DOCS, LABELS, ngram_range=(1, 1), stop_words=None, min_df=2)
    assert list(terms) == ["apple", "banana", "cherry"]
    # c0 = [3,1,1] (5 words), c1 = [0,1,3] (4 words) -> A = int(4.5) = 4 ; f_t = [3,2,4]
    assert info["A"] == 4
    idf = [math.log(1 + 4 / 3), math.log(1 + 4 / 2), math.log(1 + 4 / 4)]
    expected = np.array([[0.6 * idf[0], 0.2 * idf[1], 0.2 * idf[2]],
                         [0.0, 0.25 * idf[1], 0.75 * idf[2]]])
    np.testing.assert_allclose(W.toarray(), expected, atol=1e-12)


def test_default_vectorizer_and_labels():
    docs = ["the river flows to the sea and the river delta is wide"] * 3 + \
           ["the football club won the league and the club plays football"] * 3
    labels = [0, 0, 0, 1, 1, 1]
    terms, info = ctfidf_terms(docs, labels)
    assert info["ngram_range"] == [1, 2] and info["min_df"] == 2 and info["stop_words"] == "english"
    lab0, lab1 = make_label(terms[0]), make_label(terms[1])
    assert "river" in lab0 and "club" in lab1 or "football" in lab1
    assert len(lab0.split(", ")) == 3
    assert make_label([]) == "untitled"
    assert make_label([("a", 1.0), ("b", 0.5)]) == "a, b"
