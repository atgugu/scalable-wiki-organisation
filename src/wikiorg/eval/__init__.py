"""Evaluation, statistics and the ledger (workstream W3).

Modules
-------
state        WikiState mirror of the ``wiki_state.json`` contract; gold loading.
categories   NMI / ARI / AMI / V-measure family / purity / inverse purity / P1; random baseline.
labels       judge-embedding label similarity (mpnet), calibration floors, token Jaccard, NPMI.
links        hyperlink P@k / R@k / MAP with oracle and random baselines; same-class precision;
             intra-list diversity; category mix; link validity.
structure    graph statistics (networkx) with the Kamps & Koolen (2009) reference row.
cost         price table, token -> USD, reconciliation, per-run aggregates, marginal-cost fit.
stability    pairwise ARI / NMI between repeated runs.
human        rating-sheet construction and scoring (Krippendorff's alpha, Spearman, Cliff's delta).
judge        LLM-judge agreement with the human raters (no LLM calls here).
bootstrap    BCa bootstrap, 80 % subsampling for partition metrics, effect sizes, Holm, ``decide``.
suite        ``evaluate_run`` (all metric blocks for one state) and ``compare_arms`` (E1-E3).
report       append-only ledger, run-directory evaluation, tables T1-T6 and figure CSVs.
figures      F2 Pareto, F3 MMR ablation, F4 marginal cost, k-sensitivity (300 dpi, 15 cm).

Heavy imports (sentence-transformers, matplotlib) are deferred to the functions that need them.
"""
from .state import Gold, WikiState, coerce_state, gold_from_rows, read_gold  # noqa: F401
