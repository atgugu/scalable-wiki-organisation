| group | runs | pairs | ari_mean | ari_min | nmi_mean | name_exact | name_sim | link_jaccard | link_identical | with_links | topics_per_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline_same_seed | 2 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13;13 |
| pipeline_cross_seed | 5 | 10 | 0.922 | 0.870 | 0.983 | 0.862 | 0.941 | 1.000 | 1.000 | 1.000 | 13;13;13;13;13 |
| llm_wiki | 3 | 3 | 1.000 | 1.000 | 1.000 | 0.772 | 0.936 | 1.000 | 1.000 | 0.050 | 19;19;19 |
| one_shot_labeler_exploratory | 2 | 1 | 1.000 | 1.000 | 1.000 | 0.737 | 0.919 | 0.695 | 0.600 | 0.800 | 19;19 |

Table: Run-to-run agreement on the same 20 documents in the same order (unstructured set; LLM arms on Claude Sonnet 5). Pipeline links depend only on the embeddings, so their agreement carries no seed variance. {#tbl_repro_v2}
