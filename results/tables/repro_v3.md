| group | runs | pairs | ari_mean | ari_min | nmi_mean | name_exact | name_sim | link_jaccard | link_identical | with_links | topics_per_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline_same_seed | 2 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4;4 |
| pipeline_cross_seed | 5 | 10 | 0.477 | 0.187 | 0.606 | 0.600 | 0.804 | 1.000 | 1.000 | 1.000 | 4;4;4;4;4 |
| pipeline_auto_cross_seed | 5 | 10 | 0.795 | 0.693 | 0.928 | 0.650 | 0.860 | 1.000 | 1.000 | 1.000 | 8;8;8;8;8 |
| llm_wiki | 2 | 1 | 1.000 | 1.000 | 1.000 | 0.368 | 0.843 | 1.000 | 1.000 | 0.050 | 19;19 |
| one_shot_labeler | 2 | 1 | 0.384 | 0.384 | 0.914 | 0.412 | 0.826 | 0.613 | 0.400 | 1.000 | 12;17 |

Table: Run-to-run agreement on the structured 20-document set (4 classes x 5; LLM arms on Claude Haiku 4.5). {#tbl_repro_v3}
