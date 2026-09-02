| Corpus | Documents | Gold classes | System | Topics | NMI | ARI | P1 | Links P@5 | Docs with links |
|---|---|---|---|---|---|---|---|---|---|
| DBPedia, order 1 | 62 | 18 | LLM wiki | 44 | 0.851 | 0.370 | 0.704 | 0.226 | 0.27 |
| DBPedia, order 1 | 62 | 18 | Pipeline | 18 | 0.766 | 0.382 | 0.682 | 0.306 | 1.00 |
| DBPedia, order 2 | 30 | 16 | LLM wiki | 22 | 0.942 | 0.657 | 0.889 | 0.267 | 0.30 |
| DBPedia, order 2 | 30 | 16 | Pipeline | 16 | 0.906 | 0.584 | 0.853 | 0.200 | 1.00 |
| WikiCS, order 1 | 32 | 9 | LLM wiki | 10 | 0.870 | 0.738 | 0.859 | 0.362 | 0.81 |
| WikiCS, order 1 | 32 | 9 | Pipeline | 9 | 0.733 | 0.468 | 0.744 | 0.656 | 1.00 |

Table: The two systems on the same documents, at the largest scale the interrupted LLM-wiki runs reached. The pipeline is the mean over five seeds at the gold number of topics. No LLM call was made for this table: the wikis were extracted from the runs already recorded. {#tbl_headtohead}
