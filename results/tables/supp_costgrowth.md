| run | model | ingests_total | fit_at_ingest | intercept_usd | slope_usd_per_100_pages | pages_per_ingest | mean_usd_per_ingest | last_pages | last_usd | total_usd |
|---|---|---|---|---|---|---|---|---|---|---|
| DBPedia, order 1 | claude-sonnet-5 | 62 | 20 | 0.163 | 0.099 | 7.5 | 0.230 | 469 | 0.377 | 20.21 |
| DBPedia, order 2 | claude-sonnet-5 | 30 | 20 | 0.182 | 0.076 | 8.2 | 0.244 | 235 | 0.250 | 8.23 |
| WikiCS | claude-sonnet-5 | 32 | 20 | 0.199 | 0.072 | 5.4 | 0.240 | 146 | 0.224 | 8.48 |

Table: LLM-wiki (arm B) per-ingest cost as the wiki grows: the pre-registered stop-rule fit at ingest 20 (cost = intercept + slope × pages) for three independent wikis, and the last ingest each run reached before it was paused. These runs pre-date the 20-document ceiling and are supplementary. {#tbl_supp_costgrowth}
