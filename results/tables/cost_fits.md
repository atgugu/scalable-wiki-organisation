| run | fit | intercept_usd | slope_usd_per_100_pages | slope_ci_per_100 | r2 | last_pages | total_usd |
|---|---|---|---|---|---|---|---|
| DBPedia, order 1 | frozen at 20 | 0.163 | 0.099 | [0.058, 0.136] | 0.6 | 469 | 20.21 |
| DBPedia, order 1 | post-hoc, all 62 | 0.188 | 0.052 | [0.041, 0.064] | 0.5 | 469 | 20.21 |
| DBPedia, order 2 | frozen at 20 | 0.182 | 0.076 | [0.041, 0.117] | 0.43 | 235 | 8.23 |
| WikiCS | frozen at 20 | 0.199 | 0.072 | [−0.019, 0.138] | 0.1 | 146 | 8.48 |

Table: LLM-wiki ingest cost against wiki pages before the ingest: the pre-registered fits frozen at ingest 20 and the post-hoc refit on all 62 ingests of the first order. Slope in USD per 100 pages with a BCa bootstrap interval over ingests. {#tbl_cost_fits}
