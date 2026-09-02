| system | model | usd_total | usd_per_doc | wall_s_per_doc | note |
|---|---|---|---|---|---|
| Pipeline (A): structure only | none (GPU) | $0.00015 | $0.00001 | 0.1 | embeddings + spherical k-means + c-TF-IDF + kNN/MMR; cloud-equivalent GPU price |
| A+ : pipeline + LLM cluster names | claude-haiku-4-5 | $0.09889 | $0.00494 |  | one call per cluster (13) |
| Hybrid (H): pipeline structure + names + LLM-written pages | claude-haiku-4-5 | $0.47630 | $0.02382 | 28.4 | 19/20 pages written; one no-tools call per document |
| LLM wiki (B), first 20 ingests | claude-sonnet-5 | $4.59578 | $0.22979 | 92.0 | agentic ingest, one session per document |
| LLM wiki (B), replicate 2 | claude-sonnet-5 | $4.14203 | $0.20710 | 80.6 | agentic ingest, one session per document |
| LLM wiki (B), replicate 3 | claude-sonnet-5 | $4.93332 | $0.24667 | 96.0 | agentic ingest, one session per document |
| LLM wiki (B), structured set, replicate 1 | claude-haiku-4-5 | $2.27475 | $0.11374 | 106.4 | agentic ingest, one session per document |
| LLM wiki (B), structured set, replicate 2 | claude-haiku-4-5 | $2.24383 | $0.11219 | 104.8 | agentic ingest, one session per document |
| One-shot labeler (C), first 20 calls | claude-sonnet-5 | $0.14227 | $0.00711 | 4.1 | one no-tools call per document |
| One-shot labeler (C), structured set | claude-haiku-4-5 | $0.33130 | $0.01656 | 29.5 | one no-tools call per document |

Table: Cost and time to organise (and, where noted, write) a 20-document wiki. LLM costs are list-price equivalents recomputed from the CLI's token counts; the pipeline's is a cloud-equivalent GPU price. {#tbl_cost_20}
