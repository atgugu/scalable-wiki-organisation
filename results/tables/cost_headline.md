| System | Model | Total | Per document | Seconds per document |
|---|---|---|---|---|
| Pipeline, structure only | no model | $150 per million | $7 per million | 0 |
| Pipeline, with LLM-named topics | haiku-4-5 | $0.099 | $0.005 |  |
| Hybrid, with LLM-written pages | haiku-4-5 | $0.476 | $0.024 | 28 |
| LLM wiki | sonnet-5 | $4.60 | $0.230 | 92 |
| LLM wiki | haiku-4-5 | $2.27 | $0.114 | 106 |
| One-shot labeller | haiku-4-5 | $0.331 | $0.017 | 29 |

Table: Cost and time to organise a twenty-document collection, and where noted to write its pages. Costs are list-price equivalents recomputed from the token counts the tool reports; the pipeline's is a cloud-equivalent price for the GPU time it used. {#tbl_cost_headline}
