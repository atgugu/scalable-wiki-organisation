| System | Corpus | Model | Total | Per document | Seconds per document |
|---|---|---|---|---|---|
| Pipeline, structure only | DBPedia | none | $0.00015 | $0.0000075 | 0.08 |
| Pipeline, LLM names | DBPedia | Haiku 4.5 | $0.099 | $0.005 | – |
| Hybrid, LLM pages | DBPedia | Haiku 4.5 | $0.476 | $0.024 | 28 |
| LLM wiki | DBPedia | Sonnet 5 | $4.60 | $0.230 | 92 |
| LLM wiki | structured | Haiku 4.5 | $2.27 | $0.114 | 106 |
| One-shot labeller | structured | Haiku 4.5 | $0.331 | $0.017 | 29 |

Table: Cost and time to organise a twenty-document collection, and where noted to write its pages. Costs are list-price equivalents from the token counts the tool reports; the pipeline's is a cloud-equivalent price for the compute time it used. Corpus names the twenty-document set. {#tbl_cost_headline_corpus}
