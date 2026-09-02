| Documents | LLM wiki, add one | LLM wiki, minutes | LLM wiki, build all | Pipeline, rebuild all | Ratio, add one to rebuild all | Index lines | Beyond the fit |
|---|---|---|---|---|---|---|---|
| 100 | $0.59 | 2 | $39 | $0.0009 | 624x | 766 | yes |
| 250 | $1.19 | 4 | $172 | $0.0024 | 504x | 1,916 | yes |
| 500 | $2.18 | 7 | $594 | $0.0047 | 464x | 3,832 | yes |
| 1,000 | $4.18 | 12 | $2,187 | $0.0094 | 444x | 7,664 | yes |
| 5,000 | $20.16 | 54 | $50,869 | $0.0471 | 428x | 38,321 | yes |
| 10,000 | $40.12 | 106 | $201,576 | $0.0941 | 426x | 76,643 | yes |

Table: What the next document costs as a collection grows. The laws are fitted on all 62 measured ingests of the longest run, over a wiki of 0 to 469 pages: cost is $0.188 plus $0.000521 per page and the wiki grows 7.7 pages per document. Rows marked beyond the fit are illustrative, not predictions. The last column is the length of the index the agent re-reads on every ingest; its default file read stops at 2,000 lines, which the index passes at about 261 documents. {#tbl_scaling}
