| corpus | N | k | seeds | nmi | p1 | same_class_p5 | hyperlink_p5 | usd |
|---|---|---|---|---|---|---|---|---|
| dbpedia | 20 | 13 | 5 | 0.895 ± 0.014 | 0.830 | 0.120 | n/a | 0.00015 |
| dbpedia | 100 | 18 | 5 | 0.744 ± 0.029 | 0.670 | 0.428 | n/a | 0.00075 |
| dbpedia | 300 | 40 | 5 | 0.763 ± 0.005 | 0.633 | 0.476 | n/a | 0.00223 |
| dbpedia | 1000 | 70 | 5 | 0.726 ± 0.006 | 0.567 | 0.586 | n/a | 0.00741 |
| dbpedia | 2000 | 70 | 5 | 0.691 ± 0.006 | 0.541 | 0.667 | n/a | 0.01882 |
| dbpedia_s | 20 | 4 | 5 | 0.434 ± 0.058 | 0.619 | 0.390 | n/a | 0.00015 |
| wikics | 100 | 10 | 5 | 0.710 ± 0.032 | 0.723 | 0.642 | 0.866 | 0.00307 |
| wikics | 500 | 10 | 5 | 0.766 ± 0.021 | 0.813 | 0.818 | 0.881 | 0.01697 |
| wikics | 1000 | 10 | 5 | 0.662 ± 0.010 | 0.766 | 0.773 | 0.744 | 0.03067 |

Table: Pipeline (arm A) against ground truth at every corpus size: oracle k, headline λ = 0.7, mean over seeds (± sd for NMI). Hyperlink precision@5 exists only for WikiCS, whose links are real Wikipedia hyperlinks. Cost is the cloud-equivalent GPU price of one run. {#tbl_supp_armA}
