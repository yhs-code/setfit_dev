# 8-shot SetFit-HN-50 vs SetFit Baseline

| Dataset | Measure | SetFit avg | SetFit std | HN-50 avg | HN-50 std | Avg delta | Std delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sst5 | accuracy | 44.1 | 2.3 | 43.5 | 3.4 | -0.6 | +1.1 |
| emotion | accuracy | 49.1 | 3.6 | 50.0 | 4.4 | +0.9 | +0.8 |
| enron_spam | accuracy | 90.1 | 3.4 | 90.1 | 3.2 | +0.0 | -0.2 |
| ag_news | accuracy | 83.0 | 3.2 | 83.4 | 2.4 | +0.4 | -0.8 |
| amazon_counterfactual_en | matthews_correlation | 40.3 | 13.5 | 44.9 | 13.3 | +4.6 | -0.2 |
| SentEval-CR | accuracy | 88.9 | 1.4 | 88.8 | 2.0 | -0.1 | +0.6 |
| Average | N/A | 65.9 | 4.6 | 66.8 | 4.8 | +0.9 | +0.2 |

Notes:
- HN-50 uses 50% hard negatives ranked by frozen encoder cosine similarity and 50% random negatives.
- Positive avg delta means HN-50 is higher than the original SetFit 8-shot baseline.
