# Paired Split Analysis: SetFit-HN-50 vs SetFit

| Dataset | Measure | Pairs | Baseline avg | Candidate avg | Mean delta | Delta std | Win/Tie/Loss | paired t-test p | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SentEval-CR | accuracy | 10 | 88.88 | 88.79 | -0.09 | 1.21 | 3/0/7 | 0.8139 | 0.5566 |
| ag_news | accuracy | 10 | 82.97 | 83.43 | +0.46 | 1.49 | 6/0/4 | 0.3562 | 0.5566 |
| amazon_counterfactual_en | matthews_correlation | 10 | 40.30 | 44.88 | +4.58 | 3.59 | 10/0/0 | 0.0030 | 0.0020 |
| emotion | accuracy | 10 | 49.05 | 49.98 | +0.92 | 3.86 | 6/0/4 | 0.4699 | 0.3750 |
| enron_spam | accuracy | 10 | 90.14 | 90.06 | -0.07 | 1.32 | 6/0/4 | 0.8613 | 0.9219 |
| sst5 | accuracy | 10 | 44.11 | 43.47 | -0.64 | 1.94 | 5/0/5 | 0.3212 | 0.4316 |
| Average | accuracy | 60 | 65.91 | 66.77 | +0.86 | 2.95 | 36/0/24 | 0.0280 | 0.0411 |

Notes:
- Scores are paired by identical dataset and split folder, such as `train-8-0`.
- Positive mean delta means the candidate method outperformed the baseline on average.
- p-values are descriptive because each dataset has only 10 paired splits.
