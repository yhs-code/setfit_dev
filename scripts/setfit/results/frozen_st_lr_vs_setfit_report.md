# Frozen ST + LR vs SetFit

Both methods use the same test benchmark datasets and the same few-shot split generation. Scores are percentages; Amazon uses Matthews correlation scaled by 100.

## Average By Shot

| Shot | Frozen ST + LR | SetFit | SetFit - Frozen |
| --- | --- | --- | --- |
| 2 | 46.9 +/- 5.5 | 49.7 +/- 6.8 | 2.8 |
| 4 | 51.8 +/- 5.3 | 58.0 +/- 6.3 | 6.2 |
| 8 | 57.6 +/- 3.2 | 65.9 +/- 4.6 | 8.3 |
| 16 | 61.2 +/- 2.5 | 70.5 +/- 2.7 | 9.3 |

## Per-dataset Comparison

| Dataset | Measure | Shot | Frozen ST + LR | SetFit | SetFit - Frozen |
| --- | --- | --- | --- | --- | --- |
| sst5 | accuracy | 2 | 31.0 +/- 2.1 | 33.0 +/- 2.9 | 2.0 |
| sst5 | accuracy | 4 | 33.9 +/- 1.9 | 39.1 +/- 3.0 | 5.2 |
| sst5 | accuracy | 8 | 36.8 +/- 2.1 | 44.1 +/- 2.3 | 7.3 |
| sst5 | accuracy | 16 | 38.7 +/- 2.0 | 46.4 +/- 1.6 | 7.7 |
| emotion | accuracy | 2 | 28.9 +/- 6.0 | 29.2 +/- 6.4 | 0.3 |
| emotion | accuracy | 4 | 32.4 +/- 3.4 | 37.5 +/- 4.2 | 5.1 |
| emotion | accuracy | 8 | 37.8 +/- 3.0 | 49.1 +/- 3.6 | 11.3 |
| emotion | accuracy | 16 | 41.6 +/- 2.9 | 59.5 +/- 3.0 | 17.9 |
| enron_spam | accuracy | 2 | 76.3 +/- 7.8 | 76.3 +/- 7.1 | 0.0 |
| enron_spam | accuracy | 4 | 85.2 +/- 5.0 | 87.1 +/- 4.1 | 1.9 |
| enron_spam | accuracy | 8 | 90.4 +/- 1.8 | 90.1 +/- 3.4 | -0.3 |
| enron_spam | accuracy | 16 | 92.9 +/- 1.1 | 92.7 +/- 2.1 | -0.2 |
| ag_news | accuracy | 2 | 57.6 +/- 4.1 | 63.9 +/- 5.4 | 6.3 |
| ag_news | accuracy | 4 | 65.9 +/- 3.0 | 76.7 +/- 4.1 | 10.8 |
| ag_news | accuracy | 8 | 74.3 +/- 2.8 | 83.0 +/- 3.2 | 8.7 |
| ag_news | accuracy | 16 | 79.6 +/- 1.1 | 85.2 +/- 1.3 | 5.6 |
| amazon_counterfactual_en | matthews_correlation | 2 | 14.7 +/- 5.1 | 19.0 +/- 8.3 | 4.3 |
| amazon_counterfactual_en | matthews_correlation | 4 | 17.9 +/- 9.4 | 25.6 +/- 13.6 | 7.7 |
| amazon_counterfactual_en | matthews_correlation | 8 | 22.2 +/- 7.3 | 40.3 +/- 13.5 | 18.1 |
| amazon_counterfactual_en | matthews_correlation | 16 | 28.6 +/- 6.8 | 50.7 +/- 6.9 | 22.1 |
| SentEval-CR | accuracy | 2 | 72.8 +/- 7.7 | 76.7 +/- 10.5 | 3.9 |
| SentEval-CR | accuracy | 4 | 75.7 +/- 9.1 | 81.7 +/- 9.0 | 6.0 |
| SentEval-CR | accuracy | 8 | 84.2 +/- 2.3 | 88.9 +/- 1.4 | 4.7 |
| SentEval-CR | accuracy | 16 | 85.9 +/- 1.3 | 88.8 +/- 1.4 | 2.9 |

## Interpretation

- SetFit consistently improves over Frozen ST + LR at every shot size on the average score.
- The average gain is largest at 16-shot in this run, which supports using the contrastive fine-tuning stage as the core method contribution.
- Frozen ST + LR is still strong on easier binary tasks such as Enron spam, so it is a useful lower-cost baseline rather than a weak strawman.
