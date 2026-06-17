# Test-only SetFit Ablation Summary

Scores are percentages. `amazon_counterfactual_en` uses Matthews correlation, also scaled by 100. Stability: stable <= 2 std, moderate <= 5 std, unstable > 5 std.

## Baseline 8-shot iter=20 LR

| dataset | measure | score | stability |
| --- | --- | --- | --- |
| sst5 | accuracy | 44.1 +/- 2.3 | moderate |
| emotion | accuracy | 49.1 +/- 3.6 | moderate |
| enron_spam | accuracy | 90.1 +/- 3.4 | moderate |
| ag_news | accuracy | 83.0 +/- 3.2 | moderate |
| amazon_counterfactual_en | matthews_correlation | 40.3 +/- 13.5 | unstable |
| SentEval-CR | accuracy | 88.9 +/- 1.4 | stable |
| Average | N/A | 65.9 +/- 4.6 | moderate |

## Shot-size ablation

| dataset | measure | 2-shot | 4-shot | 8-shot | 16-shot | 64-shot |
| --- | --- | --- | --- | --- | --- | --- |
| sst5 | accuracy | 33.0 +/- 2.9 | 39.1 +/- 3.0 | 44.1 +/- 2.3 | 46.4 +/- 1.6 | 51.8 +/- 0.5 |
| emotion | accuracy | 29.2 +/- 6.4 | 37.5 +/- 4.2 | 49.1 +/- 3.6 | 59.5 +/- 3.0 | 76.5 +/- 1.3 |
| enron_spam | accuracy | 76.3 +/- 7.1 | 87.1 +/- 4.1 | 90.1 +/- 3.4 | 92.7 +/- 2.1 | 95.9 +/- 0.5 |
| ag_news | accuracy | 63.9 +/- 5.4 | 76.7 +/- 4.1 | 83.0 +/- 3.2 | 85.2 +/- 1.3 | 88.1 +/- 0.6 |
| amazon_counterfactual_en | matthews_correlation | 19.0 +/- 8.3 | 25.6 +/- 13.6 | 40.3 +/- 13.5 | 50.7 +/- 6.9 | 62.0 +/- 2.8 |
| SentEval-CR | accuracy | 76.7 +/- 10.5 | 81.7 +/- 9.0 | 88.9 +/- 1.4 | 88.8 +/- 1.4 | 90.7 +/- 0.7 |
| Average | N/A | 49.7 +/- 6.8 | 58.0 +/- 6.3 | 65.9 +/- 4.6 | 70.5 +/- 2.7 | 77.5 +/- 1.1 |

## num_iterations ablation

| dataset | measure | iter=2 | iter=5 | iter=10 | iter=20 |
| --- | --- | --- | --- | --- | --- |
| sst5 | accuracy | 37.8 +/- 2.1 | 40.2 +/- 2.9 | 43.0 +/- 3.0 | 44.1 +/- 2.3 |
| emotion | accuracy | 39.4 +/- 3.2 | 43.6 +/- 2.5 | 47.2 +/- 4.2 | 49.1 +/- 3.6 |
| enron_spam | accuracy | 90.6 +/- 1.6 | 90.6 +/- 2.2 | 90.7 +/- 3.3 | 90.1 +/- 3.4 |
| ag_news | accuracy | 78.6 +/- 2.6 | 81.6 +/- 2.8 | 83.1 +/- 2.6 | 83.0 +/- 3.2 |
| amazon_counterfactual_en | matthews_correlation | 28.0 +/- 7.6 | 29.5 +/- 8.4 | 33.1 +/- 10.3 | 40.3 +/- 13.5 |
| SentEval-CR | accuracy | 85.6 +/- 2.8 | 87.8 +/- 2.4 | 89.3 +/- 1.2 | 88.9 +/- 1.4 |
| Average | N/A | 60.0 +/- 3.3 | 62.2 +/- 3.5 | 64.4 +/- 4.1 | 65.9 +/- 4.6 |

## Classifier-head ablation

| dataset | measure | LR | SVC-RBF | KNN |
| --- | --- | --- | --- | --- |
| sst5 | accuracy | 44.1 +/- 2.3 | 44.5 +/- 2.5 | 44.1 +/- 2.5 |
| emotion | accuracy | 49.1 +/- 3.6 | 48.6 +/- 4.4 | 49.2 +/- 3.7 |
| enron_spam | accuracy | 90.1 +/- 3.4 | 90.0 +/- 3.6 | 90.2 +/- 3.4 |
| ag_news | accuracy | 83.0 +/- 3.2 | 82.2 +/- 3.7 | 83.2 +/- 2.8 |
| amazon_counterfactual_en | matthews_correlation | 40.3 +/- 13.5 | 42.7 +/- 14.1 | 38.6 +/- 13.1 |
| SentEval-CR | accuracy | 88.9 +/- 1.4 | 88.8 +/- 1.5 | 89.0 +/- 1.3 |
| Average | N/A | 65.9 +/- 4.6 | 66.2 +/- 4.9 | 65.7 +/- 4.4 |

## Frozen ST + LR

| dataset | measure | 2-shot | 4-shot | 8-shot | 16-shot | 64-shot |
| --- | --- | --- | --- | --- | --- | --- |
| sst5 | accuracy | 31.0 +/- 2.1 | 33.9 +/- 1.9 | 36.8 +/- 2.1 | 38.7 +/- 2.0 | 42.2 +/- 1.6 |
| emotion | accuracy | 28.9 +/- 6.0 | 32.4 +/- 3.4 | 37.8 +/- 3.0 | 41.6 +/- 2.9 | 50.8 +/- 1.5 |
| enron_spam | accuracy | 76.3 +/- 7.8 | 85.2 +/- 5.0 | 90.4 +/- 1.8 | 92.9 +/- 1.1 | 95.4 +/- 0.6 |
| ag_news | accuracy | 57.6 +/- 4.1 | 65.9 +/- 3.0 | 74.3 +/- 2.8 | 79.6 +/- 1.1 | 84.4 +/- 0.7 |
| amazon_counterfactual_en | matthews_correlation | 14.7 +/- 5.1 | 17.9 +/- 9.4 | 22.2 +/- 7.3 | 28.6 +/- 6.8 | 40.8 +/- 2.1 |
| SentEval-CR | accuracy | 72.8 +/- 7.7 | 75.7 +/- 9.1 | 84.2 +/- 2.3 | 85.9 +/- 1.3 | 88.2 +/- 0.7 |
| Average | N/A | 46.9 +/- 5.5 | 51.8 +/- 5.3 | 57.6 +/- 3.2 | 61.2 +/- 2.5 | 66.9 +/- 1.2 |

## Main observations

- Shot size has the largest effect on the average SetFit score: 2-shot 49.7, 4-shot 58.0, 8-shot 65.9, 16-shot 70.5, 64-shot 77.5.
- 64-shot substantially reduces variance compared with low-shot settings.
- Classifier and num_iterations ablations remain 8-shot-only controls.
