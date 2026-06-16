# Test-only SetFit Ablation Summary

Scores are percentages. `amazon_counterfactual_en` uses Matthews correlation, also scaled by 100. Stability: stable <= 2 std, moderate <= 5 std, unstable > 5 std.

## Execution status

- main: 60 result files, summary: `/home/yanghongsheng/SetFit/setfit_dev/scripts/setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_main/summary_table.csv`
- shot: 180 result files, summary: `/home/yanghongsheng/SetFit/setfit_dev/scripts/setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_shot_ablation/summary_table.csv`
- iter2: 60 result files, summary: `/home/yanghongsheng/SetFit/setfit_dev/scripts/setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_2-batch_16-test_iter_2/summary_table.csv`
- iter5: 60 result files, summary: `/home/yanghongsheng/SetFit/setfit_dev/scripts/setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_5-batch_16-test_iter_5/summary_table.csv`
- iter10: 60 result files, summary: `/home/yanghongsheng/SetFit/setfit_dev/scripts/setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_10-batch_16-test_iter_10/summary_table.csv`
- svc: 60 result files, summary: `/home/yanghongsheng/SetFit/setfit_dev/scripts/setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-svc-rbf-iterations_20-batch_16-test_clf_svc_rbf_valid/summary_table.csv`
- knn: 60 result files, summary: `/home/yanghongsheng/SetFit/setfit_dev/scripts/setfit/results/paraphrase-mpnet-base-v2-CosineSimilarityLoss-knn-iterations_20-batch_16-test_clf_knn_valid/summary_table.csv`

Note: the shot ablation directory now contains only the valid 2/4/16-shot runs. The 8-shot column below comes from the complete baseline run in `test_main`.

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

| dataset | measure | 2-shot | 4-shot | 8-shot | 16-shot |
| --- | --- | --- | --- | --- | --- |
| sst5 | accuracy | 33.0 +/- 2.9 | 39.1 +/- 3.0 | 44.1 +/- 2.3 | 46.4 +/- 1.6 |
| emotion | accuracy | 29.2 +/- 6.4 | 37.5 +/- 4.2 | 49.1 +/- 3.6 | 59.5 +/- 3.0 |
| enron_spam | accuracy | 76.3 +/- 7.1 | 87.1 +/- 4.1 | 90.1 +/- 3.4 | 92.7 +/- 2.1 |
| ag_news | accuracy | 63.9 +/- 5.4 | 76.7 +/- 4.1 | 83.0 +/- 3.2 | 85.2 +/- 1.3 |
| amazon_counterfactual_en | matthews_correlation | 19.0 +/- 8.3 | 25.6 +/- 13.6 | 40.3 +/- 13.5 | 50.7 +/- 6.9 |
| SentEval-CR | accuracy | 76.7 +/- 10.5 | 81.7 +/- 9.0 | 88.9 +/- 1.4 | 88.8 +/- 1.4 |
| Average | N/A | 49.7 +/- 6.8 | 58.0 +/- 6.3 | 65.9 +/- 4.6 | 70.5 +/- 2.7 |

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

| dataset | measure | logistic_regression | svc-rbf | knn |
| --- | --- | --- | --- | --- |
| sst5 | accuracy | 44.1 +/- 2.3 | 44.5 +/- 2.5 | 44.1 +/- 2.5 |
| emotion | accuracy | 49.1 +/- 3.6 | 48.6 +/- 4.4 | 49.2 +/- 3.7 |
| enron_spam | accuracy | 90.1 +/- 3.4 | 90.0 +/- 3.6 | 90.2 +/- 3.4 |
| ag_news | accuracy | 83.0 +/- 3.2 | 82.2 +/- 3.7 | 83.2 +/- 2.8 |
| amazon_counterfactual_en | matthews_correlation | 40.3 +/- 13.5 | 42.7 +/- 14.1 | 38.6 +/- 13.1 |
| SentEval-CR | accuracy | 88.9 +/- 1.4 | 88.8 +/- 1.5 | 89.0 +/- 1.3 |
| Average | N/A | 65.9 +/- 4.6 | 66.2 +/- 4.9 | 65.7 +/- 4.4 |

## Main observations

- Shot size has the largest effect on the average score: 2-shot 49.7, 4-shot 58.0, 8-shot 65.9, 16-shot 70.5.
- num_iterations improves average performance from 60.0 at 2 to 65.9 at 20, with diminishing gains after 10 on several accuracy tasks.
- Classifier heads are close on average: logistic_regression 65.9 +/- 4.6, svc-rbf 66.2 +/- 4.9, knn 65.7 +/- 4.4. Best average: svc-rbf.
- High-variance cases are concentrated in low-shot settings and Amazon MCC. Examples: baseline:8-shot, iter=20, LR:amazon_counterfactual_en (13.5), shot_size:2-shot:emotion (6.4), shot_size:2-shot:enron_spam (7.1), shot_size:2-shot:ag_news (5.4), shot_size:2-shot:amazon_counterfactual_en (8.3), shot_size:2-shot:SentEval-CR (10.5), shot_size:4-shot:amazon_counterfactual_en (13.6), shot_size:4-shot:SentEval-CR (9.0), shot_size:8-shot:amazon_counterfactual_en (13.5), shot_size:16-shot:amazon_counterfactual_en (6.9), num_iterations:2:amazon_counterfactual_en (7.6), num_iterations:5:amazon_counterfactual_en (8.4).
