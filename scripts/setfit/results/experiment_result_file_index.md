# Experiment Result File Index

所有路径均相对于 `scripts/setfit/results/`。

| 实验/用途 | 结果表 CSV | 报告 MD | 原始 summary_table |
| --- | --- | --- | --- |
| 实验配置总表 | `ablation_experiment_configs.csv` | `ablation_experiment_configs.md` | N/A |
| SetFit test-only 主消融汇总：baseline、shot、num_iterations、classifier | `test_only_ablation_summary.csv` | `test_only_ablation_report.md` | 见下方 raw result dirs |
| Frozen ST + LR vs SetFit | `frozen_st_lr_vs_setfit_comparison.csv` | `frozen_st_lr_vs_setfit_report.md` | `paraphrase-mpnet-base-v2-frozen_st_lr-logistic_regression-batch_16-test_frozen_st_lr/summary_table.csv` |
| SetFit-HN-50 8-shot vs SetFit baseline | `hard_negative_hn50_vs_setfit_8shot.csv` | `hard_negative_hn50_vs_setfit_8shot_report.md` | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_8/summary_table.csv` |
| SetFit-HN-50 8-shot paired split 显著性分析 | `hard_negative_hn50_vs_setfit_8shot_paired.csv` | `hard_negative_hn50_vs_setfit_8shot_paired.md` | 同上，逐 split 读取 `results.json` |
| Hard Negative ratio 消融：HN-25/HN-50/HN-75/HN-100 | `hard_negative_ratio_ablation_summary.csv` | `hard_negative_ratio_ablation_report.md` | 见下方 hard-negative raw dirs |
| SetFit-HN-50 shot 消融 vs SetFit：2/4/8/16-shot | `hard_negative_hn50_shot_ablation_vs_setfit.csv` | `hard_negative_hn50_shot_ablation_report.md` | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_shot_ablation/summary_table.csv` |
| SetFit-HN-50 shot paired split 显著性分析 | `hard_negative_hn50_shot_paired_summary.csv` | `hard_negative_hn50_shot_paired_report.md` | 同上，逐 split 读取 `results.json` |

## Raw Result Dirs

| 实验 | 原始 summary_table |
| --- | --- |
| SetFit baseline, 8-shot, LR, iter=20 | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_main/summary_table.csv` |
| SetFit shot 消融，2/4/16-shot | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_shot_ablation/summary_table.csv` |
| SetFit num_iterations=2 | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_2-batch_16-test_iter_2/summary_table.csv` |
| SetFit num_iterations=5 | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_5-batch_16-test_iter_5/summary_table.csv` |
| SetFit num_iterations=10 | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_10-batch_16-test_iter_10/summary_table.csv` |
| SetFit num_iterations=20 | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_main/summary_table.csv` |
| Classifier baseline, LogisticRegression | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_main/summary_table.csv` |
| Classifier SVC-RBF | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-svc-rbf-iterations_20-batch_16-test_clf_svc_rbf_valid/summary_table.csv` |
| Classifier KNN | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-knn-iterations_20-batch_16-test_clf_knn_valid/summary_table.csv` |
| Frozen ST + LR, 2/4/8/16-shot | `paraphrase-mpnet-base-v2-frozen_st_lr-logistic_regression-batch_16-test_frozen_st_lr/summary_table.csv` |
| Hard Negative HN-25, 8-shot | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p25-test_hard_negative_hn25_8/summary_table.csv` |
| Hard Negative HN-50, 8-shot | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_8/summary_table.csv` |
| Hard Negative HN-75, 8-shot | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p75-test_hard_negative_hn75_8/summary_table.csv` |
| Hard Negative HN-100, 8-shot | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_1p0-test_hard_negative_hn100_8/summary_table.csv` |
| Hard Negative HN-50 shot 消融，2/4/16-shot | `paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_shot_ablation/summary_table.csv` |

## Recommended Reading Order

1. `test_only_ablation_report.md`
2. `frozen_st_lr_vs_setfit_report.md`
3. `hard_negative_ratio_ablation_report.md`
4. `hard_negative_hn50_shot_ablation_report.md`
5. `hard_negative_hn50_shot_paired_report.md`
