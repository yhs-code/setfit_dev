# Experiment Result File Index

所有路径均相对于 `scripts/setfit/results/`。

| 实验/用途 | 结果表 CSV | 报告 MD | 备注 |
| --- | --- | --- | --- |
| 实验配置总表 | `ablation_experiment_configs.csv` | `ablation_experiment_configs.md` | 已更新到 64-shot |
| SetFit test-only 主消融汇总：baseline、shot、num_iterations、classifier、Frozen ST + LR | `test_only_ablation_summary.csv` | `test_only_ablation_report.md` | shot 表包含 2/4/8/16/64 |
| Frozen ST + LR vs SetFit | `frozen_st_lr_vs_setfit_comparison.csv` | `frozen_st_lr_vs_setfit_report.md` | 包含 2/4/8/16/64 |
| SetFit-HN-50 8-shot vs SetFit baseline | `hard_negative_hn50_vs_setfit_8shot.csv` | `hard_negative_hn50_vs_setfit_8shot_report.md` | 8-shot 专项对比 |
| SetFit-HN-50 8-shot paired split 显著性分析 | `hard_negative_hn50_vs_setfit_8shot_paired.csv` | `hard_negative_hn50_vs_setfit_8shot_paired.md` | 8-shot 专项 paired 分析 |
| Hard Negative ratio 消融：HN-25/HN-50/HN-75/HN-100 | `hard_negative_ratio_ablation_summary.csv` | `hard_negative_ratio_ablation_report.md` | 8-shot ratio 消融 |
| SetFit-HN-50 shot 消融 vs SetFit | `hard_negative_hn50_shot_ablation_vs_setfit.csv` | `hard_negative_hn50_shot_ablation_report.md` | 包含 2/4/8/16/64 |
| SetFit-HN-50 shot paired split 显著性分析 | `hard_negative_hn50_shot_paired_summary.csv` | `hard_negative_hn50_shot_paired_report.md` | 包含 2/4/8/16/64 |
| 统一合并表目录 | `consolidated_ablation_results/*.csv` | `consolidated_ablation_results/*.md` | 推荐优先阅读 |
