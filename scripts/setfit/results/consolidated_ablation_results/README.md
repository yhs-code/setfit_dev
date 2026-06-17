# Consolidated Ablation Results

这个目录由 `scripts/setfit/rebuild_consolidated_ablation_results.py` 从原始 `results.json` 重新生成。

| 文件 | 内容 |
| --- | --- |
| `00_unified_score_results.csv/md` | 所有能统一的分数型结果长表：method、shot、num_iterations、classifier、HN ratio |
| `01_all_experiment_configs.csv/md` | 所有实验配置和命令参数 |
| `02_core_setfit_ablations.csv/md` | 原始 SetFit 主消融：shot、num_iterations、classifier |
| `03_unified_method_by_shot_comparison.csv/md` | Frozen ST + LR、SetFit、SetFit-HN-50 的统一 shot 对比 |
| `04_hard_negative_ratio_ablation.csv/md` | Hard Negative ratio 消融 |
| `05_paired_significance_summary.csv/md` | HN-50 vs SetFit 的 paired split 稳定性/显著性分析 |

## Recommended Reading

一般只需要看三类文件：

1. `00_unified_score_results.md`：所有分数型结果的总表。
2. `05_paired_significance_summary.md`：HN-50 相比 SetFit 是否稳定提升。
3. `01_all_experiment_configs.md`：需要确认实验配置和参数时再看。

如果觉得 `00_unified_score_results.md` 太长，可以只看 `03_unified_method_by_shot_comparison.md`，它是最核心的 Frozen ST + LR / SetFit / SetFit-HN-50 对比。

64-shot 结果已补齐，表格已同步更新。
