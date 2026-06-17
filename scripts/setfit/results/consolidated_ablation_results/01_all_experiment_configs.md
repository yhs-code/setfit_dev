# All Experiment Configs

| experiment | method | shots | num_iterations | classifier | pair_strategy | hard_negative_ratio | result_dirs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SetFit shot ablation | SetFit | 2/4/8/16/64 | 20 | logistic_regression | random |  | paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_shot_ablation; paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-test_main |
| Frozen ST + LR | Frozen ST + LR | 2/4/8/16/64 |  | logistic_regression |  |  | paraphrase-mpnet-base-v2-frozen_st_lr-logistic_regression-batch_16-test_frozen_st_lr |
| Hard Negative shot ablation | SetFit-HN-50 | 2/4/8/16/64 | 20 | logistic_regression | hard_negative | 0.5 | paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_shot_ablation; paraphrase-mpnet-base-v2-CosineSimilarityLoss-logistic_regression-iterations_20-batch_16-pair_hard_negative_ratio_0p5-test_hard_negative_hn50_8 |
| Hard Negative ratio ablation | SetFit-HN | 8 | 20 | logistic_regression | hard_negative | 0.25/0.5/0.75/1.0 | HN-25/HN-50/HN-75/HN-100 8-shot dirs |
