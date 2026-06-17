# SetFit Ablation Experiment Configs

| experiment | setting | command | result_table |
| --- | --- | --- | --- |
| SetFit shot ablation | 2/4/8/16/64-shot | python run_fewshot.py --sample_sizes 2 4 16 64 --is_test_set=true --num_iterations=20 --exp_name test_shot_ablation; 8-shot reused from test_main | test_only_ablation_summary.csv |
| Frozen ST + LR | 2/4/8/16/64-shot | python run_frozen_st_lr.py --sample_sizes 2 4 8 16 64 --is_test_set=true --exp_name test_frozen_st_lr | frozen_st_lr_vs_setfit_comparison.csv |
| SetFit-HN-50 shot ablation | 2/4/8/16/64-shot | python run_fewshot.py --sample_sizes 2 4 16 64 --is_test_set=true --num_iterations=20 --pair_strategy hard_negative --hard_negative_ratio=0.5 --exp_name test_hard_negative_hn50_shot_ablation; 8-shot reused from test_hard_negative_hn50_8 | hard_negative_hn50_shot_ablation_vs_setfit.csv |
| num_iterations ablation | 8-shot, iter=2/5/10/20 | python run_fewshot.py --sample_sizes 8 --num_iterations {2,5,10,20} --is_test_set=true | test_only_ablation_summary.csv |
| classifier ablation | 8-shot, LR/SVC-RBF/KNN | python run_fewshot.py --sample_sizes 8 --classifier {logistic_regression,svc-rbf,knn} --is_test_set=true | test_only_ablation_summary.csv |
| Hard Negative ratio ablation | 8-shot, HN-25/HN-50/HN-75/HN-100 | python run_fewshot.py --sample_sizes 8 --pair_strategy hard_negative --hard_negative_ratio {0.25,0.5,0.75,1.0} --is_test_set=true | hard_negative_ratio_ablation_summary.csv |
