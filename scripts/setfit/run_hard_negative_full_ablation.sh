#!/usr/bin/env bash
set -u

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export DISABLE_TQDM=1

PY=/home/yanghongsheng/.conda/envs/setfit/bin/python

run_cmd() {
  echo "===== START $(date '+%F %T') :: $* ====="
  "$@"
  status=$?
  echo "===== END $(date '+%F %T') status=${status} :: $* ====="
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
}

run_cmd "$PY" scripts/setfit/run_fewshot.py \
  --sample_sizes=8 \
  --is_test_set=true \
  --num_iterations=20 \
  --pair_strategy hard_negative \
  --hard_negative_ratio=0.25 \
  --exp_name test_hard_negative_hn25_8

run_cmd "$PY" scripts/setfit/run_fewshot.py \
  --sample_sizes=8 \
  --is_test_set=true \
  --num_iterations=20 \
  --pair_strategy hard_negative \
  --hard_negative_ratio=0.75 \
  --exp_name test_hard_negative_hn75_8

run_cmd "$PY" scripts/setfit/run_fewshot.py \
  --sample_sizes=8 \
  --is_test_set=true \
  --num_iterations=20 \
  --pair_strategy hard_negative \
  --hard_negative_ratio=1.0 \
  --exp_name test_hard_negative_hn100_8

run_cmd "$PY" scripts/setfit/run_fewshot.py \
  --sample_sizes 2 4 16 \
  --is_test_set=true \
  --num_iterations=20 \
  --pair_strategy hard_negative \
  --hard_negative_ratio=0.5 \
  --exp_name test_hard_negative_hn50_shot_ablation

echo "FULL_HARD_NEGATIVE_ABLATION_EXIT_CODE=0"
