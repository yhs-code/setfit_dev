#!/usr/bin/env bash
set -euo pipefail

cd /home/yanghongsheng/SetFit/setfit_dev

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export DISABLE_TQDM=1

PY=/home/yanghongsheng/.conda/envs/setfit/bin/python
RESULT_DIR=scripts/setfit/results
LOCK_DIR="$RESULT_DIR/.64shot_completion.lock"
STATUS_FILE="$RESULT_DIR/64shot_completion.status"

mkdir -p "$RESULT_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another 64-shot completion job appears to be running: $LOCK_DIR"
  exit 1
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

run_cmd() {
  echo
  echo "===== START $(date '+%F %T') :: $* ====="
  "$@"
  echo "===== END $(date '+%F %T') :: $* ====="
}

echo "running $(date '+%F %T')" > "$STATUS_FILE"

run_cmd "$PY" scripts/setfit/run_fewshot.py \
  --sample_sizes 64 \
  --is_test_set=true \
  --num_iterations=20 \
  --exp_name test_shot_ablation

run_cmd "$PY" scripts/setfit/run_frozen_st_lr.py \
  --sample_sizes 64 \
  --is_test_set=true \
  --exp_name test_frozen_st_lr

run_cmd "$PY" scripts/setfit/run_fewshot.py \
  --sample_sizes 64 \
  --is_test_set=true \
  --num_iterations=20 \
  --pair_strategy hard_negative \
  --hard_negative_ratio=0.5 \
  --exp_name test_hard_negative_hn50_shot_ablation

run_cmd "$PY" scripts/setfit/rebuild_consolidated_ablation_results.py

echo "complete $(date '+%F %T')" > "$STATUS_FILE"
