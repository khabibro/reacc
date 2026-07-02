#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-both}"
PY150_GPT2_CKPT="${PY150_GPT2_CKPT:-checkpoints/gpt2}"
JAVA_GPT2_CKPT="${JAVA_GPT2_CKPT:-checkpoints/gpt2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
SEED="${SEED:-42}"
BACKGROUND="${BACKGROUND:-0}"

cd "$ROOT_DIR"

run_one() {
  local name="$1"
  local lang="$2"
  local data_dir="$3"
  local lit_file="$4"
  local output_dir="$5"
  local ckpt="$6"
  local save_name="$7"
  local log_file="$8"

  mkdir -p "$output_dir" "$(dirname "$log_file")" results

  if pgrep -f "generate/run_lm.py .*--data_dir=$data_dir" >/dev/null; then
    echo "A run for $data_dir already appears to be active. Stop it first or wait for it to finish."
    return 1
  fi

  echo "Starting $name evaluation"
  echo "  checkpoint: $ckpt"
  echo "  log:        $log_file"
  echo "  save:       $save_name"

  local cmd=(
    python -u generate/run_lm.py
    --data_dir="$data_dir"
    --lit_file="$lit_file"
    --langs="$lang"
    --output_dir="$output_dir"
    --pretrain_dir="$ckpt"
    --save_name="$save_name"
    --model_type=gpt2
    --block_size="$BLOCK_SIZE"
    --eval_line
    --logging_steps="$LOGGING_STEPS"
    --seed="$SEED"
  )

  if [ "$BACKGROUND" = "1" ]; then
    if command -v caffeinate >/dev/null 2>&1; then
      nohup caffeinate -dimsu "${cmd[@]}" > "$log_file" 2>&1 &
    else
      nohup "${cmd[@]}" > "$log_file" 2>&1 &
    fi
    echo "Started in background. Watch with:"
    echo "  tail -f $log_file"
  else
    "${cmd[@]}" 2>&1 | tee "$log_file"
  fi
}

case "$TASK" in
  py150)
    run_one "PY150" "python" "dataset/py150" "dataset/py150/literals.json" "save/py150" "$PY150_GPT2_CKPT" "save/py150/gpt2_predictions.txt" "logs/gpt2_py150_line.log"
    ;;
  java|javaCorpus)
    run_one "JavaCorpus" "java" "dataset/javaCorpus" "dataset/javaCorpus/literals.json" "save/javaCorpus" "$JAVA_GPT2_CKPT" "save/javaCorpus/gpt2_predictions.txt" "logs/gpt2_java_line.log"
    ;;
  both)
    run_one "PY150" "python" "dataset/py150" "dataset/py150/literals.json" "save/py150" "$PY150_GPT2_CKPT" "save/py150/gpt2_predictions.txt" "logs/gpt2_py150_line.log"
    run_one "JavaCorpus" "java" "dataset/javaCorpus" "dataset/javaCorpus/literals.json" "save/javaCorpus" "$JAVA_GPT2_CKPT" "save/javaCorpus/gpt2_predictions.txt" "logs/gpt2_java_line.log"
    ;;
  *)
    echo "Usage: $0 [py150|java|both]" >&2
    exit 2
    ;;
esac
