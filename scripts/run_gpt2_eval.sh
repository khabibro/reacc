#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-py150}"
PY150_GPT2_CKPT="${PY150_GPT2_CKPT:-py150-ckpt}"
PY150_ANSWERS_FILE="${PY150_ANSWERS_FILE:-}"
PY150_GPT2_REPO="${PY150_GPT2_REPO:-AISE-TUDelft/CodeGPT-Py150}"
AUTO_DOWNLOAD_PY150_CKPT="${AUTO_DOWNLOAD_PY150_CKPT:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
SEED="${SEED:-42}"
BACKGROUND="${BACKGROUND:-0}"
ALLOW_BASE_GPT2="${ALLOW_BASE_GPT2:-0}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

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

  if [ "$AUTO_DOWNLOAD_PY150_CKPT" = "1" ] && [ "$ckpt" = "py150-ckpt" ] && [ ! -d "$ckpt" ]; then
    echo "Auto-downloading public PY150 checkpoint from $PY150_GPT2_REPO into $ckpt..."
    PY150_GPT2_REPO="$PY150_GPT2_REPO" "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("PY150_GPT2_REPO", "AISE-TUDelft/CodeGPT-Py150")
snapshot_download(
    repo_id=repo_id,
    local_dir="py150-ckpt",
    local_dir_use_symlinks=False,
)
PY
  fi

  if [ ! -d "$ckpt" ]; then
    echo "Checkpoint directory does not exist: $ckpt" >&2
    echo "Set the correct fine-tuned checkpoint path, for example:" >&2
    echo "  export PY150_GPT2_CKPT=py150-ckpt" >&2
    return 1
  fi

  if [ "$ckpt" = "checkpoints/gpt2" ] && [ "$ALLOW_BASE_GPT2" != "1" ]; then
    echo "Refusing to evaluate raw base GPT-2 at checkpoints/gpt2." >&2
    echo "The ReACC/CodeXGLUE GPT-2 row requires a fine-tuned checkpoint such as py150-ckpt." >&2
    echo "For a sanity run only, rerun with ALLOW_BASE_GPT2=1." >&2
    return 1
  fi

  if pgrep -f "generate/run_lm.py .*--data_dir=$data_dir" >/dev/null; then
    echo "A run for $data_dir already appears to be active. Stop it first or wait for it to finish."
    return 1
  fi

  echo "Starting $name evaluation"
  echo "  checkpoint: $ckpt"
  echo "  log:        $log_file"
  echo "  save:       $save_name"

  local cmd=(
    "$PYTHON_BIN" -u generate/run_lm.py
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

  if [ -n "$PY150_ANSWERS_FILE" ]; then
    cmd+=(--answers_file="$PY150_ANSWERS_FILE")
  fi

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
  *)
    echo "Usage: $0 [py150]" >&2
    exit 2
    ;;
esac
