#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEXGLUE_DIR="${CODEXGLUE_DIR:-/tmp/CodeXGLUE}"
PER_NODE_GPU="${PER_NODE_GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-save/gpt2_py150_train}"
FINAL_CKPT_DIR="${FINAL_CKPT_DIR:-py150-ckpt}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-checkpoints/gpt2}"
LOG_FILE="${LOG_FILE:-logs/gpt2_py150_finetune.log}"
DOWNLOAD_BASE_GPT2="${DOWNLOAD_BASE_GPT2:-1}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

cd "$ROOT_DIR"
mkdir -p logs save results "$TRAIN_OUTPUT_DIR"

if [ "$DOWNLOAD_BASE_GPT2" = "1" ] && [ ! -d "$BASE_CKPT_DIR" ]; then
  echo "Downloading base GPT-2 checkpoint into $BASE_CKPT_DIR..."
  "$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="gpt2",
    local_dir="checkpoints/gpt2",
    local_dir_use_symlinks=False,
)
PY
fi

if [ ! -d "$BASE_CKPT_DIR" ]; then
  echo "Missing base checkpoint directory: $BASE_CKPT_DIR" >&2
  echo "Run with DOWNLOAD_BASE_GPT2=1 or download gpt2 first." >&2
  exit 1
fi

echo "Preparing CodeXGLUE token-completion data..."
if [ ! -d "$CODEXGLUE_DIR/.git" ]; then
  rm -rf "$CODEXGLUE_DIR"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/CodeXGLUE.git "$CODEXGLUE_DIR"
fi
git -C "$CODEXGLUE_DIR" sparse-checkout set Code-Code/CodeCompletion-token

pushd "$CODEXGLUE_DIR/Code-Code/CodeCompletion-token/dataset/py150" >/dev/null
if [ ! -d py150_files ]; then
  if [ ! -f py150_files.tar.gz ]; then
    echo "Downloading py150_files.tar.gz..."
    curl -L --retry 3 --fail -o py150_files.tar.gz http://files.srl.inf.ethz.ch/data/py150_files.tar.gz
  fi
  mkdir -p py150_files
  tar -C py150_files -zxf py150_files.tar.gz
  rm -f py150_files.tar.gz
  tar -zxvf py150_files/data.tar.gz -C py150_files >/dev/null
  rm -f py150_files/data.tar.gz
fi
"$PYTHON_BIN" preprocess.py --base_dir=py150_files --output_dir=token_completion
popd >/dev/null

echo "Fine-tuning GPT-2 on PY150..."
pushd "$CODEXGLUE_DIR/Code-Code/CodeCompletion-token/code" >/dev/null
if [ "$PER_NODE_GPU" = "1" ]; then
  launch_cmd=("$PYTHON_BIN" -u)
elif command -v torchrun >/dev/null 2>&1; then
  launch_cmd=(torchrun --standalone --nproc_per_node="$PER_NODE_GPU")
else
  launch_cmd=("$PYTHON_BIN" -m torch.distributed.launch --nproc_per_node="$PER_NODE_GPU")
fi

if [ "$PER_NODE_GPU" = "1" ]; then
  "${launch_cmd[@]}" run_lm.py \
    --data_dir="../dataset/py150/token_completion" \
    --lit_file="../dataset/py150/literals.json" \
    --langs=python \
    --output_dir="$ROOT_DIR/$TRAIN_OUTPUT_DIR" \
    --model_type=gpt2 \
    --pretrain_dir="$ROOT_DIR/$BASE_CKPT_DIR" \
    --block_size=1024 \
    --do_train \
    --evaluate_during_training \
    --per_gpu_train_batch_size=12 \
    --gradient_accumulation_steps=8 \
    --per_gpu_eval_batch_size=12 \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --num_train_epochs=30 \
    --logging_steps=100 \
    --save_steps=1000 \
    --seed=42 \
    --overwrite_output_dir \
    --not_pretrain \
    --log_file="$ROOT_DIR/$LOG_FILE" \
    2>&1 | tee "$ROOT_DIR/$LOG_FILE"
else
  "${launch_cmd[@]}" run_lm.py \
  --data_dir="../dataset/py150/token_completion" \
  --lit_file="../dataset/py150/literals.json" \
  --langs=python \
  --output_dir="$ROOT_DIR/$TRAIN_OUTPUT_DIR" \
  --model_type=gpt2 \
  --pretrain_dir="$ROOT_DIR/$BASE_CKPT_DIR" \
  --block_size=1024 \
  --do_train \
  --evaluate_during_training \
  --per_gpu_train_batch_size=12 \
  --gradient_accumulation_steps=8 \
  --per_gpu_eval_batch_size=12 \
  --learning_rate=2e-5 \
  --weight_decay=0.01 \
  --num_train_epochs=30 \
  --logging_steps=100 \
  --save_steps=1000 \
  --seed=42 \
  --overwrite_output_dir \
  --not_pretrain \
  --log_file="$ROOT_DIR/$LOG_FILE" \
  2>&1 | tee "$ROOT_DIR/$LOG_FILE"
fi
popd >/dev/null

FINAL_SRC="$ROOT_DIR/$TRAIN_OUTPUT_DIR/checkpoint-last"
if [ ! -d "$FINAL_SRC" ]; then
  echo "Training finished, but checkpoint-last was not found at $FINAL_SRC" >&2
  exit 1
fi

rm -rf "$ROOT_DIR/$FINAL_CKPT_DIR"
mkdir -p "$ROOT_DIR/$FINAL_CKPT_DIR"
cp -R "$FINAL_SRC"/. "$ROOT_DIR/$FINAL_CKPT_DIR"/

echo "Fine-tuning complete."
echo "Final checkpoint copied to: $FINAL_CKPT_DIR"
echo "Use this for ReACC evaluation:"
echo "  PY150_GPT2_CKPT=$FINAL_CKPT_DIR ./scripts/run_gpt2_eval.sh py150"
