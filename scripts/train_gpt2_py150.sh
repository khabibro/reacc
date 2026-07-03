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
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EVALUATE_DURING_TRAINING="${EVALUATE_DURING_TRAINING:-0}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

if [ -z "$TRAIN_BATCH_SIZE" ] || [ -z "$GRAD_ACCUM_STEPS" ] || [ -z "$EVAL_BATCH_SIZE" ]; then
  DEVICE_KIND="$("$PYTHON_BIN" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
)"
  if [ "$DEVICE_KIND" = "mps" ]; then
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-24}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
  else
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
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

"$PYTHON_BIN" - "$CODEXGLUE_DIR/Code-Code/CodeCompletion-token/code/run_lm.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = '''    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
'''
new = '''    if args.local_rank == -1 or args.no_cuda:
        if torch.cuda.is_available() and not args.no_cuda:
            device = torch.device("cuda")
            args.n_gpu = torch.cuda.device_count()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.no_cuda:
            device = torch.device("mps")
            args.n_gpu = 0
        else:
            device = torch.device("cpu")
            args.n_gpu = 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
'''
if old in text:
    path.write_text(text.replace(old, new))
PY

pushd "$CODEXGLUE_DIR/Code-Code/CodeCompletion-token/dataset/py150" >/dev/null
if [ ! -f py150_files/python100k_train.txt ] || [ ! -f py150_files/python50k_eval.txt ]; then
  rm -rf py150_files token_completion
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
if [ ! -s token_completion/train.txt ] || [ ! -s token_completion/dev.txt ] || [ ! -s token_completion/test.txt ]; then
  "$PYTHON_BIN" preprocess.py --base_dir=py150_files --output_dir=token_completion
else
  echo "Already exists: token_completion/train.txt, dev.txt, test.txt"
fi
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

common_args=(
  run_lm.py
  --data_dir="../dataset/py150/token_completion"
  --lit_file="../dataset/py150/literals.json"
  --langs=python
  --output_dir="$ROOT_DIR/$TRAIN_OUTPUT_DIR"
  --model_type=gpt2
  --pretrain_dir="$ROOT_DIR/$BASE_CKPT_DIR"
  --block_size=1024
  --do_train
  --per_gpu_train_batch_size="$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  --per_gpu_eval_batch_size="$EVAL_BATCH_SIZE"
  --learning_rate=2e-5
  --weight_decay=0.01
  --num_train_epochs=30
  --logging_steps=100
  --save_steps=1000
  --seed=42
  --overwrite_output_dir
  --not_pretrain
  --log_file="$ROOT_DIR/$LOG_FILE"
)

if [ "$EVALUATE_DURING_TRAINING" = "1" ]; then
  common_args+=(--evaluate_during_training)
fi

if [ "$PER_NODE_GPU" = "1" ]; then
  "${launch_cmd[@]}" "${common_args[@]}" \
    2>&1 | tee "$ROOT_DIR/$LOG_FILE"
else
  "${launch_cmd[@]}" "${common_args[@]}" \
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
