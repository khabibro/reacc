#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEXGLUE_DIR="${CODEXGLUE_DIR:-/tmp/CodeXGLUE}"
DOWNLOAD_BASE_GPT2="${DOWNLOAD_BASE_GPT2:-0}"
DOWNLOAD_PY150_GPT2="${DOWNLOAD_PY150_GPT2:-1}"
PY150_GPT2_REPO="${PY150_GPT2_REPO:-AISE-TUDelft/CodeGPT-Py150}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

cd "$ROOT_DIR"

echo "Preparing CodeXGLUE line-completion dataset for PY150..."
if [ ! -d "$CODEXGLUE_DIR/.git" ]; then
  rm -rf "$CODEXGLUE_DIR"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/CodeXGLUE.git "$CODEXGLUE_DIR"
fi

git -C "$CODEXGLUE_DIR" sparse-checkout set Code-Code/CodeCompletion-line

mkdir -p dataset/py150 logs save/py150 results

cp "$CODEXGLUE_DIR/Code-Code/CodeCompletion-line/dataset/py150/literals.json" dataset/py150/literals.json
cp "$CODEXGLUE_DIR/Code-Code/CodeCompletion-line/dataset/py150/line_completion/test.json" dataset/py150/test.json

echo "Dataset line counts:"
wc -l dataset/py150/test.json

if [ "$DOWNLOAD_BASE_GPT2" = "1" ]; then
  echo "Downloading base GPT-2 checkpoint into checkpoints/gpt2..."
  mkdir -p checkpoints/gpt2
  for file in config.json vocab.json merges.txt tokenizer_config.json pytorch_model.bin; do
    path="checkpoints/gpt2/$file"
    if [ "$file" = "pytorch_model.bin" ] && [ -f "$path" ] && [ "$(wc -c < "$path")" -lt 500000000 ]; then
      echo "Removing incomplete checkpoint: $path"
      rm -f "$path"
    fi
    if [ ! -s "$path" ]; then
      curl -L --retry 3 --fail -o "checkpoints/gpt2/$file" "https://huggingface.co/gpt2/resolve/main/$file"
    else
      echo "Already exists: checkpoints/gpt2/$file"
    fi
  done
fi

if [ "$DOWNLOAD_PY150_GPT2" = "1" ]; then
  if [ ! -d "py150-ckpt" ] || [ ! -s "py150-ckpt/pytorch_model.bin" ]; then
    echo "Downloading public PY150 checkpoint from $PY150_GPT2_REPO into py150-ckpt..."
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
  else
    echo "Already exists: py150-ckpt/pytorch_model.bin"
  fi
fi

echo "Done."
echo "Use the local PY150 checkpoint for evaluation:"
echo "  export PY150_GPT2_CKPT=py150-ckpt"
echo "Base GPT-2 can be downloaded only for a sanity run:"
echo "  DOWNLOAD_BASE_GPT2=1 ./scripts/prepare_gpt2_eval.sh"
echo "  ALLOW_BASE_GPT2=1 PY150_GPT2_CKPT=checkpoints/gpt2 ./scripts/run_gpt2_eval.sh py150"
