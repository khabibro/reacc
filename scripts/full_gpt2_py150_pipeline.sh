#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "[1/4] Preparing data and checkpoint"
DOWNLOAD_PY150_GPT2=0 ./scripts/prepare_gpt2_eval.sh

echo "[2/4] Fine-tuning GPT-2 on PY150"
./scripts/train_gpt2_py150.sh

echo "[3/4] Evaluating PY150 checkpoint"
PY150_GPT2_CKPT="${PY150_GPT2_CKPT:-py150-ckpt}" ./scripts/run_gpt2_eval.sh py150

echo "[4/4] Extracting result table"
./scripts/extract_gpt2_results.py

echo "Done."
echo "Table:"
cat results/gpt2_baseline_table.md
