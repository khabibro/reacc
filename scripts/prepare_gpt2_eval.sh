#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEXGLUE_DIR="${CODEXGLUE_DIR:-/tmp/CodeXGLUE}"
DOWNLOAD_BASE_GPT2="${DOWNLOAD_BASE_GPT2:-1}"

cd "$ROOT_DIR"

echo "Preparing CodeXGLUE line-completion datasets..."
if [ ! -d "$CODEXGLUE_DIR/.git" ]; then
  rm -rf "$CODEXGLUE_DIR"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/CodeXGLUE.git "$CODEXGLUE_DIR"
fi

git -C "$CODEXGLUE_DIR" sparse-checkout set Code-Code/CodeCompletion-line

mkdir -p dataset/py150 dataset/javaCorpus logs save/py150 save/javaCorpus results

cp "$CODEXGLUE_DIR/Code-Code/CodeCompletion-line/dataset/py150/literals.json" dataset/py150/literals.json
cp "$CODEXGLUE_DIR/Code-Code/CodeCompletion-line/dataset/py150/line_completion/test.json" dataset/py150/test.json
cp "$CODEXGLUE_DIR/Code-Code/CodeCompletion-line/dataset/javaCorpus/literals.json" dataset/javaCorpus/literals.json
cp "$CODEXGLUE_DIR/Code-Code/CodeCompletion-line/dataset/javaCorpus/line_completion/test.json" dataset/javaCorpus/test.json

echo "Dataset line counts:"
wc -l dataset/py150/test.json dataset/javaCorpus/test.json

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

echo "Done."
echo "Use fine-tuned checkpoints for paper-comparable numbers:"
echo "  export PY150_GPT2_CKPT=/path/to/py150-finetuned-gpt2"
echo "  export JAVA_GPT2_CKPT=/path/to/java-finetuned-gpt2"
echo "For a sanity run with base GPT-2:"
echo "  export PY150_GPT2_CKPT=checkpoints/gpt2"
echo "  export JAVA_GPT2_CKPT=checkpoints/gpt2"
