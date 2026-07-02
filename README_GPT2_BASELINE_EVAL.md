# GPT-2 Baseline Evaluation

This guide runs the full GPT-2 baseline evaluation for ReACC/CodeXGLUE line-level code completion on:

- PY150: 10,000 test samples
- JavaCorpus: 3,000 test samples

For paper-comparable results, use the fine-tuned GPT-2 checkpoints for each dataset. Using the downloaded base `gpt2` checkpoint will run successfully, but it will not reproduce the paper row.

## 1. Setup Environment

Run from the ReACC repo root.

```bash
cd /path/to/ReACC

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CUDA laptop / workstation:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# If your CUDA stack needs CUDA 12.1 instead, use this instead:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.30.2 fuzzywuzzy python-Levenshtein tqdm numpy
```

Verify the GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

For fast execution, `cuda available` should be `True`.

## 2. Prepare Dataset Files

ReACC expects `test.json` and `literals.json` directly under each dataset directory.

```bash
cd /path/to/ReACC

rm -rf /tmp/CodeXGLUE
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/CodeXGLUE.git /tmp/CodeXGLUE
cd /tmp/CodeXGLUE
git sparse-checkout set Code-Code/CodeCompletion-line

cd /path/to/ReACC
mkdir -p dataset/py150 dataset/javaCorpus logs save/py150 save/javaCorpus results

cp /tmp/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/py150/literals.json dataset/py150/literals.json
cp /tmp/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/py150/line_completion/test.json dataset/py150/test.json

cp /tmp/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/javaCorpus/literals.json dataset/javaCorpus/literals.json
cp /tmp/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/javaCorpus/line_completion/test.json dataset/javaCorpus/test.json

wc -l dataset/py150/test.json dataset/javaCorpus/test.json
```

Expected counts:

```text
10000 dataset/py150/test.json
 3000 dataset/javaCorpus/test.json
```

## 3. Set Checkpoint Paths

Set these to the fine-tuned GPT-2 checkpoints you want to evaluate.

```bash
export PY150_GPT2_CKPT=/absolute/path/to/py150-gpt2-checkpoint
export JAVA_GPT2_CKPT=/absolute/path/to/java-gpt2-checkpoint
```

If you only want a sanity run with base GPT-2, download it and point both variables to it:

```bash
mkdir -p checkpoints/gpt2
cd checkpoints/gpt2
for f in config.json vocab.json merges.txt tokenizer_config.json pytorch_model.bin; do
  curl -L --retry 3 --fail -o "$f" "https://huggingface.co/gpt2/resolve/main/$f"
done
cd /path/to/ReACC

export PY150_GPT2_CKPT=checkpoints/gpt2
export JAVA_GPT2_CKPT=checkpoints/gpt2
```

## 4. Run Full PY150 Evaluation

This saves the full log and predictions.

```bash
cd /path/to/ReACC
source .venv/bin/activate
mkdir -p logs save/py150 results

export CUDA_VISIBLE_DEVICES=0

nohup python -u generate/run_lm.py \
  --data_dir=dataset/py150 \
  --lit_file=dataset/py150/literals.json \
  --langs=python \
  --output_dir=save/py150 \
  --pretrain_dir="$PY150_GPT2_CKPT" \
  --save_name=save/py150/gpt2_predictions.txt \
  --model_type=gpt2 \
  --block_size=1024 \
  --eval_line \
  --logging_steps=10 \
  --seed=42 > logs/gpt2_py150_line.log 2>&1 &
```

Watch progress:

```bash
tail -f logs/gpt2_py150_line.log
```

The run is complete when the log contains:

```text
Test 10000 samples
Edit sim: ..., EM: ...
```

## 5. Run Full JavaCorpus Evaluation

This saves the full log and predictions.

```bash
cd /path/to/ReACC
source .venv/bin/activate
mkdir -p logs save/javaCorpus results

export CUDA_VISIBLE_DEVICES=0

nohup python -u generate/run_lm.py \
  --data_dir=dataset/javaCorpus \
  --lit_file=dataset/javaCorpus/literals.json \
  --langs=java \
  --output_dir=save/javaCorpus \
  --pretrain_dir="$JAVA_GPT2_CKPT" \
  --save_name=save/javaCorpus/gpt2_predictions.txt \
  --model_type=gpt2 \
  --block_size=1024 \
  --eval_line \
  --logging_steps=10 \
  --seed=42 > logs/gpt2_java_line.log 2>&1 &
```

Watch progress:

```bash
tail -f logs/gpt2_java_line.log
```

The run is complete when the log contains:

```text
Test 3000 samples
Edit sim: ..., EM: ...
```

## 6. Save Final Result Table

After both runs finish, create Markdown and CSV result files.

```bash
cd /path/to/ReACC
source .venv/bin/activate
mkdir -p results

python - <<'PY'
import csv
import re
from pathlib import Path

def extract_line_metrics(path):
    text = Path(path).read_text(errors="ignore")
    matches = re.findall(r"Edit sim:\s*([0-9.]+),\s*EM:\s*([0-9.]+)", text)
    if not matches:
        raise SystemExit(f"Missing final metrics in {path}")
    edit, em = matches[-1]
    return float(em) * 100, float(edit)

py_em, py_edit = extract_line_metrics("logs/gpt2_py150_line.log")
java_em, java_edit = extract_line_metrics("logs/gpt2_java_line.log")

headers = [
    "Model",
    "PY150 (Perplexity)",
    "PY150 (Exact Match)",
    "PY150 (Edit Sim)",
    "JavaCorpus (Perplexity)",
    "JavaCorpus (Exact Match)",
    "JavaCorpus (Edit Sim)",
]

row = [
    "GPT-2",
    "-",
    f"{py_em:.2f}",
    f"{py_edit:.2f}",
    "-",
    f"{java_em:.2f}",
    f"{java_edit:.2f}",
]

md = []
md.append("| " + " | ".join(headers) + " |")
md.append("|---|---:|---:|---:|---:|---:|---:|")
md.append("| " + " | ".join(row) + " |")
Path("results/gpt2_baseline_table.md").write_text("\n".join(md) + "\n")

with open("results/gpt2_baseline_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerow(row)

print("\n".join(md))
print("\nSaved:")
print("  results/gpt2_baseline_table.md")
print("  results/gpt2_baseline_table.csv")
PY
```

## 7. Paper Row For Reference

The paper reports this GPT-2 row:

| Model | PY150 (Perplexity) | PY150 (Exact Match) | PY150 (Edit Sim) | JavaCorpus (Perplexity) | JavaCorpus (Exact Match) | JavaCorpus (Edit Sim) |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | - | 41.73 | 70.60 | - | 27.50 | 60.36 |

Perplexity is shown as `-` because the line-level CodeXGLUE/ReACC GPT-2 table reports Exact Match and Edit Similarity, not perplexity.
