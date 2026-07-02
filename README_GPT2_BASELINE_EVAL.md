# GPT-2 Baseline Evaluation

This repo includes a small workflow for evaluating the GPT-2 baseline row for ReACC/CodeXGLUE line-level code completion.

It evaluates the full official test sets:

- PY150: 10,000 samples
- JavaCorpus: 3,000 samples

The final table is saved as:

- `results/gpt2_baseline_table.md`
- `results/gpt2_baseline_table.csv`

## Important Checkpoint Note

For paper-comparable numbers, use the fine-tuned GPT-2 checkpoints for PY150 and JavaCorpus.

The helper script can download base GPT-2 into `checkpoints/gpt2`, but base GPT-2 is only useful for a sanity run. It will not reproduce the paper row:

| Model | PY150 (Perplexity) | PY150 (Exact Match) | PY150 (Edit Sim) | JavaCorpus (Perplexity) | JavaCorpus (Exact Match) | JavaCorpus (Edit Sim) |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | - | 41.73 | 70.60 | - | 27.50 | 60.36 |

Perplexity is `-` because the line-level GPT-2 table reports Exact Match and Edit Similarity, not perplexity.

## 1. Create Environment

Run from the ReACC repo root.

### CUDA Laptop Or Workstation

```bash
cd /path/to/ReACC

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpt2-eval.txt
```

If your GPU host uses CUDA 12.1 instead of CUDA 12.4, install PyTorch with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### Mac

```bash
cd /Users/admin/Research/ReACC

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install -r requirements-gpt2-eval.txt
```

The patched evaluator automatically chooses `cuda`, then Apple `mps`, then `cpu`.

## 2. Prepare Datasets And Optional Base GPT-2

This downloads the CodeXGLUE line-completion test files and copies them into the layout expected by ReACC.

```bash
cd /path/to/ReACC
./scripts/prepare_gpt2_eval.sh
```

Expected dataset counts:

```text
10000 dataset/py150/test.json
 3000 dataset/javaCorpus/test.json
```

By default, the script also downloads base GPT-2 into `checkpoints/gpt2`. To skip that download:

```bash
DOWNLOAD_BASE_GPT2=0 ./scripts/prepare_gpt2_eval.sh
```

## 3. Set Checkpoints

For exact paper-style reproduction, point these variables to the fine-tuned GPT-2 checkpoints:

```bash
export PY150_GPT2_CKPT=/absolute/path/to/py150-finetuned-gpt2
export JAVA_GPT2_CKPT=/absolute/path/to/java-finetuned-gpt2
```

For a base-GPT-2 sanity run:

```bash
export PY150_GPT2_CKPT=checkpoints/gpt2
export JAVA_GPT2_CKPT=checkpoints/gpt2
```

## 4. Run Evaluations

The evaluator writes predictions incrementally and resumes automatically from an existing prediction file. If a run is interrupted, rerun the same command.

### CUDA Host

Run both datasets sequentially:

```bash
cd /path/to/ReACC
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

./scripts/run_gpt2_eval.sh both
```

Or run one dataset:

```bash
./scripts/run_gpt2_eval.sh py150
./scripts/run_gpt2_eval.sh java
```

### Mac Background Run

On Mac, run one dataset at a time.

```bash
cd /Users/admin/Research/ReACC
source .venv/bin/activate

BACKGROUND=1 ./scripts/run_gpt2_eval.sh py150
tail -f logs/gpt2_py150_line.log
```

After PY150 finishes:

```bash
BACKGROUND=1 ./scripts/run_gpt2_eval.sh java
tail -f logs/gpt2_java_line.log
```

## 5. Extract And Save Final Table

Run this after both logs contain their final `Edit sim` and `EM` lines.

```bash
cd /path/to/ReACC
source .venv/bin/activate

./scripts/extract_gpt2_results.py
```

Outputs:

```text
results/gpt2_baseline_table.md
results/gpt2_baseline_table.csv
```

## Useful Checks

Check active runs:

```bash
ps aux | grep run_lm.py
```

Watch logs:

```bash
tail -f logs/gpt2_py150_line.log
tail -f logs/gpt2_java_line.log
```

Check prediction counts:

```bash
wc -l save/py150/gpt2_predictions.txt save/javaCorpus/gpt2_predictions.txt
```

Expected completed counts:

```text
10000 save/py150/gpt2_predictions.txt
 3000 save/javaCorpus/gpt2_predictions.txt
```
