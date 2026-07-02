# GPT-2 PY150 Baseline Evaluation

This workflow evaluates the GPT-2 baseline for the ReACC/CodeXGLUE line-level code completion task on PY150.

It follows the original ReACC README convention:

```bash
DATADIR=dataset/py150
PRETRAINDIR=py150-ckpt
```

The PY150 test set has 10,000 samples. The final PY150 table is saved as:

- `results/gpt2_baseline_table.md`
- `results/gpt2_baseline_table.csv`

## Important Checkpoint Note

The ReACC README expects `PRETRAINDIR=py150-ckpt`, meaning a GPT-2 checkpoint fine-tuned for PY150 code completion. Raw/base GPT-2 is not the paper baseline and can produce meaningless scores.

Paper reference row:

| Model | PY150 (Perplexity) | PY150 (Exact Match) | PY150 (Edit Sim) |
|---|---:|---:|---:|
| GPT-2 | - | 41.73 | 70.60 |

Perplexity is `-` because the line-level CodeXGLUE/ReACC table reports Exact Match and Edit Similarity, not perplexity.

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

If your GPU host uses CUDA 12.1 instead of CUDA 12.4:

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

## 2. Prepare PY150 Dataset

This downloads the CodeXGLUE line-completion test files and copies them into the layout expected by ReACC.

```bash
cd /path/to/ReACC
./scripts/prepare_gpt2_eval.sh
```

Expected PY150 count:

```text
10000 dataset/py150/test.json
```

## 3. Add The Fine-Tuned GPT-2 Checkpoint

Place or download your PY150-fine-tuned GPT-2 checkpoint at:

```text
py150-ckpt/
```

It must contain Hugging Face GPT-2 checkpoint files such as:

```text
py150-ckpt/config.json
py150-ckpt/pytorch_model.bin
py150-ckpt/vocab.json
py150-ckpt/merges.txt
```

By default, `./scripts/prepare_gpt2_eval.sh` will auto-download the public Py150 checkpoint from Hugging Face into `py150-ckpt/`.

If your checkpoint is somewhere else, set:

```bash
export PY150_GPT2_CKPT=/absolute/path/to/py150-finetuned-gpt2
```

Important: the public Hugging Face model `AISE-TUDelft/CodeGPT-Py150` is available online and is a usable Py150 checkpoint, but I could not verify that it is the exact checkpoint used in the paper. Use it as the local checkpoint unless you have the original paper weights.

## 4. Run PY150 Evaluation

CUDA foreground run:

```bash
cd /path/to/ReACC
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

./scripts/run_gpt2_eval.sh py150
```

Mac background run:

```bash
cd /Users/admin/Research/ReACC
source .venv/bin/activate

BACKGROUND=1 ./scripts/run_gpt2_eval.sh py150
tail -f logs/gpt2_py150_line.log
```

The evaluator writes predictions incrementally and resumes automatically from `save/py150/gpt2_predictions.txt`. If a run is interrupted, rerun the same command.

The run is complete when the log contains:

```text
Test 10000 samples
Edit sim: ..., EM: ...
```

For exact paper-style scoring, also pass the official PY150 answer file if you have it:

```bash
PY150_ANSWERS_FILE=/path/to/answers.jsonl ./scripts/run_gpt2_eval.sh py150
```

Without that file, the repo can still generate predictions, but the exact EM/Edit Sim row will not match the paper because the public `test.json` in this checkout has empty `gt` fields.

## 5. Extract And Save PY150 Result

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

## Base GPT-2 Sanity Run Only

Raw/base GPT-2 is blocked by default to prevent accidentally reporting invalid results.

To run it only as a sanity check:

```bash
DOWNLOAD_BASE_GPT2=1 ./scripts/prepare_gpt2_eval.sh
ALLOW_BASE_GPT2=1 PY150_GPT2_CKPT=checkpoints/gpt2 ./scripts/run_gpt2_eval.sh py150
```

Do not compare that result to the paper row.

## Useful Checks

Check active runs:

```bash
ps aux | grep run_lm.py
```

Watch logs:

```bash
tail -f logs/gpt2_py150_line.log
```

Check prediction count:

```bash
wc -l save/py150/gpt2_predictions.txt
```

Expected completed count:

```text
10000 save/py150/gpt2_predictions.txt
```
