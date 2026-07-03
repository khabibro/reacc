# GPT-2 PY150 Baseline: Fine-Tune And Evaluate

This guide prepares the ReACC/CodeXGLUE workflow for the PY150 GPT-2 baseline row from the code completion table:

| Model | PY150 (Perplexity) | PY150 (Exact Match) | PY150 (Edit Sim) |
|---|---:|---:|---:|
| GPT-2 | - | 41.73 | 70.60 |

The row is not raw zero-shot GPT-2. It requires GPT-2 fine-tuned on the PY150 code-completion training data, then evaluated on the PY150 line-completion test set.

Perplexity is `-` because the line-level table reports Exact Match and Edit Similarity only.

## What This Repo Provides

The helper scripts in this repo now support the PY150 GPT-2 workflow:

- `scripts/prepare_gpt2_eval.sh`: prepares the PY150 line-completion test files.
- `scripts/train_gpt2_py150.sh`: downloads CodeXGLUE, downloads base `gpt2`, preprocesses PY150 token-completion data, fine-tunes GPT-2, and copies the final checkpoint to `py150-ckpt/`.
- `scripts/run_gpt2_eval.sh`: runs ReACC line-completion evaluation using `py150-ckpt/`.
- `scripts/extract_gpt2_results.py`: extracts the final PY150 table row from the log.
- `scripts/full_gpt2_py150_pipeline.sh`: runs prepare, train, evaluate, and extract in sequence.

## Important Answer-File Note

The public CodeXGLUE PY150 line-completion `test.json` currently copied into this repo has 10,000 examples but empty `gt` fields. That means:

- You can train GPT-2 without the answer file.
- You can generate predictions without the answer file.
- You cannot compute exact local EM/Edit Similarity without an official PY150 line-completion answer file.

For exact scoring, set:

```bash
export PY150_ANSWERS_FILE=/absolute/path/to/py150_answers.jsonl
```

The answer file should be jsonl with one object per test example and a non-empty `gt` field.

## Linux Setup With Conda `llm`

Run from the ReACC repo root on the Linux machine:

```bash
cd /path/to/reacc
conda activate llm

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpt2-eval.txt

chmod +x scripts/prepare_gpt2_eval.sh \
  scripts/train_gpt2_py150.sh \
  scripts/run_gpt2_eval.sh \
  scripts/extract_gpt2_results.py \
  scripts/full_gpt2_py150_pipeline.sh
```

If the machine uses CUDA 12.1 instead of CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Mac Setup

Run from this repo root:

```bash
cd /Users/admin/Research/reacc

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch
pip install -r requirements-gpt2-eval.txt
```

Check Apple MPS:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
PY
```

The training helper automatically patches the temporary CodeXGLUE trainer to choose `cuda`, then Apple `mps`, then `cpu`.

## Fine-Tune GPT-2

The paper-style effective batch size is 96. The script keeps that effective batch size but chooses safer defaults per device:

| Device | Per-device batch | Gradient accumulation | Effective batch |
|---|---:|---:|---:|
| CUDA/Linux default | 12 | 8 | 96 |
| Mac MPS default | 4 | 24 | 96 |

Training settings:

- base model: `gpt2`
- data: PY150 token-completion train split
- epochs: `30`
- learning rate: `2e-5`
- block size: `1024`
- checkpoint output: `py150-ckpt/`
- training output: `save/gpt2_py150_train/`
- log: `logs/gpt2_py150_finetune.log`

### Foreground Training

```bash
source .venv/bin/activate
./scripts/train_gpt2_py150.sh
```

### Mac Background Training

Use this if you want the Mac to keep training while the terminal is idle:

```bash
cd /Users/admin/Research/reacc
source .venv/bin/activate
mkdir -p logs

nohup caffeinate -dimsu bash -lc \
  'cd /Users/admin/Research/reacc && source .venv/bin/activate && ./scripts/train_gpt2_py150.sh' \
  > logs/gpt2_py150_finetune_driver.log 2>&1 &
```

Watch the run:

```bash
tail -f logs/gpt2_py150_finetune_driver.log
```

Check active training:

```bash
ps aux | grep run_lm.py | grep -v grep
```

### Override Batch Settings

If Mac MPS still runs out of memory, reduce the per-device batch and increase accumulation:

```bash
TRAIN_BATCH_SIZE=2 GRAD_ACCUM_STEPS=48 EVAL_BATCH_SIZE=2 ./scripts/train_gpt2_py150.sh
```

If it still fails:

```bash
TRAIN_BATCH_SIZE=1 GRAD_ACCUM_STEPS=96 EVAL_BATCH_SIZE=1 ./scripts/train_gpt2_py150.sh
```

These keep the effective batch size at 96, but they are slower.

## Current Mac OOM Lesson

On this Mac, `TRAIN_BATCH_SIZE=12 GRAD_ACCUM_STEPS=8` reached MPS out-of-memory at the first training step:

```text
RuntimeError: MPS backend out of memory
```

That is why the script now defaults to `TRAIN_BATCH_SIZE=4 GRAD_ACCUM_STEPS=24` on MPS.

The expensive PY150 feature cache is saved at:

```text
save/gpt2_py150_train/train_blocksize_1024_wordsize_1_rank_0
```

After the cache exists, restarting training skips most preprocessing.

## Expected Time

The CodeXGLUE README reports about 25 hours for PY150 on 2 NVIDIA P100 GPUs. A Mac MPS run is usually much slower.

Practical expectation:

- strong CUDA workstation: about 1 day, depending on GPU
- Mac MPS: likely multiple days
- CPU: not recommended

## Evaluate PY150 Line Completion

After training finishes, `scripts/train_gpt2_py150.sh` copies the final `checkpoint-last` files into:

```text
py150-ckpt/
```

Prepare line-completion test files:

```bash
./scripts/prepare_gpt2_eval.sh
```

Run evaluation with the official answer file:

```bash
PY150_ANSWERS_FILE=/absolute/path/to/py150_answers.jsonl \
PY150_GPT2_CKPT=py150-ckpt \
./scripts/run_gpt2_eval.sh py150
```

Watch:

```bash
tail -f logs/gpt2_py150_line.log
```

The run is complete when the log contains:

```text
Test 10000 samples
Edit sim: ..., EM: ...
```

## Extract The Final Table

```bash
./scripts/extract_gpt2_results.py
```

Outputs:

```text
results/gpt2_baseline_table.md
results/gpt2_baseline_table.csv
```

## Full Linux Pipeline

If you already have the official answer file:

```bash
cd /path/to/reacc
conda activate llm

export PY150_ANSWERS_FILE=/absolute/path/to/py150_answers.jsonl
PER_NODE_GPU=1 ./scripts/full_gpt2_py150_pipeline.sh
```

For multi-GPU CUDA training:

```bash
export PY150_ANSWERS_FILE=/absolute/path/to/py150_answers.jsonl
PER_NODE_GPU=4 ./scripts/full_gpt2_py150_pipeline.sh
```

## Public Checkpoint Option

`scripts/prepare_gpt2_eval.sh` can download `AISE-TUDelft/CodeGPT-Py150` into `py150-ckpt/`.

That model is useful as a public PY150 checkpoint, but it is not verified to be the exact GPT-2 checkpoint used in the ReACC paper. For the paper GPT-2 row, the most faithful route is to fine-tune base `gpt2` using the CodeXGLUE PY150 recipe.

## Base GPT-2 Sanity Run Only

Raw/base GPT-2 is blocked by default so it is not accidentally reported as the paper baseline.

For a sanity check only:

```bash
DOWNLOAD_BASE_GPT2=1 ./scripts/prepare_gpt2_eval.sh
ALLOW_BASE_GPT2=1 PY150_GPT2_CKPT=checkpoints/gpt2 ./scripts/run_gpt2_eval.sh py150
```

Do not compare raw GPT-2 output to the paper row.

## Useful Files

Training logs:

```text
logs/gpt2_py150_finetune_driver.log
logs/gpt2_py150_finetune.log
```

Training checkpoint and cache:

```text
save/gpt2_py150_train/
py150-ckpt/
```

Evaluation outputs:

```text
save/py150/gpt2_predictions.txt
logs/gpt2_py150_line.log
results/gpt2_baseline_table.md
results/gpt2_baseline_table.csv
```

Useful checks:

```bash
ps aux | grep run_lm.py | grep -v grep
wc -l save/py150/gpt2_predictions.txt
tail -f logs/gpt2_py150_finetune_driver.log
tail -f logs/gpt2_py150_line.log
```
