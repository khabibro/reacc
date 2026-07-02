#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def extract_line_metrics(path: Path):
    text = path.read_text(errors="ignore")
    matches = re.findall(r"Edit sim:\s*([0-9.]+),\s*EM:\s*([0-9.]+)", text)
    if not matches:
        raise SystemExit(f"Missing final metrics in {path}")
    edit, em = matches[-1]
    return float(em) * 100, float(edit)


def main():
    parser = argparse.ArgumentParser(description="Extract the PY150 GPT-2 baseline table from ReACC logs.")
    parser.add_argument("--py-log", default="logs/gpt2_py150_line.log")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    py_em, py_edit = extract_line_metrics(Path(args.py_log))
    headers = [
        "Model",
        "PY150 (Perplexity)",
        "PY150 (Exact Match)",
        "PY150 (Edit Sim)",
    ]
    row = [
        "GPT-2",
        "-",
        f"{py_em:.2f}",
        f"{py_edit:.2f}",
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("|---|---:|---:|---:|")
    md.append("| " + " | ".join(row) + " |")

    (out_dir / "gpt2_baseline_table.md").write_text("\n".join(md) + "\n")
    with (out_dir / "gpt2_baseline_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)

    print("\n".join(md))
    print("\nSaved:")
    print(f"  {out_dir / 'gpt2_baseline_table.md'}")
    print(f"  {out_dir / 'gpt2_baseline_table.csv'}")


if __name__ == "__main__":
    main()
