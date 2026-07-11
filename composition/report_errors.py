"""Print token error rates (1 - accuracy) from a composition training logs.csv.

Usage:
    python report_errors.py /path/to/logs.csv
    python report_errors.py /path/to/run_dir          # uses run_dir/logs.csv
    python report_errors.py /path/to/logs.csv --all-steps
"""

import argparse
import csv
import os
import sys


# Eval sets logged by train.py
PROBE_KEYS = [
    "id_comp",
    "ood_comp",
    "id_g_only",
    "ood_g_only",
    "id_f_only",
    "ood_f_only",
]


def load_rows(path):
    with open(path, newline="", encoding="utf8") as fh:
        return list(csv.DictReader(fh))


def latest_eval_by_step(rows):
    """Group metrics that share the same step (one eval round)."""
    by_step = {}
    for row in rows:
        step = row.get("step") or row.get("global_step")
        if step in (None, "", "NA"):
            # HF sometimes only puts epoch; try to infer from eval_* presence
            if not any(k.startswith("eval_") and k.endswith("_accuracy") for k in row):
                continue
            step = row.get("epoch", "?")
        step = str(step)
        bucket = by_step.setdefault(step, {})
        for k, v in row.items():
            if v not in (None, "", "NA"):
                bucket[k] = v
    return by_step


def fmt_err(acc_str):
    try:
        acc = float(acc_str)
        return f"{(1.0 - acc) * 100:6.2f}%"
    except (TypeError, ValueError):
        return "   n/a"


def print_table(step, metrics):
    print(f"\nStep {step}")
    print(f"  {'split':<14} {'error':>8}  {'accuracy':>8}")
    print(f"  {'-' * 14} {'-' * 8}  {'-' * 8}")
    for name in PROBE_KEYS:
        acc_key = f"eval_{name}_accuracy"
        if acc_key not in metrics:
            continue
        acc = metrics[acc_key]
        try:
            acc_f = float(acc)
            acc_s = f"{acc_f * 100:6.2f}%"
        except ValueError:
            acc_s = "   n/a"
        print(f"  {name:<14} {fmt_err(acc):>8}  {acc_s:>8}")

    # Training loss if present
    if "loss" in metrics:
        print(f"  train_loss: {metrics['loss']}")
    if "eval_id_comp_ce" in metrics:
        print(f"  id_comp CE: {metrics['eval_id_comp_ce']}")
    if "eval_ood_comp_ce" in metrics:
        print(f"  ood_comp CE: {metrics['eval_ood_comp_ce']}")


def main():
    p = argparse.ArgumentParser(description="Report error rates from composition logs.")
    p.add_argument("path", help="logs.csv or a run directory containing logs.csv")
    p.add_argument(
        "--all-steps",
        action="store_true",
        help="Print every eval step (default: final eval only)",
    )
    args = p.parse_args()

    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, "logs.csv")
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(path)
    by_step = latest_eval_by_step(rows)
    if not by_step:
        print("No eval accuracy rows found in logs.")
        sys.exit(1)

    # Prefer numeric steps
    def step_key(s):
        try:
            return (0, float(s))
        except ValueError:
            return (1, s)

    steps = sorted(by_step.keys(), key=step_key)
    print(f"Logs: {path}")
    print(f"Chance error (50-way Y): {100 * (1 - 1 / 50):.1f}%")
    print(f"Chance error (5-way Z):  {100 * (1 - 1 / 5):.1f}%")

    if args.all_steps:
        for s in steps:
            print_table(s, by_step[s])
    else:
        print_table(steps[-1], by_step[steps[-1]])
        print(
            "\nTip: pass --all-steps to see error vs training step N."
        )


if __name__ == "__main__":
    main()
