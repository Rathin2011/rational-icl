"""Verify Phase-1 g/f learning: CE vs Bayes + token accuracy on ID and OOD."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

from config import CompUrnsConfig, Z_OFFSET, Z_SIZE, Y_OFFSET, Y_SIZE
from data import CompUrnsEvalDataset, get_or_create_tasks
from phase1_gate import (
    bayes_ce_f_only_batch,
    bayes_ce_g_only_batch,
    evaluate_phase1_gate,
)
from predictors import f_predictive, g_predictive


@torch.no_grad()
def token_acc(model, dataset, device: str, out_offset: int, out_size: int) -> float:
    """Argmax accuracy on output tokens restricted to the correct alphabet slice."""
    correct = total = 0
    for ex in dataset:
        ids = ex["input_ids"].unsqueeze(0).to(device)
        labels = ex["labels"]
        logits = model(input_ids=ids).logits[0].cpu()
        for t in range(labels.numel() - 1):
            lab = int(labels[t + 1].item())
            if lab == -100:
                continue
            slice_logits = logits[t, out_offset : out_offset + out_size]
            pred = out_offset + int(slice_logits.argmax().item())
            correct += int(pred == lab)
            total += 1
    return correct / max(total, 1)


def bayes_acc_g(examples) -> float:
    c = t = 0
    for ex in examples:
        pairs = ex["pairs"]
        for i, (x, z) in enumerate(pairs):
            p = g_predictive(pairs[:i], query_x=x)
            c += int(int(np.argmax(p)) == z)
            t += 1
    return c / max(t, 1)


def bayes_acc_f(examples) -> float:
    c = t = 0
    for ex in examples:
        pairs = ex["pairs"]
        for i, (z, y) in enumerate(pairs):
            p = f_predictive(pairs[:i], query_z=z)
            c += int(int(np.argmax(p)) == y)
            t += 1
    return c / max(t, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ckpt={args.checkpoint}")

    cfg = CompUrnsConfig(num_tasks=64, seed=args.seed)
    twg, twf = get_or_create_tasks(cfg.train_tasks_path, cfg.num_tasks, cfg.seed)
    owg, owf = get_or_create_tasks(
        cfg.ood_tasks_path, cfg.num_ood_tasks, cfg.seed + 1000
    )

    model = GPTNeoXForCausalLM.from_pretrained(args.checkpoint).to(device).eval()

    rows = []
    for split, wg, wf, s0 in [
        ("ID", twg, twf, args.seed),
        ("OOD", owg, owf, args.seed + 100),
    ]:
        gset = CompUrnsEvalDataset(wg, wf, "g_only", args.n_eval, s0)
        fset = CompUrnsEvalDataset(wg, wf, "f_only", args.n_eval, s0 + 1)
        ok, gate = evaluate_phase1_gate(model, gset, fset, device)
        row = {
            "split": split,
            "gate_passed": bool(ok),
            **{k: float(v) for k, v in gate.items()},
            "bayes_ce_g_check": float(bayes_ce_g_only_batch(gset.examples)),
            "bayes_ce_f_check": float(bayes_ce_f_only_batch(fset.examples)),
            "model_acc_g": token_acc(model, gset, device, Z_OFFSET, Z_SIZE),
            "bayes_acc_g": bayes_acc_g(gset.examples),
            "model_acc_f": token_acc(model, fset, device, Y_OFFSET, Y_SIZE),
            "bayes_acc_f": bayes_acc_f(fset.examples),
            "chance_g": 1.0 / Z_SIZE,
            "chance_f": 1.0 / Y_SIZE,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    out = args.out_json
    if out is None:
        out = os.path.join(
            os.path.dirname(args.checkpoint), "gf_learn_check.json"
        )
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
