"""Phase A: score existing |Z|=5 checkpoints for §5.1 regime map.

Label rule:
  d_rel > 0.5  -> M
  else if best z_decode >= 0.4 -> C_GG
  else -> G

Writes:
  $CACHE/composition/analysis/phase_5_1_results.csv
  $CACHE/composition/analysis/figs/phase_5_1_scatter.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM

# Ensure composition/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import X_SIZE, Y_SIZE, NUM_PAIRS, CompositionConfig, read_cache_dir
from data import CompositionEvalDataset, sample_tasks, save_tasks
import compare_predictors as CP
import probe_composition as P


Z_DECODE_THRESH = 0.4
D_REL_M_THRESH = 0.5

RUNS = [
    # (D, run_dirname, z_size, legacy_untagged)
    (512, "D512-8L-1H-64d-256ff-lr0.0005-bs128-100000steps-seed1", 5, True),
    (1024, "D1024-8L-1H-64d-256ff-lr0.0005-bs128-100000steps-seed1", 5, True),
    (2048, "D2048-8L-1H-64d-256ff-lr0.0005-bs128-100000steps-seed1", 5, True),
    (4096, "D4096-X50-Z5-Y50-8L-1H-64d-256ff-lr0.0005-bs128-100000steps-seed1", 5, False),
    (8192, "D8192-X50-Z5-Y50-8L-1H-64d-256ff-lr0.0005-bs128-100000steps-seed1", 5, False),
]

STEPS_FRAC = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def patch_z(z_size):
    import config as C
    import data as D

    C.Z_SIZE = z_size
    C.Z_OFFSET = X_SIZE
    C.Y_OFFSET = X_SIZE + z_size
    C.VOCAB_SIZE = X_SIZE + z_size + Y_SIZE
    D.Z_SIZE = z_size
    D.Z_OFFSET = C.Z_OFFSET
    D.Y_OFFSET = C.Y_OFFSET
    P.Z_SIZE = z_size
    P.Z_OFFSET = C.Z_OFFSET
    P.Y_OFFSET = C.Y_OFFSET
    CP.Y_SIZE = Y_SIZE  # unchanged
    return C.Z_OFFSET, C.Y_OFFSET


def pick_steps(ckpt_dir):
    nums = sorted(
        int(x.split("-")[1])
        for x in os.listdir(ckpt_dir)
        if x.startswith("checkpoint-")
    )
    if not nums:
        return []
    out = []
    for f in STEPS_FRAC:
        i = int(round(f * (len(nums) - 1)))
        out.append(nums[i])
    # unique preserve order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def load_tasks(cfg, z_size, legacy):
    if legacy:
        train_path = os.path.join(
            cfg.tasks_dir, f"train-D{cfg.num_tasks}-seed{cfg.seed}.npz"
        )
        ood_path = os.path.join(
            cfg.tasks_dir, f"ood-D{cfg.num_ood_tasks}-seed{cfg.seed + 1000}.npz"
        )
    else:
        train_path = cfg.train_tasks_path
        ood_path = cfg.ood_tasks_path

    def _load(path, n, seed):
        if os.path.exists(path):
            data = np.load(path)
            return data["g"], data["f"]
        g, f = sample_tasks(n, seed)
        save_tasks(path, g, f)
        return g, f

    train_g, train_f = _load(train_path, cfg.num_tasks, cfg.seed)
    ood_g, ood_f = _load(ood_path, cfg.num_ood_tasks, cfg.seed + 1000)
    return train_g, train_f, ood_g, ood_f


@torch.no_grad()
def score_d_rel(model, train_h, eval_ds, y_off, device, max_examples=128):
    n = min(len(eval_ds), max_examples)
    d_hM, d_hG, d_GM = [], [], []
    correct = 0
    total = 0
    bs = 32
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = torch.stack([eval_ds[i]["input_ids"] for i in range(start, end)])
        logits = model(input_ids=batch.to(device)).logits
        for b in range(end - start):
            pairs = CP.decode_pairs(batch[b].numpy(), 0, y_off)
            for i in range(NUM_PAIRS):
                pos = 2 * i
                p_h = (
                    F.softmax(logits[b, pos, y_off : y_off + Y_SIZE].float(), dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                prefix = pairs[:i]
                qx, y_true = pairs[i]
                p_G = CP.atomic_G_dist(prefix, qx, Y_SIZE)
                p_M = CP.memorizing_M_dist(prefix, qx, train_h, Y_SIZE)
                d_hM.append(CP.sym_kl(p_h, p_M))
                d_hG.append(CP.sym_kl(p_h, p_G))
                d_GM.append(CP.sym_kl(p_G, p_M))
                total += 1
                if int(p_h.argmax()) == y_true:
                    correct += 1
    d_hM, d_hG, d_GM = map(float, (np.mean(d_hM), np.mean(d_hG), np.mean(d_GM)))
    r = 0.0 if d_GM < 1e-12 else (d_hG - d_hM) / d_GM
    r = float(np.clip(r, -1, 1))
    d_rel = (r + 1) / 2
    return {
        "d_rel": d_rel,
        "model_acc": correct / total,
        "d(h,M)": d_hM,
        "d(h,G)": d_hG,
        "d(G,M)": d_GM,
    }


def score_probe(model, g, f, device, n_trials=80, layers=None):
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))
    best = {"z_decode_acc": -1.0, "patch_to_y_b_given_diff_y": 0.0, "layer": -1}
    for layer in layers:
        summary, _ = P.run_probe(
            model,
            g,
            f,
            n_trials=n_trials,
            layer_idx=layer,
            seed=1 + 17 + layer,
            device=device,
            task_pool="train",
        )
        if summary["z_decode_acc"] > best["z_decode_acc"]:
            best = {
                "z_decode_acc": summary["z_decode_acc"],
                "patch_to_y_b_given_diff_y": summary["patch_to_y_b_given_diff_y"],
                "layer": layer,
                "clean_acc_b": summary["clean_acc_b"],
            }
    return best


def label_regime(d_rel, z_decode):
    if d_rel > D_REL_M_THRESH:
        return "M"
    if z_decode >= Z_DECODE_THRESH:
        return "C_GG"
    return "G"


def plot_scatter(rows, out_path):
    colors = {"M": "#e76f51", "G": "#2a9d8f", "C_GG": "#264653"}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for lab in ("M", "G", "C_GG"):
        pts = [r for r in rows if r["regime"] == lab]
        if not pts:
            continue
        ax.scatter(
            [r["D"] for r in pts],
            [r["step"] for r in pts],
            c=colors[lab],
            s=55,
            label=lab,
            edgecolors="k",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("D (num tasks)")
    ax.set_ylabel("N (checkpoint step)")
    ax.set_title(
        "§5.1 regime map (|Z|=5)\n"
        "M: d_rel>0.5 · C_GG: not-M & z_decode≥0.4 · G: otherwise"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="regime")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_examples", type=int, default=128)
    ap.add_argument("--n_trials", type=int, default=80)
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    cache = args.cache_dir or read_cache_dir()
    runs_root = os.path.join(cache, "composition", "runs")
    analysis = os.path.join(cache, "composition", "analysis")
    os.makedirs(analysis, exist_ok=True)
    csv_path = os.path.join(analysis, "phase_5_1_results.csv")
    fig_path = os.path.join(analysis, "figs", "phase_5_1_scatter.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    rows = []
    for D, run_name, z_size, legacy in RUNS:
        run_dir = os.path.join(runs_root, run_name)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            print(f"SKIP missing {run_dir}")
            continue
        patch_z(z_size)
        cfg = CompositionConfig(num_tasks=D, seed=1, cache_dir=cache, max_steps=100_000)
        train_g, train_f, _, _ = load_tasks(cfg, z_size, legacy)
        train_h = np.stack(
            [CP.composite_h(train_g[d], train_f[d]) for d in range(train_g.shape[0])]
        )
        eval_ds = CompositionEvalDataset(
            train_g, train_f, "comp", min(cfg.n_eval, 512), cfg.seed + 1
        )
        _, y_off = patch_z(z_size)
        steps = pick_steps(ckpt_dir)
        print(f"\n=== D={D} legacy={legacy} steps={steps} ===")
        for step in steps:
            ckpt = os.path.join(ckpt_dir, f"checkpoint-{step}")
            print(f"  scoring step={step} ...", flush=True)
            model = GPTNeoXForCausalLM.from_pretrained(ckpt)
            model.to(device).eval()
            mg = score_d_rel(
                model, train_h, eval_ds, y_off, device, max_examples=args.max_examples
            )
            pr = score_probe(
                model,
                train_g,
                train_f,
                device,
                n_trials=args.n_trials,
                layers=[0, 2, 4, 6, 7],
            )
            regime = label_regime(mg["d_rel"], pr["z_decode_acc"])
            row = {
                "D": D,
                "N_max": 100000,
                "step": step,
                "z_size": z_size,
                "legacy": int(legacy),
                "run": run_name,
                "checkpoint": ckpt,
                "d_rel": mg["d_rel"],
                "model_acc": mg["model_acc"],
                "best_z_decode": pr["z_decode_acc"],
                "best_patch_diff_y": pr["patch_to_y_b_given_diff_y"],
                "best_probe_layer": pr["layer"],
                "regime": regime,
            }
            rows.append(row)
            print(
                f"    d_rel={mg['d_rel']:.3f} z_dec={pr['z_decode_acc']:.3f} "
                f"-> {regime}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # incremental save
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            plot_scatter(rows, fig_path)

    with open(os.path.join(analysis, "phase_5_1_results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nDone. {len(rows)} points -> {csv_path}")
    plot_scatter(rows, fig_path)


if __name__ == "__main__":
    main()
