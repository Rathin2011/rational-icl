"""Compare Transformer next-token preds to closed-form M and G (Wurgaft-style).

Atomic G: uniform prior over all deterministic h: X -> Y.
  - If query x seen in prefix: delta on that y
  - Else: Uniform(Y)

Memorizing M: discrete prior over the D training tasks' composites h = f o g.
  - Hard likelihood: 1 if all prefix pairs consistent with task, else 0
  - Predictive: mixture of h_d(x) over tasks with positive posterior mass
  - If no task matches: Uniform(Y)

Relative distance (Wurgaft):
  r = (d(h,G) - d(h,M)) / d(G,M)
  d_rel = (r + 1) / 2
  -> 0 closer to G, 1 closer to M

Distance: symmetrized KL on the Y-slice predictive distribution.

Usage (from composition/):
  python compare_predictors.py \\
    --checkpoint $CACHE_DIR/composition/runs/.../checkpoints/checkpoint-100000 \\
    --num_tasks 2048 --z_size 45 --split ood
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM

from config import (
    X_SIZE,
    Y_SIZE,
    NUM_PAIRS,
    CompositionConfig,
    read_cache_dir,
)
from data import CompositionEvalDataset, sample_tasks, save_tasks


def offsets(z_size):
    x_off = 0
    z_off = X_SIZE
    y_off = X_SIZE + z_size
    return x_off, z_off, y_off


def composite_h(g_row, f_row):
    """h[x] = f[g[x]] for one task."""
    return f_row[g_row]


def atomic_G_dist(prefix_pairs, query_x, y_size=Y_SIZE):
    """Return length-Y probability vector for atomic G."""
    table = {}
    for x, y in prefix_pairs:
        table[int(x)] = int(y)
    p = np.full(y_size, 1.0 / y_size, dtype=np.float64)
    qx = int(query_x)
    if qx in table:
        p[:] = 0.0
        p[table[qx]] = 1.0
    return p


def memorizing_M_dist(prefix_pairs, query_x, train_h, y_size=Y_SIZE):
    """Return length-Y probability vector for M over train composites train_h[D, X]."""
    masses = np.ones(train_h.shape[0], dtype=np.float64)
    for x, y in prefix_pairs:
        masses *= (train_h[:, int(x)] == int(y))
    total = masses.sum()
    p = np.zeros(y_size, dtype=np.float64)
    if total <= 0:
        p[:] = 1.0 / y_size
        return p
    masses /= total
    np.add.at(p, train_h[:, int(query_x)], masses)
    return p


def decode_pairs(input_ids, x_off, y_off):
    pairs = []
    for i in range(NUM_PAIRS):
        x = int(input_ids[2 * i]) - x_off
        y = int(input_ids[2 * i + 1]) - y_off
        pairs.append((x, y))
    return pairs


def sym_kl(p, q, eps=1e-12):
    """Symmetrized KL for 1D discrete distributions (numpy)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = np.sum(p * (np.log(p) - np.log(q)))
    kl_qp = np.sum(q * (np.log(q) - np.log(p)))
    return 0.5 * (kl_pq + kl_qp)


def load_or_create_tasks_for_z(cfg, z_size, pool="train"):
    """Load tasks; support old untagged paths for z_size==5."""
    if pool == "train":
        tagged = cfg.train_tasks_path
        legacy = os.path.join(
            cfg.tasks_dir, f"train-D{cfg.num_tasks}-seed{cfg.seed}.npz"
        )
        path, n, seed = tagged, cfg.num_tasks, cfg.seed
    else:
        tagged = cfg.ood_tasks_path
        legacy = os.path.join(
            cfg.tasks_dir, f"ood-D{cfg.num_ood_tasks}-seed{cfg.seed + 1000}.npz"
        )
        path, n, seed = tagged, cfg.num_ood_tasks, cfg.seed + 1000

    if os.path.exists(path):
        data = np.load(path)
        return data["g"], data["f"]
    if z_size == 5 and os.path.exists(legacy):
        data = np.load(legacy)
        return data["g"], data["f"]
    # sample with the active config sizes (caller must set Z_SIZE consistently)
    g, f = sample_tasks(n, seed)
    save_tasks(path, g, f)
    return g, f


@torch.no_grad()
def compare(checkpoint, train_h, eval_ds, z_size, device, max_examples=256, batch_size=32):
    x_off, z_off, y_off = offsets(z_size)
    model = GPTNeoXForCausalLM.from_pretrained(checkpoint)
    model.to(device).eval()

    n = min(len(eval_ds), max_examples)
    d_hM, d_hG, d_GM = [], [], []
    model_correct = 0
    total = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ids = torch.stack([eval_ds[i]["input_ids"] for i in range(start, end)])
        logits = model(input_ids=batch_ids.to(device)).logits  # (B, T, V)

        for b in range(end - start):
            ids_np = batch_ids[b].numpy()
            pairs = decode_pairs(ids_np, x_off, y_off)
            for i in range(NUM_PAIRS):
                pos = 2 * i
                y_logits = logits[b, pos, y_off : y_off + Y_SIZE].float()
                p_h = F.softmax(y_logits, dim=-1).cpu().numpy().astype(np.float64)

                prefix = pairs[:i]
                qx = pairs[i][0]
                y_true = pairs[i][1]
                p_G = atomic_G_dist(prefix, qx, Y_SIZE)
                p_M = memorizing_M_dist(prefix, qx, train_h, Y_SIZE)

                d_hM.append(sym_kl(p_h, p_M))
                d_hG.append(sym_kl(p_h, p_G))
                d_GM.append(sym_kl(p_G, p_M))

                total += 1
                if int(p_h.argmax()) == y_true:
                    model_correct += 1

    d_hM = float(np.mean(d_hM))
    d_hG = float(np.mean(d_hG))
    d_GM = float(np.mean(d_GM))
    # avoid div by 0 if G≈M
    if d_GM < 1e-12:
        r = 0.0
    else:
        r = (d_hG - d_hM) / d_GM
    r = float(np.clip(r, -1.0, 1.0))
    d_rel = (r + 1.0) / 2.0

    return {
        "n_examples": n,
        "n_positions": total,
        "model_acc": model_correct / total,
        "d(h,M)": d_hM,
        "d(h,G)": d_hG,
        "d(G,M)": d_GM,
        "r": r,
        "d_rel": d_rel,  # 0 -> G, 1 -> M
        "closer_to": "M" if d_rel > 0.5 else "G",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Wurgaft-style M/G relative distance for composition.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_tasks", "-D", type=int, required=True)
    p.add_argument("--z_size", type=int, required=True, choices=(5, 45))
    p.add_argument("--split", choices=("id", "ood", "both"), default="both")
    p.add_argument("--max_examples", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="Optional JSON path for results")
    return p.parse_args()


def main():
    args = parse_args()
    # Patch module-level Z for task sampling / config paths when creating new files.
    import config as cfgmod
    import data as datamod

    cfgmod.Z_SIZE = args.z_size
    cfgmod.Z_OFFSET = X_SIZE
    cfgmod.Y_OFFSET = X_SIZE + args.z_size
    cfgmod.VOCAB_SIZE = X_SIZE + args.z_size + Y_SIZE
    datamod.Z_SIZE = args.z_size
    datamod.Z_OFFSET = cfgmod.Z_OFFSET
    datamod.Y_OFFSET = cfgmod.Y_OFFSET

    cache_dir = args.cache_dir or read_cache_dir()
    cfg = CompositionConfig(
        num_tasks=args.num_tasks, seed=args.seed, cache_dir=cache_dir, max_steps=100_000
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  Z={args.z_size}  D={args.num_tasks}")
    print(f"checkpoint={args.checkpoint}")

    train_g, train_f = load_or_create_tasks_for_z(cfg, args.z_size, "train")
    ood_g, ood_f = load_or_create_tasks_for_z(cfg, args.z_size, "ood")
    train_h = np.stack([composite_h(train_g[d], train_f[d]) for d in range(train_g.shape[0])])

    splits = []
    if args.split in ("id", "both"):
        splits.append(
            (
                "id_comp",
                CompositionEvalDataset(
                    train_g, train_f, "comp", cfg.n_eval, cfg.seed + 1
                ),
            )
        )
    if args.split in ("ood", "both"):
        splits.append(
            (
                "ood_comp",
                CompositionEvalDataset(ood_g, ood_f, "comp", cfg.n_eval, cfg.seed + 2),
            )
        )

    all_out = {
        "checkpoint": args.checkpoint,
        "num_tasks": args.num_tasks,
        "z_size": args.z_size,
        "max_steps": cfg.max_steps,
        "splits": {},
    }
    # Infer training horizon / step from path when present
    import re as _re

    mN = _re.search(r"(\d+)steps", args.checkpoint)
    mS = _re.search(r"checkpoint-(\d+)", args.checkpoint)
    if mN:
        all_out["N"] = int(mN.group(1))
    if mS:
        all_out["step"] = int(mS.group(1))

    for name, ds in splits:
        print(f"\n=== {name} ===")
        stats = compare(
            args.checkpoint, train_h, ds, args.z_size, device, args.max_examples
        )
        all_out["splits"][name] = stats
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print(
            f"  -> d_rel={stats['d_rel']:.3f} "
            f"(0=G, 1=M) => closer to {stats['closer_to']}"
        )

    if args.out is None:
        # Default dump into analysis/ so plot_phase.py always sees new runs
        tag = f"Z{args.z_size}_D{args.num_tasks}"
        step = all_out.get("step", "final")
        args.out = os.path.join(
            cache_dir, "composition", "analysis", f"mg_rel_{tag}_step{step}.json"
        )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(all_out, fh, indent=2)
        print(f"\nWrote {args.out}")
        # Refresh plots if matplotlib is available
        try:
            from plot_phase import main as plot_main
            import sys as _sys

            _sys.argv = ["plot_phase.py"]
            plot_main()
        except Exception as exc:
            print(f"(plot_phase skipped: {exc})")


if __name__ == "__main__":
    main()
