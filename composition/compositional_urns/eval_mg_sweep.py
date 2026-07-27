"""Behavioral M vs G relative-distance sweep (Wurgaft-style, comp-only baseline).

G here is `approx_C_GG_predictive` -- the Bayes-optimal population-level
generalizing predictor (g, f are literally Dirichlet-distributed in the
generative model, so this is the true posterior predictive marginal, not an
approximation to some other object). This is deliberately NOT
`atomic_G_predictive`, which PROTOCOL.md documents as a misspecified stand-in
that ignores the g/f factor structure -- see relative_distance.py's docstring.
No patching, no C_GG-as-a-third-category: this script only asks which of the
two exact behavioral predictors (M, G) a trained model's predictive
distribution is closer to, at each checkpoint and prefix length.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

from config import CompUrnsConfig, X_SIZE, Y_OFFSET, Y_SIZE
from data import build_shared_prefix_comp_sequence, get_or_create_tasks
from predictors import approx_C_GG_predictive, memorizing_M_predictive, sym_kl

PREFIX_LENGTHS = (2, 4, 8, 16, 32)
TV_MARGIN = 0.05


def total_variation(p, q) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


@torch.no_grad()
def model_query_dist(model, ids, qpos, device) -> np.ndarray:
    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(input_ids=ids_t).logits[0]
    y_logits = logits[qpos, Y_OFFSET : Y_OFFSET + Y_SIZE]
    p = torch.softmax(y_logits.float(), dim=-1).cpu().numpy()
    return p / p.sum()


def score_split(
    model,
    train_w_g,
    train_w_f,
    split_w_g,
    split_w_f,
    split_name: str,
    n_sequences: int,
    seed: int,
    device: str,
    g_concentration: float = 1.0,
    f_concentration: float = 1.0,
) -> list:
    """M is always scored against the training pool; G needs no pool."""
    rng = np.random.default_rng(seed)
    d_split = split_w_g.shape[0]
    rows = []
    for L in PREFIX_LENGTHS:
        for _ in range(n_sequences):
            d = int(rng.integers(d_split))
            query_x = int(rng.integers(X_SIZE))
            ids, pairs, qpos = build_shared_prefix_comp_sequence(
                split_w_g[d], split_w_f[d], rng, query_x, num_context_pairs=L
            )
            p_theta = model_query_dist(model, ids, qpos, device)
            m = memorizing_M_predictive(pairs, query_x, train_w_g, train_w_f)
            g = approx_C_GG_predictive(
                pairs, query_x, g_concentration=g_concentration, f_concentration=f_concentration
            )
            tv_mg = total_variation(m, g)
            kl_m = sym_kl(p_theta, m)
            kl_g = sym_kl(p_theta, g)
            d_rel_m = kl_m / max(kl_m + kl_g, 1e-12)
            rows.append(
                {
                    "split": split_name,
                    "L": L,
                    "tv_mg": tv_mg,
                    "retained": tv_mg > TV_MARGIN,
                    "d_rel_m": d_rel_m,
                }
            )
    return rows


def parse_step(ckpt_dir: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", ckpt_dir)
    return int(m.group(1)) if m else -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_tasks", "-D", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run_tag", type=str, default="comp-only")
    p.add_argument("--phase", type=str, default="baseline")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--g_concentration", type=float, default=1.0)
    p.add_argument("--f_concentration", type=float, default=1.0)
    p.add_argument("--n_sequences", type=int, default=64)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()

    # batch_size/learning_rate/concentration must match the training run's
    # config -- all feed into CompUrnsConfig.run_name() and hence the
    # checkpoint path.
    cfg = CompUrnsConfig(
        num_tasks=args.num_tasks,
        seed=args.seed,
        run_tag=args.run_tag,
        phase=args.phase,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
        cache_dir=args.cache_dir,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_w_g, train_w_f = get_or_create_tasks(
        cfg.train_tasks_path,
        cfg.num_tasks,
        cfg.seed,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
    )
    ood_w_g, ood_w_f = get_or_create_tasks(
        cfg.ood_tasks_path,
        cfg.num_ood_tasks,
        cfg.seed + 1000,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
    )

    ckpt_dirs = sorted(
        glob.glob(os.path.join(cfg.checkpoints_dir, "checkpoint-*")), key=parse_step
    )
    if not ckpt_dirs:
        raise SystemExit(f"No checkpoints found under {cfg.checkpoints_dir}")

    out_csv = args.out_csv or os.path.join(cfg.run_path, "mg_sweep.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "num_tasks",
        "seed",
        "step",
        "split",
        "L",
        "tv_mg",
        "retained",
        "d_rel_m",
    ]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ckpt_dir in ckpt_dirs:
            step = parse_step(ckpt_dir)
            print(f"Evaluating step {step} ({ckpt_dir})")
            model = GPTNeoXForCausalLM.from_pretrained(ckpt_dir)
            model.to(device)
            model.eval()
            for split_name, sw_g, sw_f, seed_off in (
                ("id", train_w_g, train_w_f, 100),
                ("ood", ood_w_g, ood_w_f, 200),
            ):
                rows = score_split(
                    model,
                    train_w_g,
                    train_w_f,
                    sw_g,
                    sw_f,
                    split_name,
                    args.n_sequences,
                    seed=args.seed + seed_off,
                    device=device,
                    g_concentration=args.g_concentration,
                    f_concentration=args.f_concentration,
                )
                for r in rows:
                    r["num_tasks"] = args.num_tasks
                    r["seed"] = args.seed
                    r["step"] = step
                    writer.writerow(r)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
