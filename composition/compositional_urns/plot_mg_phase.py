"""Aggregate eval_mg_sweep.py outputs into a Wurgaft-style (N, D) phase diagram."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CompUrnsConfig


def load_all(d_values, seeds, run_tag, phase, cache_dir, batch_size, learning_rate) -> pd.DataFrame:
    frames = []
    for D in d_values:
        for seed in seeds:
            cfg = CompUrnsConfig(
                num_tasks=D,
                seed=seed,
                run_tag=run_tag,
                phase=phase,
                batch_size=batch_size,
                learning_rate=learning_rate,
                cache_dir=cache_dir,
            )
            path = os.path.join(cfg.run_path, "mg_sweep.csv")
            if not os.path.exists(path):
                print(f"missing: {path}")
                continue
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No mg_sweep.csv files found for the requested D/seed grid.")
    return pd.concat(frames, ignore_index=True)


def aggregate_grid(df: pd.DataFrame, split: str, L: int):
    """Reduce raw per-sequence rows to a (D x step) grid of mean retained d_rel_m.

    Returns (steps, d_values, grid, retention) where `grid[i,j]` is the mean
    d_rel_m for d_values[i] at steps[j] (NaN if that cell has no retained
    rows), and `retention` is a Series of retention_rate indexed by
    (num_tasks, step) -- computed over ALL rows, not just retained ones, so a
    low-retention cell doesn't silently vanish from the grid without a trace.
    """
    sub = df[(df["split"] == split) & (df["L"] == L)]
    retention = (
        sub.groupby(["num_tasks", "step"])["retained"].mean().rename("retention_rate")
    )
    kept = sub[sub["retained"]]
    agg = kept.groupby(["num_tasks", "step"])["d_rel_m"].mean().reset_index()

    steps = sorted(agg["step"].unique())
    d_values = sorted(agg["num_tasks"].unique())
    grid = np.full((len(d_values), len(steps)), np.nan)
    for i, D in enumerate(d_values):
        for j, s in enumerate(steps):
            row = agg[(agg["num_tasks"] == D) & (agg["step"] == s)]
            if len(row):
                grid[i, j] = row["d_rel_m"].values[0]
    return steps, d_values, grid, retention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_values", type=int, nargs="+", default=[4, 16, 64, 256, 1024])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--run_tag", type=str, default="comp-only")
    p.add_argument("--phase", type=str, default="baseline")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="ood", choices=["id", "ood"])
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--out", type=str, default="mg_phase.png")
    args = p.parse_args()

    df = load_all(
        args.d_values,
        args.seeds,
        args.run_tag,
        args.phase,
        args.cache_dir,
        args.batch_size,
        args.learning_rate,
    )

    steps, d_values, grid, retention = aggregate_grid(df, args.split, args.L)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45)
    ax.set_yticks(range(len(d_values)))
    ax.set_yticklabels(d_values)
    ax.set_xlabel("training steps (N)")
    ax.set_ylabel("task diversity (D)")
    ax.set_title(f"d_rel_M ({args.split}, L={args.L}) -- 0=memorizing, 1=generalizing")
    fig.colorbar(im, ax=ax, label="d_rel_M")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    print("\nRetention rate (fraction with TV(M,G) > 0.05), split/L/D/step:")
    print(retention.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
