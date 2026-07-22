"""Aggregate M/G (and later C_GG) relative-distance results and plot vs D, N.

Reads JSON dumps from compare_predictors.py under
  $CACHE_DIR/composition/analysis/mg_rel_*.json
and writes:
  analysis/predictor_results.csv   — flat table of all scored checkpoints
  analysis/figs/d_rel_vs_D_*.png   — ID d_rel vs D (0=G, 1=M)
  analysis/figs/phase_scatter_*.png — (D, N) colored by closer_to

C_GG is reserved as a third label once a closed-form C_GG scorer exists;
rows without it still plot on the M–G axis.

Usage (from composition/):
  python plot_phase.py
  python plot_phase.py --z_size 5
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from config import read_cache_dir


def parse_checkpoint_meta(checkpoint):
    """Best-effort D, N, step, z_size from a run/checkpoint path."""
    meta = {"D": None, "N": None, "step": None, "z_size": None}
    if not checkpoint:
        return meta
    m = re.search(r"/D(\d+)", checkpoint)
    if m:
        meta["D"] = int(m.group(1))
    m = re.search(r"-Z(\d+)-", checkpoint)
    if m:
        meta["z_size"] = int(m.group(1))
    elif re.search(r"/D\d+-8L-", checkpoint) or re.search(r"/D\d+-2L-", checkpoint):
        # Legacy untagged bottleneck runs were |Z|=5
        meta["z_size"] = 5
    m = re.search(r"(\d+)steps", checkpoint)
    if m:
        meta["N"] = int(m.group(1))
    m = re.search(r"checkpoint-(\d+)", checkpoint)
    if m:
        meta["step"] = int(m.group(1))
    return meta


def load_json_rows(analysis_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(analysis_dir, "mg_rel_*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        meta = parse_checkpoint_meta(data.get("checkpoint", ""))
        D = data.get("num_tasks", meta["D"])
        z = data.get("z_size", meta["z_size"])
        N = meta["N"]
        step = meta["step"]
        for split, stats in (data.get("splits") or {}).items():
            rows.append(
                {
                    "source_json": os.path.basename(path),
                    "checkpoint": data.get("checkpoint", ""),
                    "D": D,
                    "N": N if N is not None else "",
                    "step": step if step is not None else "",
                    "z_size": z,
                    "split": split,
                    "model_acc": stats.get("model_acc"),
                    "d_hM": stats.get("d(h,M)"),
                    "d_hG": stats.get("d(h,G)"),
                    "d_GM": stats.get("d(G,M)"),
                    "d_rel": stats.get("d_rel"),  # 0=G, 1=M
                    "closer_to": stats.get("closer_to"),
                    "d_rel_CGG": "",  # placeholder for future C_GG axis
                    "closer_3way": stats.get("closer_to"),  # until C_GG exists
                }
            )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _finite(rows, key):
    out = []
    for r in rows:
        try:
            v = float(r[key])
            if np.isfinite(v):
                out.append((r, v))
        except (TypeError, ValueError):
            continue
    return out


def plot_d_rel_vs_D(rows, fig_dir, z_size=None, split="id_comp"):
    """Line/scatter: d_rel vs D for one split, optionally one z_size."""
    subset = [
        r
        for r in rows
        if r["split"] == split and (z_size is None or int(r["z_size"]) == int(z_size))
    ]
    by_z = {}
    for r in subset:
        by_z.setdefault(int(r["z_size"]), []).append(r)

    if not by_z:
        print(f"No rows for split={split} z={z_size}")
        return None

    n_panels = len(by_z)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2), squeeze=False)
    for ax, (z, group) in zip(axes[0], sorted(by_z.items())):
        pts = []
        for r in group:
            try:
                pts.append((int(r["D"]), float(r["d_rel"]), r.get("N") or "?"))
            except (TypeError, ValueError):
                continue
        pts.sort(key=lambda t: t[0])
        if not pts:
            continue
        Ds = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(Ds, ys, "-o", color="#1f4e79", markersize=7, label="d_rel (0=G, 1=M)")
        for D, y, N in pts:
            ax.annotate(
                f"N={N}",
                (D, y),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
                color="#444",
            )
        ax.axhline(0.5, color="#888", ls="--", lw=1, label="M/G midpoint")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ds)
        ax.set_xticklabels([str(d) for d in Ds])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("D (num tasks)")
        ax.set_ylabel(r"$d_{\mathrm{rel}}$  (0 ≈ G,  1 ≈ M)")
        ax.set_title(f"|Z|={z}  ·  {split}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    tag = f"Z{z_size}" if z_size is not None else "allZ"
    out = os.path.join(fig_dir, f"d_rel_vs_D_{tag}_{split}.png")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_phase_scatter(rows, fig_dir, split="id_comp"):
    """(D, N) scatter colored by closer_to (M vs G; C_GG when available)."""
    subset = [r for r in rows if r["split"] == split]
    pts = []
    for r in subset:
        try:
            D = int(r["D"])
            N = int(r["N"]) if r["N"] not in ("", None) else None
            if N is None:
                continue
            z = int(r["z_size"])
            label = r.get("closer_3way") or r.get("closer_to") or "?"
            d_rel = float(r["d_rel"])
            pts.append((D, N, z, label, d_rel))
        except (TypeError, ValueError):
            continue
    if not pts:
        print(f"No (D,N) points for phase scatter ({split})")
        return None

    colors = {"G": "#2a9d8f", "M": "#e76f51", "C_GG": "#264653", "?": "#999"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label in sorted(set(p[3] for p in pts)):
        xs = [p[0] for p in pts if p[3] == label]
        ys = [p[1] for p in pts if p[3] == label]
        ax.scatter(
            xs,
            ys,
            s=70,
            c=colors.get(label, "#999"),
            label=label,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
    for D, N, z, label, d_rel in pts:
        ax.annotate(
            f"Z{z}\n{d_rel:.2f}",
            (D, N),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("D (num tasks)")
    ax.set_ylabel("N (max_steps)")
    ax.set_title(f"Predictor regime map · {split}\n(color = closer_to; annotation = d_rel)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="closer to")
    fig.tight_layout()
    out = os.path.join(fig_dir, f"phase_scatter_{split}.png")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Plot M/G/C_GG phase results vs D, N.")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--z_size", type=int, default=None, help="Optional filter for d_rel vs D")
    p.add_argument("--split", default="id_comp", help="Split to emphasize (default id_comp)")
    return p.parse_args()


def main():
    args = parse_args()
    cache = args.cache_dir or read_cache_dir()
    analysis = os.path.join(cache, "composition", "analysis")
    fig_dir = os.path.join(analysis, "figs")

    rows = load_json_rows(analysis)
    if not rows:
        raise SystemExit(f"No mg_rel_*.json under {analysis}")

    csv_path = os.path.join(analysis, "predictor_results.csv")
    write_csv(csv_path, rows)
    print(f"Wrote {csv_path}  ({len(rows)} rows)")

    # Always make both Z panels for the requested split, plus per-z if filtered
    plot_d_rel_vs_D(rows, fig_dir, z_size=None, split=args.split)
    if args.z_size is not None:
        plot_d_rel_vs_D(rows, fig_dir, z_size=args.z_size, split=args.split)
    else:
        for z in sorted({int(r["z_size"]) for r in rows if r["z_size"] not in ("", None)}):
            plot_d_rel_vs_D(rows, fig_dir, z_size=z, split=args.split)

    plot_phase_scatter(rows, fig_dir, split=args.split)
    # Also dump ood companion curves
    if args.split == "id_comp":
        plot_d_rel_vs_D(rows, fig_dir, z_size=None, split="ood_comp")
        plot_phase_scatter(rows, fig_dir, split="ood_comp")


if __name__ == "__main__":
    main()
