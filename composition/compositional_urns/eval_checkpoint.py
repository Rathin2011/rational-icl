"""Checkpoint evaluation: CE splits, soft d_rel, optional patching."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM

from config import (
    Y_OFFSET,
    Y_SIZE,
    NUM_PAIRS,
    D_REL_MARGIN,
    N_PATCH_TRIPLES,
)
from data import CompUrnsEvalDataset
from phase1_gate import evaluate_phase1_gate, model_ce_on_eval
from predictors import (
    approx_C_GG_predictive,
    atomic_G_predictive,
    memorizing_M_predictive,
    soft_rel_weights,
    sym_kl,
)
from relative_distance import cgg_wins
from probe import run_patching_probe, select_best_patch_layer


@torch.no_grad()
def model_y_dists_on_comp(model, dataset, device: str, max_queries_per_seq: int = 8):
    """For each comp example, collect model Y-dists at last few output slots.

    Returns list of dicts with prefix pairs (before each query), query_x, model_dist.
    """
    model.eval()
    rows = []
    for ex in dataset:
        ids = ex["input_ids"].unsqueeze(0).to(device)
        pairs = ex["pairs"]  # (x,y) length NUM_PAIRS
        logits = model(input_ids=ids).logits[0]  # (seq, V)
        # Use last max_queries_per_seq output positions
        start = max(1, NUM_PAIRS - max_queries_per_seq)
        for i in range(start, NUM_PAIRS):
            # position predicting y_i is 2*i (the x token); logits[2*i] -> token 2*i+1
            pos = 2 * i
            y_logits = logits[pos, Y_OFFSET : Y_OFFSET + Y_SIZE]
            p = torch.softmax(y_logits.float(), dim=-1).cpu().numpy()
            p = p / p.sum()
            prefix = pairs[:i]
            query_x = pairs[i][0]
            rows.append({"prefix": prefix, "query_x": query_x, "model_dist": p})
    return rows


def soft_d_rel_on_comp(
    model,
    dataset,
    train_w_g,
    train_w_f,
    device: str,
    g_concentration: float = 1.0,
    f_concentration: float = 1.0,
) -> Dict[str, float]:
    rows = model_y_dists_on_comp(model, dataset, device)
    acc = {"M": 0.0, "G": 0.0, "C_GG": 0.0}
    n = 0
    for row in rows:
        pref, qx, h = row["prefix"], row["query_x"], row["model_dist"]
        refs = {
            "M": memorizing_M_predictive(pref, qx, train_w_g, train_w_f),
            "G": atomic_G_predictive(pref, qx),
            "C_GG": approx_C_GG_predictive(
                pref, qx, g_concentration=g_concentration, f_concentration=f_concentration
            ),
        }
        d = {k: sym_kl(h, refs[k]) for k in refs}
        w = soft_rel_weights(d)
        for k in acc:
            acc[k] += w[k]
        n += 1
    return {k: acc[k] / max(n, 1) for k in acc}


def evaluate_checkpoint(
    checkpoint: str,
    train_w_g,
    train_w_f,
    ood_w_g,
    ood_w_f,
    device: str = None,
    n_eval: int = 128,
    seed: int = 1,
    do_probe: bool = True,
    out_json: Optional[str] = None,
    g_concentration: float = 1.0,
    f_concentration: float = 1.0,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTNeoXForCausalLM.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    id_g = CompUrnsEvalDataset(train_w_g, train_w_f, "g_only", n_eval, seed)
    id_f = CompUrnsEvalDataset(train_w_g, train_w_f, "f_only", n_eval, seed + 1)
    id_c = CompUrnsEvalDataset(train_w_g, train_w_f, "comp", n_eval, seed + 2)
    ood_c = CompUrnsEvalDataset(ood_w_g, ood_w_f, "comp", n_eval, seed + 3)
    ood_g = CompUrnsEvalDataset(ood_w_g, ood_w_f, "g_only", n_eval, seed + 4)
    ood_f = CompUrnsEvalDataset(ood_w_g, ood_w_f, "f_only", n_eval, seed + 5)

    gate_ok, gate = evaluate_phase1_gate(
        model,
        id_g,
        id_f,
        device,
        g_concentration=g_concentration,
        f_concentration=f_concentration,
    )
    metrics = {
        "checkpoint": checkpoint,
        "phase1_gate_passed": bool(gate_ok),
        "phase1_gate": gate,
        "ce_id_g_only": model_ce_on_eval(model, id_g, device),
        "ce_id_f_only": model_ce_on_eval(model, id_f, device),
        "ce_id_comp": model_ce_on_eval(model, id_c, device),
        "ce_ood_comp": model_ce_on_eval(model, ood_c, device),
        "ce_ood_g_only": model_ce_on_eval(model, ood_g, device),
        "ce_ood_f_only": model_ce_on_eval(model, ood_f, device),
    }

    # Soft d_rel on a smaller subset for speed
    id_c_small = CompUrnsEvalDataset(train_w_g, train_w_f, "comp", min(64, n_eval), seed + 10)
    ood_c_small = CompUrnsEvalDataset(ood_w_g, ood_w_f, "comp", min(64, n_eval), seed + 11)
    d_id = soft_d_rel_on_comp(
        model, id_c_small, train_w_g, train_w_f, device, g_concentration, f_concentration
    )
    d_ood = soft_d_rel_on_comp(
        model, ood_c_small, train_w_g, train_w_f, device, g_concentration, f_concentration
    )
    metrics["d_rel_id"] = d_id
    metrics["d_rel_ood"] = d_ood
    metrics["cgg_wins_id"] = cgg_wins(d_id, D_REL_MARGIN)
    metrics["cgg_wins_ood"] = cgg_wins(d_ood, D_REL_MARGIN)

    if do_probe:
        # Probe on train-pool tasks (known factors for route/short targets).
        # Select patch layer on a VALIDATION seed, report on a disjoint TEST
        # seed at that layer -- selecting and reporting on the same trials
        # inflates the effect (see probe.select_best_patch_layer docstring).
        n_layers = getattr(model.config, "num_hidden_layers", 1)
        best_layer, val_by_layer = select_best_patch_layer(
            model,
            train_w_g,
            train_w_f,
            device,
            n_layers=n_layers,
            val_seed=seed + 500,
            n_val_triples=min(N_PATCH_TRIPLES, 50),
        )
        metrics["probe"] = run_patching_probe(
            model,
            train_w_g,
            train_w_f,
            device,
            n_triples=min(N_PATCH_TRIPLES, 100),
            layer=best_layer,
            seed=seed,
        )
        metrics["probe"]["layer_selected_on_validation"] = best_layer
        metrics["probe_layer_sweep_validation"] = val_by_layer

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Wrote {out_json}")

    return metrics


if __name__ == "__main__":
    import argparse
    from config import CompUrnsConfig
    from data import get_or_create_tasks

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_tasks", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_eval", type=int, default=128)
    p.add_argument("--no_probe", action="store_true")
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--g_concentration", type=float, default=1.0)
    p.add_argument("--f_concentration", type=float, default=1.0)
    args = p.parse_args()

    cfg = CompUrnsConfig(
        num_tasks=args.num_tasks,
        seed=args.seed,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
    )
    twg, twf = get_or_create_tasks(
        cfg.train_tasks_path,
        cfg.num_tasks,
        cfg.seed,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
    )
    owg, owf = get_or_create_tasks(
        cfg.ood_tasks_path,
        cfg.num_ood_tasks,
        cfg.seed + 1000,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
    )
    evaluate_checkpoint(
        args.checkpoint,
        twg,
        twf,
        owg,
        owf,
        n_eval=args.n_eval,
        seed=args.seed,
        do_probe=not args.no_probe,
        out_json=args.out_json,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
    )
