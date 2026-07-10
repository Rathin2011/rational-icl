"""Intermediate-variable composition probe.

Tests whether the model computes x -> z -> y (composition) rather than a flat
x -> y map.

Protocol
--------
1. Build an ICL context of compositional (x, y) pairs for a fixed task (g, f).
2. Query with x_a (true z_a = g(x_a), y_a = f(z_a)).
3. Query with x_b != x_a (true z_b, y_b).
4. At a bridge layer L, at the query-x position, patch the residual from the
   x_b run into the x_a run and check whether the predicted Y becomes y_b.
   That means the patched state carried z_b and the rest of the net applied f.
5. Also decode Z from the residual at the query position (LM head restricted to
   Z tokens) and check whether f(z_hat) equals the true y for that query.

Usage (from composition/):
    python probe_composition.py --checkpoint /path/to/checkpoint-XXXX \\
        --num_tasks 64 --n_trials 200
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

from config import (
    X_SIZE,
    Z_SIZE,
    Y_SIZE,
    X_OFFSET,
    Z_OFFSET,
    Y_OFFSET,
    NUM_PAIRS,
    CompositionConfig,
    read_cache_dir,
)
from data import get_or_create_tasks, build_comp_probe_sequence


def load_model(checkpoint, device):
    model = GPTNeoXForCausalLM.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return model


def _layer_module(model, layer_idx):
    return model.gpt_neox.layers[layer_idx]


def capture_hidden(model, input_ids, layer_idx, pos, device):
    """Forward pass; return residual stream at ``pos`` after layer ``layer_idx``."""
    captured = {}

    def hook(_module, _inp, out):
        # GPTNeoX layer returns (hidden_states, ...) or just hidden_states
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[:, pos, :].detach()

    handle = _layer_module(model, layer_idx).register_forward_hook(hook)
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(device)).logits
    handle.remove()
    return logits, captured["h"]


def predict_y_from_logits(logits, query_pos):
    """Argmax Y prediction from logits at the query-x position."""
    # logits[b, query_pos, :] predicts the token at query_pos+1 (the y slot)
    y_logits = logits[:, query_pos, Y_OFFSET : Y_OFFSET + Y_SIZE]
    return int(y_logits.argmax(dim=-1).item())


def decode_z_from_hidden(model, hidden):
    """Decode a Z token from a residual vector via the LM head (Z slice only)."""
    # hidden: (1, d_model) or (d_model,)
    if hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)
    with torch.no_grad():
        # GPTNeoX: final layernorm then embed_out
        h = model.gpt_neox.final_layer_norm(hidden)
        logits = model.embed_out(h)  # (1, vocab)
    z_logits = logits[:, Z_OFFSET : Z_OFFSET + Z_SIZE]
    return int(z_logits.argmax(dim=-1).item())


def patched_forward(model, input_ids, layer_idx, pos, patch_hidden, device):
    """Forward with residual at (layer_idx, pos) replaced by ``patch_hidden``."""

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            h = out[0].clone()
            h[:, pos, :] = patch_hidden.to(h.device)
            return (h,) + out[1:]
        h = out.clone()
        h[:, pos, :] = patch_hidden.to(h.device)
        return h

    handle = _layer_module(model, layer_idx).register_forward_hook(hook)
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(device)).logits
    handle.remove()
    return logits


def sample_probe_trial(g_row, f_row, rng):
    """Sample context xs and two distinct query xs with preferably different z."""
    context_xs = rng.integers(0, X_SIZE, size=NUM_PAIRS - 1)
    # Prefer queries with different intermediates when possible
    for _ in range(50):
        x_a, x_b = rng.choice(X_SIZE, size=2, replace=False)
        if int(g_row[x_a]) != int(g_row[x_b]):
            break
    return context_xs, int(x_a), int(x_b)


def run_one_trial(model, g_row, f_row, context_xs, x_a, x_b, layer_idx, device):
    """Run clean predictions, Z-decode, and patching for one (x_a, x_b) pair."""
    ids_a, qpos, z_a, y_a = build_comp_probe_sequence(g_row, f_row, context_xs, x_a)
    ids_b, _, z_b, y_b = build_comp_probe_sequence(g_row, f_row, context_xs, x_b)

    t_a = torch.tensor(ids_a, dtype=torch.long).unsqueeze(0)
    t_b = torch.tensor(ids_b, dtype=torch.long).unsqueeze(0)

    logits_a, h_a = capture_hidden(model, t_a, layer_idx, qpos, device)
    logits_b, h_b = capture_hidden(model, t_b, layer_idx, qpos, device)

    pred_y_a = predict_y_from_logits(logits_a, qpos)
    pred_y_b = predict_y_from_logits(logits_b, qpos)

    # Decode z' from residual at query position on the x_b run
    z_hat_b = decode_z_from_hidden(model, h_b)
    y_from_f_zhat = int(f_row[z_hat_b])

    # Patch h_b into the x_a forward at the same layer/position
    logits_patch = patched_forward(model, t_a, layer_idx, qpos, h_b, device)
    pred_y_patch = predict_y_from_logits(logits_patch, qpos)

    return {
        "z_a": z_a,
        "y_a": y_a,
        "z_b": z_b,
        "y_b": y_b,
        "pred_y_a": pred_y_a,
        "pred_y_b": pred_y_b,
        "z_hat_b": z_hat_b,
        "y_from_f_zhat": y_from_f_zhat,
        "pred_y_patch": pred_y_patch,
        "clean_a_correct": pred_y_a == y_a,
        "clean_b_correct": pred_y_b == y_b,
        "z_decode_correct": z_hat_b == z_b,
        "f_of_zhat_correct": y_from_f_zhat == y_b,
        # Patch success: after inserting x_b's intermediate, output becomes y_b
        "patch_to_y_b": pred_y_patch == y_b,
        # Stronger: patch changes prediction away from y_a toward y_b when y_a != y_b
        "patch_flips_when_needed": (
            (y_a == y_b) or (pred_y_patch == y_b and pred_y_a != y_b)
        ),
        "different_z": z_a != z_b,
        "different_y": y_a != y_b,
    }


def summarize(results):
    n = len(results)
    if n == 0:
        return {}

    def rate(key, subset=None):
        rows = results if subset is None else [r for r in results if subset(r)]
        if not rows:
            return float("nan")
        return float(np.mean([r[key] for r in rows]))

    diff_z = [r for r in results if r["different_z"]]
    diff_y = [r for r in results if r["different_y"]]

    return {
        "n_trials": n,
        "n_different_z": len(diff_z),
        "n_different_y": len(diff_y),
        "clean_acc_a": rate("clean_a_correct"),
        "clean_acc_b": rate("clean_b_correct"),
        "z_decode_acc": rate("z_decode_correct"),
        "f_of_zhat_acc": rate("f_of_zhat_correct"),
        "patch_to_y_b": rate("patch_to_y_b"),
        "patch_to_y_b_given_diff_z": rate("patch_to_y_b", lambda r: r["different_z"]),
        "patch_to_y_b_given_diff_y": rate("patch_to_y_b", lambda r: r["different_y"]),
        "chance_y": 1.0 / Y_SIZE,
        "chance_z": 1.0 / Z_SIZE,
    }


def run_probe(model, g, f, n_trials, layer_idx, seed, device, task_pool="train"):
    rng = np.random.default_rng(seed)
    results = []
    num_tasks = g.shape[0]
    for _ in range(n_trials):
        d = int(rng.integers(num_tasks))
        context_xs, x_a, x_b = sample_probe_trial(g[d], f[d], rng)
        trial = run_one_trial(
            model, g[d], f[d], context_xs, x_a, x_b, layer_idx, device
        )
        trial["task"] = d
        trial["x_a"] = x_a
        trial["x_b"] = x_b
        results.append(trial)
    summary = summarize(results)
    summary["layer"] = layer_idx
    summary["task_pool"] = task_pool
    return summary, results


def parse_args():
    p = argparse.ArgumentParser(description="Intermediate-variable composition probe.")
    p.add_argument("--checkpoint", type=str, required=True, help="HF checkpoint dir")
    p.add_argument("--num_tasks", "-D", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_trials", type=int, default=200)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Bridge layers to patch (default: all layers)",
    )
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument(
        "--task_pool",
        choices=("train", "ood"),
        default="train",
        help="Probe on training tasks or held-out OOD tasks",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON path for summary + per-trial results",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir or read_cache_dir()
    cfg = CompositionConfig(
        num_tasks=args.num_tasks, seed=args.seed, cache_dir=cache_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model = load_model(args.checkpoint, device)
    n_layers = model.config.num_hidden_layers
    layers = args.layers if args.layers is not None else list(range(n_layers))

    if args.task_pool == "train":
        g, f = get_or_create_tasks(cfg.train_tasks_path, cfg.num_tasks, cfg.seed)
    else:
        g, f = get_or_create_tasks(
            cfg.ood_tasks_path, cfg.num_ood_tasks, cfg.seed + 1000
        )

    all_summaries = []
    for layer in layers:
        print(f"\n=== Layer {layer} ({args.task_pool} tasks) ===")
        summary, results = run_probe(
            model,
            g,
            f,
            n_trials=args.n_trials,
            layer_idx=layer,
            seed=args.seed + 17 + layer,
            device=device,
            task_pool=args.task_pool,
        )
        all_summaries.append({"summary": summary, "n_results": len(results)})
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Composition signature (brief)
        patch = summary["patch_to_y_b_given_diff_y"]
        f_z = summary["f_of_zhat_acc"]
        print(
            f"  -> composition signal: patch_to_y_b|diff_y={patch:.3f} "
            f"(chance_y={summary['chance_y']:.3f}), "
            f"f(z_hat)==y'={f_z:.3f}"
        )

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Re-run last layer results already in memory only for last; dump summaries
        payload = {
            "checkpoint": args.checkpoint,
            "num_tasks": args.num_tasks,
            "task_pool": args.task_pool,
            "n_trials": args.n_trials,
            "layers": all_summaries,
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
