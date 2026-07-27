"""Mechanistic probe: cross-task patching + z_decode.

Cross-task design (not same-task): source and destination residuals come
from two DISTINCT pretraining tasks. This matters for this specific task
family, not just as a stylistic choice -- with a single shared task, both
candidate answers are values that task already produces on its own for some
query, so a model that merely recognizes "this is task d" and applies its
ordinary per-task computation can land near either candidate without any
transportable z representation. Cross-task patching removes that confound:
the "routed" target f_dst(z_src) is a value NEITHER task produces on its
own, which is what makes a positive result unambiguous. See
apoorvkh/composing-functions (Khandelwal & Pavlick, arXiv:2510.01685) for
the same design choice in a different (weight-resident, non-ICL) setting.

Locks
-----
- Patch residual after layer 0 (also log layer 1)
- Cross-task: d_src != d_dst, independent queries x_src, x_dst
- Three candidates per trial: routed = f_dst(z_src_mode), transplant =
  f_src(g_src(x_src)), unaffected = f_dst(g_dst(x_dst)) -- only accepted if
  all three are argmax-distinct (the discriminating-trial filter)
- Report KL_route, KL_transplant, KL_unaffected, clean controls, z_decode
  accuracy, and trial accept rate
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    X_SIZE,
    Z_SIZE,
    Y_SIZE,
    X_OFFSET,
    Z_OFFSET,
    Y_OFFSET,
    PATCH_LAYER,
    PATCH_LAYER_LOG,
    PATCH_KL_MARGIN,
    Z_DECODE_MIN,
    N_PATCH_TRIPLES,
    NUM_PAIRS,
)
from data import build_shared_prefix_comp_sequence
from predictors import composite_marginal, sym_kl


def _layer_module(model, layer_idx: int):
    return model.gpt_neox.layers[layer_idx]


def capture_hidden(model, input_ids, layer_idx, pos, device):
    captured = {}

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[:, pos, :].detach()

    handle = _layer_module(model, layer_idx).register_forward_hook(hook)
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(device)).logits
    handle.remove()
    return logits, captured["h"]


def patched_forward(model, input_ids, layer_idx, pos, patch_vec, device):
    """Forward with residual at ``pos`` replaced after ``layer_idx``."""

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h = h.clone()
            h[:, pos, :] = patch_vec.to(h.device)
            return (h,) + out[1:]
        h = out.clone()
        h[:, pos, :] = patch_vec.to(h.device)
        return h

    handle = _layer_module(model, layer_idx).register_forward_hook(hook)
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(device)).logits
    handle.remove()
    return logits


def y_dist_from_logits(logits, query_pos) -> np.ndarray:
    """Softmax over Y slice at the position predicting the y token."""
    # logits[:, query_pos, :] predicts token at query_pos+1
    y_logits = logits[0, query_pos, Y_OFFSET : Y_OFFSET + Y_SIZE]
    p = torch.softmax(y_logits.float(), dim=-1).cpu().numpy()
    return p / p.sum()


def decode_z_lm_head(model, hidden) -> int:
    """Decode Z via LM head restricted to Z slice."""
    if hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)
    lm = getattr(model, "embed_out", None) or getattr(model, "lm_head", None)
    if lm is None:
        raise AttributeError("Model has no embed_out / lm_head for z_decode")
    with torch.no_grad():
        logits = lm(hidden)
    z_logits = logits[0, Z_OFFSET : Z_OFFSET + Z_SIZE]
    return int(z_logits.argmax().item())


def sample_cross_task_trial(
    w_g_pool: np.ndarray,
    w_f_pool: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 1000,
) -> Optional[Dict]:
    """Draw a cross-task patching trial, filtered for discriminating power.

    Only accepted if routed/transplant/unaffected are pairwise argmax-distinct
    -- the property that guarantees the routed target is a value neither task
    produces on its own, so a route-favoring patched output can't be
    explained by shortcut task-recognition alone. Returns None if no
    qualifying trial is found within max_tries (caller should skip/retry).
    """
    D = w_g_pool.shape[0]
    if D < 2:
        raise ValueError("Cross-task patching needs at least 2 pretraining tasks")

    for _ in range(max_tries):
        d_src, d_dst = (int(i) for i in rng.choice(D, size=2, replace=False))
        x_src = int(rng.integers(X_SIZE))
        x_dst = int(rng.integers(X_SIZE))

        w_g_src, w_f_src = w_g_pool[d_src], w_f_pool[d_src]
        w_g_dst, w_f_dst = w_g_pool[d_dst], w_f_pool[d_dst]

        z_src_mode = int(w_g_src[x_src].argmax())

        row = w_f_dst[z_src_mode]
        routed = row / row.sum()
        transplant = composite_marginal(w_g_src, w_f_src, x_src)
        unaffected = composite_marginal(w_g_dst, w_f_dst, x_dst)

        labels = {int(routed.argmax()), int(transplant.argmax()), int(unaffected.argmax())}
        if len(labels) == 3:
            return {
                "d_src": d_src,
                "d_dst": d_dst,
                "x_src": x_src,
                "x_dst": x_dst,
                "z_src_mode": z_src_mode,
                "w_g_src": w_g_src,
                "w_f_src": w_f_src,
                "w_g_dst": w_g_dst,
                "w_f_dst": w_f_dst,
                "routed": routed,
                "transplant": transplant,
                "unaffected": unaffected,
            }
    return None


@torch.no_grad()
def run_patching_probe(
    model,
    w_g_pool: np.ndarray,
    w_f_pool: np.ndarray,
    device: str,
    n_triples: int = N_PATCH_TRIPLES,
    layer: int = PATCH_LAYER,
    seed: int = 1,
) -> Dict[str, float]:
    """Cross-task patching: patch the source's captured residual (query
    x_src, task d_src) into the destination's forward pass (query x_dst,
    task d_dst != d_src). See module docstring for the three candidates.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    kl_route, kl_transplant, kl_unaffected = [], [], []
    kl_clean_route, kl_clean_transplant, kl_clean_unaffected = [], [], []
    z_hit = []
    n_attempted = 0
    n_accepted = 0
    max_attempts = max(n_triples * 20, 20)

    while n_accepted < n_triples and n_attempted < max_attempts:
        n_attempted += 1
        trial = sample_cross_task_trial(w_g_pool, w_f_pool, rng)
        if trial is None:
            continue

        ids_src, _, qpos_src = build_shared_prefix_comp_sequence(
            trial["w_g_src"],
            trial["w_f_src"],
            rng,
            query_x=trial["x_src"],
            num_context_pairs=NUM_PAIRS - 1,
        )
        ids_dst, _, qpos_dst = build_shared_prefix_comp_sequence(
            trial["w_g_dst"],
            trial["w_f_dst"],
            rng,
            query_x=trial["x_dst"],
            num_context_pairs=NUM_PAIRS - 1,
        )
        assert qpos_src == qpos_dst, "source/destination prefixes must be the same length"
        qpos = qpos_src

        t_src = torch.tensor(ids_src, dtype=torch.long).unsqueeze(0)
        t_dst = torch.tensor(ids_dst, dtype=torch.long).unsqueeze(0)

        _, h_src = capture_hidden(model, t_src, layer, qpos, device)
        logits_dst_clean, _ = capture_hidden(model, t_dst, layer, qpos, device)
        logits_patched = patched_forward(model, t_dst, layer, qpos, h_src, device)

        p_clean = y_dist_from_logits(logits_dst_clean, qpos)
        p_patch = y_dist_from_logits(logits_patched, qpos)

        routed, transplant, unaffected = trial["routed"], trial["transplant"], trial["unaffected"]

        kl_route.append(sym_kl(p_patch, routed))
        kl_transplant.append(sym_kl(p_patch, transplant))
        kl_unaffected.append(sym_kl(p_patch, unaffected))
        kl_clean_route.append(sym_kl(p_clean, routed))
        kl_clean_transplant.append(sym_kl(p_clean, transplant))
        kl_clean_unaffected.append(sym_kl(p_clean, unaffected))

        z_hat = decode_z_lm_head(model, h_src)
        z_hit.append(float(z_hat == trial["z_src_mode"]))

        n_accepted += 1

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    route_mean = mean(kl_route)
    transplant_mean = mean(kl_transplant)
    unaffected_mean = mean(kl_unaffected)

    out = {
        "kl_route": route_mean,
        "kl_transplant": transplant_mean,
        "kl_unaffected": unaffected_mean,
        "kl_clean_route": mean(kl_clean_route),
        "kl_clean_transplant": mean(kl_clean_transplant),
        "kl_clean_unaffected": mean(kl_clean_unaffected),
        "z_decode_acc": mean(z_hit),
        "patch_margin_ok": float(
            (route_mean + PATCH_KL_MARGIN <= transplant_mean)
            and (route_mean + PATCH_KL_MARGIN <= unaffected_mean)
        ),
        "z_decode_ok": float(mean(z_hit) >= Z_DECODE_MIN),
        "layer": layer,
        "n_triples": n_accepted,
        "n_attempted": n_attempted,
        "trial_accept_rate": n_accepted / max(n_attempted, 1),
    }
    return out


def route_margin(probe_result: Dict[str, float]) -> float:
    """How convincingly `route` beats the worse of the two controls.

    Positive means route is closer to the patched output than either
    control; larger is stronger evidence of routing. This is the score
    select_best_patch_layer optimizes over candidate layers.
    """
    return min(probe_result["kl_transplant"], probe_result["kl_unaffected"]) - probe_result[
        "kl_route"
    ]


def select_best_patch_layer(
    model,
    w_g_pool: np.ndarray,
    w_f_pool: np.ndarray,
    device: str,
    n_layers: int,
    val_seed: int = 1000,
    n_val_triples: int = 50,
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """Pick the patch layer with the strongest route-vs-controls margin on a
    VALIDATION trial set. Caller should re-score on a DISJOINT test seed at
    the selected layer for final reporting -- selecting and reporting on the
    same trials inflates the effect (same pitfall the junta-protocol
    document flags for this kind of test).
    """
    per_layer = {}
    for layer in range(n_layers):
        per_layer[layer] = run_patching_probe(
            model,
            w_g_pool,
            w_f_pool,
            device,
            n_triples=n_val_triples,
            layer=layer,
            seed=val_seed,
        )
    best_layer = max(per_layer, key=lambda l: route_margin(per_layer[l]))
    return best_layer, per_layer
