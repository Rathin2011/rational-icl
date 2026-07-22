"""Mechanistic probe: shared-prefix patching + z_decode (PROTOCOL.md).

Locks
-----
- Patch residual after layer 0 (also log layer 1)
- Same ICL prefix for x_a and x_b; only query x differs
- Targets are full composite marginals Σ_z w_g[x,z] w_f[z,:]
- Report KL_route, KL_short, clean controls, z_decode accuracy
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


def sample_diff_mode_pair(w_g: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    """Sample (x_a, x_b) with different argmax_z w_g[x]."""
    for _ in range(1000):
        xa = int(rng.integers(X_SIZE))
        xb = int(rng.integers(X_SIZE))
        if xa == xb:
            continue
        if int(w_g[xa].argmax()) != int(w_g[xb].argmax()):
            return xa, xb
    # fallback
    return 0, 1


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
    """Average KL_route / KL_short / clean controls / z_decode over triples."""
    model.eval()
    rng = np.random.default_rng(seed)
    D = w_g_pool.shape[0]

    kl_route = []
    kl_short = []
    kl_clean_route = []
    kl_clean_short = []
    z_hit = []

    for _ in range(n_triples):
        d = int(rng.integers(D))
        w_g, w_f = w_g_pool[d], w_f_pool[d]
        xa, xb = sample_diff_mode_pair(w_g, rng)

        # Shared context RNG stream: build prefix once, reuse ids skeleton
        ids_a, pairs, qpos = build_shared_prefix_comp_sequence(
            w_g, w_f, rng, query_x=xa, num_context_pairs=NUM_PAIRS - 1
        )
        ids_b = ids_a.copy()
        ids_b[qpos] = X_OFFSET + xb

        t_a = torch.tensor(ids_a, dtype=torch.long).unsqueeze(0)
        t_b = torch.tensor(ids_b, dtype=torch.long).unsqueeze(0)

        logits_a, h_a = capture_hidden(model, t_a, layer, qpos, device)
        logits_b, h_b = capture_hidden(model, t_b, layer, qpos, device)

        # Patch b's residual into a's forward
        logits_p = patched_forward(model, t_a, layer, qpos, h_b, device)

        p_clean = y_dist_from_logits(logits_a, qpos)
        p_patch = y_dist_from_logits(logits_p, qpos)

        route = composite_marginal(w_g, w_f, xa)
        short = composite_marginal(w_g, w_f, xb)

        # After patching with h_b, routing target is f(g(x_a))? PROTOCOL:
        # KL_route: patched → marginal f(g(x_a))  (routing: still apply f to patched z from b?)
        # Re-read PROTOCOL from user + design review:
        #   KL_route: patched output to marginal f(g(x_a))
        #   KL_short: patched output to marginal f(g(x_b))
        # If we patch x_b residual into x_a run, C_GG should produce f(g(x_b)) = short.
        # Wait - user's original:
        #   "KL_route: KL from patched output to the marginal f(g(x_a)) (routing target)"
        #   "KL_short: KL from patched output to the marginal f(g(x_b)) (shortcut target)"
        # And success: KL_route < KL_short
        #
        # Lookuptable probe patches x_b into x_a and expects y_b (composition).
        # So for C_GG, patched should match short = f(g(x_b)), meaning KL_short should be SMALLER.
        # User said KL_route < KL_short for success — that would mean closer to f(g(x_a)) after
        # patching in x_b's residual, which is the opposite of composition!
        #
        # Re-read user message carefully:
        # 1. Forward with x_a, save residual at x_a
        # 2. Forward with x_b, patch in the saved residual (from x_a) at same position
        # 3. KL_route to f(g(x_a)), KL_short to f(g(x_b))
        # Success: KL_route < KL_short
        #
        # So they patch x_a's residual INTO x_b's forward. Then routing keeps z_a → f(g(x_a)).
        # I'll follow the user's order: save from a, patch into b's run.

        # Redo with user's order: patch h_a into forward on ids_b
        logits_p = patched_forward(model, t_b, layer, qpos, h_a, device)
        p_patch = y_dist_from_logits(logits_p, qpos)
        p_clean_b = y_dist_from_logits(logits_b, qpos)

        kl_route.append(sym_kl(p_patch, route))
        kl_short.append(sym_kl(p_patch, short))
        kl_clean_route.append(sym_kl(p_clean_b, route))
        kl_clean_short.append(sym_kl(p_clean_b, short))

        z_hat = decode_z_lm_head(model, h_a)
        z_true = int(w_g[xa].argmax())  # mode; soft tasks — report vs mode
        z_hit.append(float(z_hat == z_true))

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    out = {
        "kl_route": mean(kl_route),
        "kl_short": mean(kl_short),
        "kl_clean_route": mean(kl_clean_route),
        "kl_clean_short": mean(kl_clean_short),
        "z_decode_acc": mean(z_hit),
        "patch_margin_ok": float(
            mean(kl_route) + PATCH_KL_MARGIN <= mean(kl_short)
        ),
        "z_decode_ok": float(mean(z_hit) >= Z_DECODE_MIN),
        "layer": layer,
        "n_triples": n_triples,
    }
    return out
