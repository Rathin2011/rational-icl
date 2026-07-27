"""Correctness checks for probe.py's cross-task activation-patching mechanics.

Two groups:
1. Low-level hook mechanics (capture_hidden / patched_forward) -- unaffected
   by same-task vs cross-task trial construction, so these use arbitrary
   two-query sequences directly rather than sample_cross_task_trial.
2. Cross-task trial construction (sample_cross_task_trial / run_patching_probe)
   -- the discriminating-trial filter and the end-to-end pipeline.

The gold-standard low-level check: patching a position with ITS OWN
unpatched value must reproduce the unpatched forward pass exactly
(deterministic eval mode, no dropout). If that doesn't hold, the hook is
grabbing or injecting the wrong layer, position, or tensor -- and every
KL_route/KL_transplant/KL_unaffected number downstream is unreliable
regardless of what the task looks like.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CompUrnsConfig, X_OFFSET, Z_SIZE
from data import build_shared_prefix_comp_sequence, sample_tasks
from model import build_model
from probe import (
    capture_hidden,
    decode_z_lm_head,
    patched_forward,
    route_margin,
    run_patching_probe,
    sample_cross_task_trial,
    select_best_patch_layer,
)


def _tiny_model_and_task(n_layers=2):
    cfg = CompUrnsConfig(
        num_tasks=1,
        seed=0,
        n_layers=n_layers,
        n_heads=1,
        d_model=16,
        d_ff=32,
        max_position_embeddings=128,
    )
    model = build_model(cfg)
    model.eval()
    w_g, w_f = sample_tasks(1, seed=0)
    return model, w_g[0], w_f[0]


def _build_pair(w_g, w_f, seed=0, num_context_pairs=8, xa=0, xb=1):
    """Two queries against the SAME sequence, for testing raw hook mechanics
    only -- not a cross-task trial (see sample_cross_task_trial for that)."""
    rng = np.random.default_rng(seed)
    ids_a, pairs, qpos = build_shared_prefix_comp_sequence(
        w_g, w_f, rng, query_x=xa, num_context_pairs=num_context_pairs
    )
    ids_b = ids_a.copy()
    ids_b[qpos] = X_OFFSET + xb
    return ids_a, ids_b, qpos, xa, xb


# --- low-level hook mechanics (layer/position-agnostic to trial design) ---


def test_ids_differ_only_at_query_position():
    model, w_g, w_f = _tiny_model_and_task()
    ids_a, ids_b, qpos, xa, xb = _build_pair(w_g, w_f)
    diff_positions = np.where(ids_a != ids_b)[0]
    assert list(diff_positions) == [qpos]
    assert xa != xb


def test_capture_hidden_matches_manual_forward():
    model, w_g, w_f = _tiny_model_and_task()
    ids_a, ids_b, qpos, xa, xb = _build_pair(w_g, w_f)
    t_a = torch.tensor(ids_a, dtype=torch.long).unsqueeze(0)

    logits_hook, h_hook = capture_hidden(model, t_a, layer_idx=0, pos=qpos, device="cpu")

    with torch.no_grad():
        out = model(input_ids=t_a, output_hidden_states=True)
    h_manual = out.hidden_states[1][:, qpos, :]

    assert torch.allclose(h_hook, h_manual, atol=1e-5), "hook did not capture layer-0's actual output"
    assert torch.allclose(logits_hook, out.logits, atol=1e-4), "hook changed the forward pass output"


def test_patch_with_own_value_is_a_noop():
    model, w_g, w_f = _tiny_model_and_task()
    ids_a, ids_b, qpos, xa, xb = _build_pair(w_g, w_f)
    t_b = torch.tensor(ids_b, dtype=torch.long).unsqueeze(0)

    logits_clean, h_b = capture_hidden(model, t_b, layer_idx=0, pos=qpos, device="cpu")
    logits_noop_patch = patched_forward(model, t_b, layer_idx=0, pos=qpos, patch_vec=h_b, device="cpu")

    assert torch.allclose(logits_clean, logits_noop_patch, atol=1e-5), (
        "Patching a position with its OWN captured value must reproduce the "
        "unpatched forward exactly -- the hook is grabbing/injecting the "
        "wrong layer, position, or tensor."
    )


def test_patch_with_different_value_actually_changes_output():
    model, w_g, w_f = _tiny_model_and_task()
    ids_a, ids_b, qpos, xa, xb = _build_pair(w_g, w_f)
    t_a = torch.tensor(ids_a, dtype=torch.long).unsqueeze(0)
    t_b = torch.tensor(ids_b, dtype=torch.long).unsqueeze(0)

    logits_clean_b, h_b = capture_hidden(model, t_b, layer_idx=0, pos=qpos, device="cpu")
    _, h_a = capture_hidden(model, t_a, layer_idx=0, pos=qpos, device="cpu")
    logits_patched = patched_forward(model, t_b, layer_idx=0, pos=qpos, patch_vec=h_a, device="cpu")

    assert not torch.allclose(h_a, h_b, atol=1e-6), "test setup degenerate: h_a == h_b already"
    assert not torch.allclose(logits_clean_b, logits_patched, atol=1e-6), (
        "Patching with a genuinely different vector (h_a) produced output "
        "identical to the unpatched run -- the hook may not be firing at all."
    )


def test_patch_does_not_leak_backward_in_time():
    """Causal attention: positions BEFORE qpos can't see qpos's patched
    value, so logits there must be untouched by the patch."""
    model, w_g, w_f = _tiny_model_and_task()
    ids_a, ids_b, qpos, xa, xb = _build_pair(w_g, w_f, num_context_pairs=8)
    t_a = torch.tensor(ids_a, dtype=torch.long).unsqueeze(0)
    t_b = torch.tensor(ids_b, dtype=torch.long).unsqueeze(0)

    logits_clean_b, _ = capture_hidden(model, t_b, layer_idx=0, pos=qpos, device="cpu")
    _, h_a = capture_hidden(model, t_a, layer_idx=0, pos=qpos, device="cpu")
    logits_patched = patched_forward(model, t_b, layer_idx=0, pos=qpos, patch_vec=h_a, device="cpu")

    assert torch.allclose(
        logits_clean_b[:, :qpos, :], logits_patched[:, :qpos, :], atol=1e-5
    ), "patch affected logits at positions before qpos -- causality violated"


def test_decode_z_lm_head_returns_valid_index():
    model, w_g, w_f = _tiny_model_and_task()
    ids_a, ids_b, qpos, xa, xb = _build_pair(w_g, w_f)
    t_a = torch.tensor(ids_a, dtype=torch.long).unsqueeze(0)
    _, h_a = capture_hidden(model, t_a, layer_idx=0, pos=qpos, device="cpu")
    z_hat = decode_z_lm_head(model, h_a)
    assert 0 <= z_hat < Z_SIZE


# --- cross-task trial construction -----------------------------------------


def test_cross_task_trial_is_distinct_and_discriminating():
    w_g, w_f = sample_tasks(8, seed=1)
    rng = np.random.default_rng(0)
    trial = sample_cross_task_trial(w_g, w_f, rng)
    assert trial is not None
    assert trial["d_src"] != trial["d_dst"]

    labels = {
        int(trial["routed"].argmax()),
        int(trial["transplant"].argmax()),
        int(trial["unaffected"].argmax()),
    }
    assert len(labels) == 3, "routed/transplant/unaffected must be pairwise argmax-distinct"

    for dist in (trial["routed"], trial["transplant"], trial["unaffected"]):
        assert np.all(dist >= 0.0)
        assert np.isclose(dist.sum(), 1.0)


def test_cross_task_trial_raises_below_two_tasks():
    w_g, w_f = sample_tasks(1, seed=0)
    rng = np.random.default_rng(0)
    try:
        sample_cross_task_trial(w_g, w_f, rng)
        assert False, "expected ValueError with D<2"
    except ValueError:
        pass


def test_route_margin_sign():
    win = {"kl_route": 0.1, "kl_transplant": 0.5, "kl_unaffected": 0.4}
    lose = {"kl_route": 0.5, "kl_transplant": 0.1, "kl_unaffected": 0.4}
    assert route_margin(win) > 0
    assert route_margin(lose) < 0


def test_select_best_patch_layer_returns_valid_layer_and_per_layer_results():
    model, _, _ = _tiny_model_and_task(n_layers=3)
    w_g, w_f = sample_tasks(4, seed=5)
    best_layer, per_layer = select_best_patch_layer(
        model, w_g, w_f, device="cpu", n_layers=3, val_seed=42, n_val_triples=5
    )
    assert best_layer in (0, 1, 2)
    assert set(per_layer.keys()) == {0, 1, 2}
    for layer, result in per_layer.items():
        assert result["layer"] == layer
        assert result["n_triples"] == 5
    # best_layer must actually be the argmax of route_margin over candidates
    assert best_layer == max(per_layer, key=lambda l: route_margin(per_layer[l]))


def test_run_patching_probe_end_to_end_schema():
    model, _, _ = _tiny_model_and_task()
    w_g, w_f = sample_tasks(4, seed=2)
    out = run_patching_probe(model, w_g, w_f, device="cpu", n_triples=5, layer=0, seed=1)

    for key in (
        "kl_route",
        "kl_transplant",
        "kl_unaffected",
        "kl_clean_route",
        "kl_clean_transplant",
        "kl_clean_unaffected",
        "z_decode_acc",
        "patch_margin_ok",
        "z_decode_ok",
        "trial_accept_rate",
    ):
        assert key in out
    assert out["n_triples"] == 5
    assert 0.0 <= out["trial_accept_rate"] <= 1.0
    assert 0.0 <= out["z_decode_acc"] <= 1.0
    assert out["patch_margin_ok"] in (0.0, 1.0)


if __name__ == "__main__":
    test_ids_differ_only_at_query_position()
    test_capture_hidden_matches_manual_forward()
    test_patch_with_own_value_is_a_noop()
    test_patch_with_different_value_actually_changes_output()
    test_patch_does_not_leak_backward_in_time()
    test_decode_z_lm_head_returns_valid_index()
    test_cross_task_trial_is_distinct_and_discriminating()
    test_cross_task_trial_raises_below_two_tasks()
    test_route_margin_sign()
    test_select_best_patch_layer_returns_valid_layer_and_per_layer_results()
    test_run_patching_probe_end_to_end_schema()
    print("All probe.py cross-task patching tests passed.")
