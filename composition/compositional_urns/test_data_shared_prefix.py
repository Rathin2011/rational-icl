"""Input/output contract tests for data.build_shared_prefix_comp_sequence.

Scope: only this function. It's the shared building block eval_mg_sweep.py
depends on for every query it scores, and it has no existing coverage
(test_predictors.py only exercises predictors.py).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import X_OFFSET, X_SIZE, Y_OFFSET
from data import build_shared_prefix_comp_sequence, sample_tasks


def test_shape_and_qpos():
    w_g, w_f = sample_tasks(1, seed=0)
    rng = np.random.default_rng(1)
    L = 5
    ids, pairs, qpos = build_shared_prefix_comp_sequence(
        w_g[0], w_f[0], rng, query_x=3, num_context_pairs=L
    )
    assert ids.shape == (2 * (L + 1),)
    assert len(pairs) == L
    assert qpos == 2 * L


def test_query_and_placeholder_tokens():
    w_g, w_f = sample_tasks(1, seed=0)
    rng = np.random.default_rng(1)
    query_x = 7
    ids, pairs, qpos = build_shared_prefix_comp_sequence(
        w_g[0], w_f[0], rng, query_x=query_x, num_context_pairs=4
    )
    assert ids[qpos] == X_OFFSET + query_x
    assert ids[qpos + 1] == Y_OFFSET  # placeholder, model predicts this slot


def test_context_pairs_match_encoded_ids():
    w_g, w_f = sample_tasks(1, seed=0)
    rng = np.random.default_rng(2)
    ids, pairs, qpos = build_shared_prefix_comp_sequence(
        w_g[0], w_f[0], rng, query_x=0, num_context_pairs=6
    )
    for i, (x, y) in enumerate(pairs):
        assert ids[2 * i] == X_OFFSET + x
        assert ids[2 * i + 1] == Y_OFFSET + y
        assert 0 <= x < X_SIZE


def test_zero_context_pairs_edge_case():
    w_g, w_f = sample_tasks(1, seed=0)
    rng = np.random.default_rng(3)
    ids, pairs, qpos = build_shared_prefix_comp_sequence(
        w_g[0], w_f[0], rng, query_x=2, num_context_pairs=0
    )
    assert len(pairs) == 0
    assert qpos == 0
    assert ids.shape == (2,)
    assert ids[0] == X_OFFSET + 2
    assert ids[1] == Y_OFFSET


def test_context_x_is_stochastic_not_fixed():
    # With num_context_pairs > 1, x values in the prefix should vary across
    # draws (rng.integers(X_SIZE) each iteration) -- catches an accidental
    # off-by-one that reuses a single x for the whole prefix.
    w_g, w_f = sample_tasks(1, seed=0)
    rng = np.random.default_rng(4)
    _, pairs, _ = build_shared_prefix_comp_sequence(
        w_g[0], w_f[0], rng, query_x=0, num_context_pairs=20
    )
    xs = [x for x, _ in pairs]
    assert len(set(xs)) > 1


if __name__ == "__main__":
    test_shape_and_qpos()
    test_query_and_placeholder_tokens()
    test_context_pairs_match_encoded_ids()
    test_zero_context_pairs_edge_case()
    test_context_x_is_stochastic_not_fixed()
    print("All build_shared_prefix_comp_sequence tests passed.")
