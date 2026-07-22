"""Unit tests: approx C_GG vs exact on short prefixes; Phase-1 predictors smoke."""

from __future__ import annotations

import sys
import os

import numpy as np

# Allow `python test_predictors.py` from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictors import (
    approx_C_GG_predictive,
    exact_C_GG_predictive,
    g_predictive,
    f_predictive,
    atomic_G_predictive,
    memorizing_M_predictive,
    sym_kl,
    soft_rel_weights,
)
from data import sample_tasks


def test_g_f_predictive_uniform_prior():
    p = g_predictive([], query_x=0)
    assert p.shape == (6,)
    assert np.allclose(p, np.ones(6) / 6)
    q = f_predictive([], query_z=0)
    assert q.shape == (20,)
    assert np.allclose(q, np.ones(20) / 20)


def test_g_predictive_updates():
    # one observation (0, 3) should bump z=3
    p = g_predictive([(0, 3)], query_x=0)
    assert p.argmax() == 3
    assert p[3] > p[0]


def test_approx_vs_exact_short_prefix():
    rng = np.random.default_rng(0)
    # L=2, |Z|=6 → 36 assignments; fast
    pairs = [(int(rng.integers(20)), int(rng.integers(20))) for _ in range(2)]
    query_x = int(rng.integers(20))
    p_approx = approx_C_GG_predictive(pairs, query_x)
    p_exact = exact_C_GG_predictive(pairs, query_x)
    assert p_approx.shape == (20,)
    assert p_exact.shape == (20,)
    assert np.all(p_approx >= 0) and np.isclose(p_approx.sum(), 1.0)
    assert np.all(p_exact >= 0) and np.isclose(p_exact.sum(), 1.0)
    # Soft agreement: total variation not huge on short empty-ish prefixes
    tv = 0.5 * np.abs(p_approx - p_exact).sum()
    # Online MF is approximate; allow moderate TV but both must be valid dists
    assert tv < 0.5, f"TV too large: {tv}"


def test_approx_vs_exact_l3():
    rng = np.random.default_rng(1)
    pairs = [(int(rng.integers(20)), int(rng.integers(20))) for _ in range(3)]
    query_x = 5
    p_approx = approx_C_GG_predictive(pairs, query_x)
    p_exact = exact_C_GG_predictive(pairs, query_x)
    tv = 0.5 * np.abs(p_approx - p_exact).sum()
    # Documented approximation: TV should be finite / sane
    assert tv < 0.75, f"TV too large at L=3: {tv}"
    # Sym-KL finite
    d = sym_kl(p_approx, p_exact)
    assert np.isfinite(d)


def test_memorizing_M_shapes():
    w_g, w_f = sample_tasks(8, seed=2)
    pairs = [(0, 1), (2, 3)]
    p = memorizing_M_predictive(pairs, query_x=0, train_w_g=w_g, train_w_f=w_f)
    assert p.shape == (20,)
    assert np.isclose(p.sum(), 1.0)


def test_atomic_G():
    p0 = atomic_G_predictive([], 0)
    assert np.allclose(p0, np.ones(20) / 20)
    p1 = atomic_G_predictive([(0, 7), (0, 7)], 0)
    assert p1.argmax() == 7


def test_soft_rel_weights():
    w = soft_rel_weights({"M": 0.2, "G": 0.3, "C_GG": 0.5})
    assert np.isclose(sum(w.values()), 1.0)
    assert w["C_GG"] == max(w.values())


if __name__ == "__main__":
    test_g_f_predictive_uniform_prior()
    test_g_predictive_updates()
    test_approx_vs_exact_short_prefix()
    test_approx_vs_exact_l3()
    test_memorizing_M_shapes()
    test_atomic_G()
    test_soft_rel_weights()
    print("All predictor tests passed.")
