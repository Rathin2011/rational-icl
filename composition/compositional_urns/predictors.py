"""Closed-form / approximate Bayesian predictors for compositional urns.

Predictors
----------
- g_predictor / f_predictor: exact Dirichlet–Categorical (Phase-1 gates)
- memorizing_M: discrete posterior over the D training (w_g, w_f) tasks
- atomic_G: misspecified direct Dirichlet on P(y|x) from (x,y) counts
- approx_C_GG: mean-field online soft-count updates (comp prefixes)
- exact_C_GG_predictive: exact sum over z-sequences (L small only)
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

from config import X_SIZE, Y_SIZE, Z_SIZE

PairXY = Tuple[int, int]
PairXZ = Tuple[int, int]
PairZY = Tuple[int, int]


def _row_normalize(a: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    s = a.sum(axis=axis, keepdims=True)
    return a / np.maximum(s, eps)


def sym_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * (np.sum(p * (np.log(p) - np.log(q))) + np.sum(q * (np.log(q) - np.log(p)))))


def soft_rel_weights(dists: dict, eps: float = 1e-12) -> dict:
    """d_Q^rel = d_Q / sum_{Q'} d_{Q'} for symmetrized-KL distances."""
    keys = list(dists.keys())
    vals = np.array([max(float(dists[k]), eps) for k in keys], dtype=np.float64)
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(keys, vals)}


# ---------------------------------------------------------------------------
# Phase-1 factor predictors
# ---------------------------------------------------------------------------
def g_predictive(prefix_xz: Sequence[PairXZ], query_x: int, z_size: int = Z_SIZE) -> np.ndarray:
    """Dirichlet–Categorical predictive over Z given (x,z) prefix."""
    alpha = np.ones((X_SIZE, z_size), dtype=np.float64)
    for x, z in prefix_xz:
        alpha[int(x), int(z)] += 1.0
    row = alpha[int(query_x)]
    return row / row.sum()


def f_predictive(prefix_zy: Sequence[PairZY], query_z: int, y_size: int = Y_SIZE) -> np.ndarray:
    """Dirichlet–Categorical predictive over Y given (z,y) prefix."""
    alpha = np.ones((Z_SIZE, y_size), dtype=np.float64)
    for z, y in prefix_zy:
        alpha[int(z), int(y)] += 1.0
    row = alpha[int(query_z)]
    return row / row.sum()


def mean_ce_to_predictor(
    true_tokens: Sequence[int],
    pred_dists: Sequence[np.ndarray],
    eps: float = 1e-12,
) -> float:
    """Average cross-entropy (nats) of predictor dists vs true token ids."""
    assert len(true_tokens) == len(pred_dists)
    total = 0.0
    for tok, p in zip(true_tokens, pred_dists):
        p = np.clip(p, eps, None)
        p = p / p.sum()
        total -= float(np.log(p[int(tok)]))
    return total / max(len(true_tokens), 1)


# ---------------------------------------------------------------------------
# M: memorizing over train pool
# ---------------------------------------------------------------------------
def task_comp_likelihood(w_g: np.ndarray, w_f: np.ndarray, pairs: Sequence[PairXY]) -> float:
    """ℓ = ∏_i Σ_z w_g[x_i,z] w_f[z,y_i] (in linear space; may underflow)."""
    ll = 0.0
    for x, y in pairs:
        p = float(np.dot(w_g[int(x)], w_f[:, int(y)]))
        if p <= 0.0:
            return 0.0
        ll += np.log(p)
    return float(np.exp(ll))


def task_comp_loglik(w_g: np.ndarray, w_f: np.ndarray, pairs: Sequence[PairXY]) -> float:
    s = 0.0
    for x, y in pairs:
        p = float(np.dot(w_g[int(x)], w_f[:, int(y)]))
        if p <= 0.0:
            return -np.inf
        s += np.log(p)
    return float(s)


def memorizing_M_predictive(
    prefix_xy: Sequence[PairXY],
    query_x: int,
    train_w_g: np.ndarray,
    train_w_f: np.ndarray,
) -> np.ndarray:
    """Posterior mixture over D train tasks; predictive P(y|x)."""
    D = train_w_g.shape[0]
    log_posts = np.empty(D, dtype=np.float64)
    for d in range(D):
        log_posts[d] = task_comp_loglik(train_w_g[d], train_w_f[d], prefix_xy)
    # uniform prior
    log_posts -= log_posts.max()
    posts = np.exp(log_posts)
    posts /= posts.sum()
    pred = np.zeros(Y_SIZE, dtype=np.float64)
    qx = int(query_x)
    for d in range(D):
        pred += posts[d] * (train_w_g[d, qx] @ train_w_f[d])
    return pred / pred.sum()


def composite_marginal(w_g: np.ndarray, w_f: np.ndarray, x: int) -> np.ndarray:
    """Σ_z w_g[x,z] w_f[z,:] as a length-|Y| distribution."""
    p = w_g[int(x)] @ w_f
    return p / p.sum()


# ---------------------------------------------------------------------------
# Atomic G (direct Dirichlet on P(y|x))
# ---------------------------------------------------------------------------
def atomic_G_predictive(
    prefix_xy: Sequence[PairXY],
    query_x: int,
    y_size: int = Y_SIZE,
    x_size: int = X_SIZE,
) -> np.ndarray:
    alpha = np.ones((x_size, y_size), dtype=np.float64)
    for x, y in prefix_xy:
        alpha[int(x), int(y)] += 1.0
    row = alpha[int(query_x)]
    return row / row.sum()


# ---------------------------------------------------------------------------
# Approximate C_GG (online mean-field soft counts)
# ---------------------------------------------------------------------------
def approx_C_GG_state_from_prefix(
    prefix_xy: Sequence[PairXY],
    x_size: int = X_SIZE,
    z_size: int = Z_SIZE,
    y_size: int = Y_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (alpha_g, alpha_f) after online soft updates on the prefix."""
    alpha_g = np.ones((x_size, z_size), dtype=np.float64)
    alpha_f = np.ones((z_size, y_size), dtype=np.float64)
    for x, y in prefix_xy:
        x = int(x)
        y = int(y)
        p_g = alpha_g[x] / alpha_g[x].sum()
        p_f_y = alpha_f[:, y] / alpha_f.sum(axis=1)  # P_f(y|z) for each z
        r = p_g * p_f_y
        r_sum = r.sum()
        if r_sum <= 0:
            r = np.ones(z_size) / z_size
        else:
            r = r / r_sum
        alpha_g[x] += r
        alpha_f[:, y] += r
    return alpha_g, alpha_f


def approx_C_GG_predictive(
    prefix_xy: Sequence[PairXY],
    query_x: int,
    x_size: int = X_SIZE,
    z_size: int = Z_SIZE,
    y_size: int = Y_SIZE,
) -> np.ndarray:
    alpha_g, alpha_f = approx_C_GG_state_from_prefix(prefix_xy, x_size, z_size, y_size)
    w_g = _row_normalize(alpha_g)
    w_f = _row_normalize(alpha_f)
    p = w_g[int(query_x)] @ w_f
    return p / p.sum()


# ---------------------------------------------------------------------------
# Exact C_GG (short prefixes): Dirichlet–Multinomial sum over z sequences
# ---------------------------------------------------------------------------
def _dirichlet_multinomial_log_marginal(
    counts_by_row: np.ndarray, prior: float = 1.0
) -> float:
    """counts_by_row: (n_rows, n_cats). Returns sum_rows log DM marginal."""
    # for each row: log[ Γ(K*α) / Γ(K*α + n) * ∏_k Γ(α + n_k) / Γ(α) ]
    n_rows, n_cats = counts_by_row.shape
    total = 0.0
    log_gamma_prior = gammaln(prior)
    for r in range(n_rows):
        n = counts_by_row[r]
        n_tot = n.sum()
        if n_tot == 0:
            continue
        total += gammaln(n_cats * prior) - gammaln(n_cats * prior + n_tot)
        total += float(np.sum(gammaln(prior + n) - log_gamma_prior))
    return float(total)


def exact_C_GG_prefix_log_marginal(
    xs: Sequence[int],
    ys: Sequence[int],
    zs: Sequence[int],
    x_size: int = X_SIZE,
    z_size: int = Z_SIZE,
    y_size: int = Y_SIZE,
) -> float:
    """Log DM marginal for one concrete z-assignment."""
    n_xz = np.zeros((x_size, z_size), dtype=np.float64)
    m_zy = np.zeros((z_size, y_size), dtype=np.float64)
    for x, y, z in zip(xs, ys, zs):
        n_xz[int(x), int(z)] += 1.0
        m_zy[int(z), int(y)] += 1.0
    return _dirichlet_multinomial_log_marginal(n_xz) + _dirichlet_multinomial_log_marginal(
        m_zy
    )


def exact_C_GG_log_evidence(
    pairs: Sequence[PairXY],
    x_size: int = X_SIZE,
    z_size: int = Z_SIZE,
    y_size: int = Y_SIZE,
) -> float:
    """log Σ_{z_{1:n}} P(y_{1:n}, z_{1:n} | x_{1:n}) under factored Dir priors."""
    n = len(pairs)
    if n == 0:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    log_terms = []
    for zs in itertools.product(range(z_size), repeat=n):
        log_terms.append(
            exact_C_GG_prefix_log_marginal(xs, ys, zs, x_size, z_size, y_size)
        )
    m = max(log_terms)
    return float(m + np.log(np.sum(np.exp(np.asarray(log_terms) - m))))


def exact_C_GG_predictive(
    prefix_xy: Sequence[PairXY],
    query_x: int,
    y_size: int = Y_SIZE,
    x_size: int = X_SIZE,
    z_size: int = Z_SIZE,
) -> np.ndarray:
    """Exact predictive P(y|x) by evidence ratios (L must be small)."""
    n = len(prefix_xy)
    if n > 5:
        raise ValueError(f"exact_C_GG_predictive only for short prefixes; got L={n}")
    log_ev_pref = exact_C_GG_log_evidence(prefix_xy, x_size, z_size, y_size)
    log_p = np.empty(y_size, dtype=np.float64)
    for y in range(y_size):
        ext = list(prefix_xy) + [(int(query_x), y)]
        log_ev = exact_C_GG_log_evidence(ext, x_size, z_size, y_size)
        log_p[y] = log_ev - log_ev_pref
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()
