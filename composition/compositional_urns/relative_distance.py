"""3-way soft relative distance among M, atomic-G, approx-C_GG.

Uses symmetrized KL on Y-slice predictives (PROTOCOL.md).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from predictors import (
    approx_C_GG_predictive,
    atomic_G_predictive,
    memorizing_M_predictive,
    soft_rel_weights,
    sym_kl,
)

PairXY = Tuple[int, int]


def predictor_dists_at_queries(
    prefix_xy: Sequence[PairXY],
    query_xs: Sequence[int],
    train_w_g: np.ndarray,
    train_w_f: np.ndarray,
) -> Dict[str, List[np.ndarray]]:
    out = {"M": [], "G": [], "C_GG": []}
    for qx in query_xs:
        out["M"].append(
            memorizing_M_predictive(prefix_xy, qx, train_w_g, train_w_f)
        )
        out["G"].append(atomic_G_predictive(prefix_xy, qx))
        out["C_GG"].append(approx_C_GG_predictive(prefix_xy, qx))
    return out


def soft_d_rel_for_model_dists(
    model_dists: Sequence[np.ndarray],
    prefix_xy: Sequence[PairXY],
    query_xs: Sequence[int],
    train_w_g: np.ndarray,
    train_w_f: np.ndarray,
) -> Dict[str, float]:
    """Average soft d_rel over query positions for one sequence."""
    refs = predictor_dists_at_queries(prefix_xy, query_xs, train_w_g, train_w_f)
    acc = {"M": 0.0, "G": 0.0, "C_GG": 0.0}
    n = 0
    for i, h in enumerate(model_dists):
        d = {
            "M": sym_kl(h, refs["M"][i]),
            "G": sym_kl(h, refs["G"][i]),
            "C_GG": sym_kl(h, refs["C_GG"][i]),
        }
        w = soft_rel_weights(d)
        for k in acc:
            acc[k] += w[k]
        n += 1
    return {k: acc[k] / max(n, 1) for k in acc}


def cgg_wins(d_rel: Dict[str, float], margin: float = 0.05) -> bool:
    """C_GG strictly smallest and ≥ margin below next-best."""
    c = d_rel["C_GG"]
    others = [d_rel["M"], d_rel["G"]]
    return c + margin <= min(others)
