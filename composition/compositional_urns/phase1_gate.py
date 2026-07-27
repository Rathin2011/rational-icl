"""Phase-1 Bayesian CE gate helpers.

Gate (both must hold on ID, output positions only):
  CE_model(g_only) - CE_gBayes <= tol
  CE_model(f_only) - CE_fBayes <= tol
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    X_OFFSET,
    Z_OFFSET,
    Y_OFFSET,
    X_SIZE,
    Z_SIZE,
    Y_SIZE,
    PHASE1_CE_TOL,
    NUM_PAIRS,
)
from predictors import g_predictive, f_predictive, mean_ce_to_predictor


def _output_positions(num_pairs: int = NUM_PAIRS):
    return [2 * i + 1 for i in range(num_pairs)]


def bayes_ce_g_only_batch(examples, concentration: float = 1.0) -> float:
    """Average CE of g-predictor vs true z tokens on g_only examples.

    concentration must match data.sample_tasks's actual g-concentration for
    this to remain the true Bayes floor rather than a misspecified one.
    """
    toks = []
    dists = []
    for ex in examples:
        pairs = ex["pairs"]  # (x,z)
        # For each output position i, predictive uses prefix pairs[:i]
        for i, (x, z) in enumerate(pairs):
            p = g_predictive(pairs[:i], query_x=x, concentration=concentration)
            toks.append(z)
            dists.append(p)
    return mean_ce_to_predictor(toks, dists)


def bayes_ce_f_only_batch(examples, concentration: float = 1.0) -> float:
    toks = []
    dists = []
    for ex in examples:
        pairs = ex["pairs"]  # (z,y)
        for i, (z, y) in enumerate(pairs):
            p = f_predictive(pairs[:i], query_z=z, concentration=concentration)
            toks.append(y)
            dists.append(p)
    return mean_ce_to_predictor(toks, dists)


@torch.no_grad()
def model_ce_on_eval(model, dataset, device: str) -> float:
    """Token CE on labeled (output) positions only."""
    model.eval()
    total_nll = 0.0
    n_tok = 0
    for ex in dataset:
        ids = ex["input_ids"].unsqueeze(0).to(device)
        labels = ex["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits
        # shift for causal LM: logits[:, t] predicts labels[:, t+1]? 
        # HF convention: logits[i] predicts token i+1 when labels aligned to input.
        # Our labels are on the same indices as output tokens; standard HF:
        # loss uses shift_logits = logits[..., :-1], shift_labels = labels[..., 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n = int((shift_labels != -100).sum().item())
        total_nll += float(loss.item())
        n_tok += n
    return total_nll / max(n_tok, 1)


def check_phase1_gate(
    model_ce_g: float,
    bayes_ce_g: float,
    model_ce_f: float,
    bayes_ce_f: float,
    tol: float = PHASE1_CE_TOL,
) -> Tuple[bool, Dict[str, float]]:
    gap_g = model_ce_g - bayes_ce_g
    gap_f = model_ce_f - bayes_ce_f
    ok = (gap_g <= tol) and (gap_f <= tol)
    return ok, {
        "model_ce_g": model_ce_g,
        "bayes_ce_g": bayes_ce_g,
        "gap_g": gap_g,
        "model_ce_f": model_ce_f,
        "bayes_ce_f": bayes_ce_f,
        "gap_f": gap_f,
        "tol": tol,
        "passed": float(ok),
    }


def evaluate_phase1_gate(
    model,
    g_eval,
    f_eval,
    device: str,
    tol: float = PHASE1_CE_TOL,
    g_concentration: float = 1.0,
    f_concentration: float = 1.0,
):
    """Full gate evaluation on fixed CompUrnsEvalDataset objects."""
    bayes_g = bayes_ce_g_only_batch(g_eval, concentration=g_concentration)
    bayes_f = bayes_ce_f_only_batch(f_eval, concentration=f_concentration)
    model_g = model_ce_on_eval(model, g_eval, device)
    model_f = model_ce_on_eval(model, f_eval, device)
    return check_phase1_gate(model_g, bayes_g, model_f, bayes_f, tol=tol)
