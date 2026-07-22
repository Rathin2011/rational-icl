"""Task sampling and sequence generation for stochastic compositional urns."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from config import (
    X_SIZE,
    Z_SIZE,
    Y_SIZE,
    X_OFFSET,
    Z_OFFSET,
    Y_OFFSET,
    NUM_PAIRS,
    SEQ_LEN,
    SEQ_TYPES,
    SEQ_TYPE_TO_ID,
)


def sample_tasks(num_tasks: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample row-stochastic (w_g, w_f) from independent Dirichlet row priors.

    Returns
    -------
    w_g : (D, X, Z)
    w_f : (D, Z, Y)
    """
    rng = np.random.default_rng(seed)
    w_g = np.empty((num_tasks, X_SIZE, Z_SIZE), dtype=np.float64)
    w_f = np.empty((num_tasks, Z_SIZE, Y_SIZE), dtype=np.float64)
    for d in range(num_tasks):
        for x in range(X_SIZE):
            w_g[d, x] = rng.dirichlet(np.ones(Z_SIZE))
        for z in range(Z_SIZE):
            w_f[d, z] = rng.dirichlet(np.ones(Y_SIZE))
    return w_g, w_f


def save_tasks(path: str, w_g: np.ndarray, w_f: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, w_g=w_g, w_f=w_f)


def load_tasks(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["w_g"], data["w_f"]


def get_or_create_tasks(path: str, num_tasks: int, seed: int, regenerate: bool = False):
    if os.path.exists(path) and not regenerate:
        return load_tasks(path)
    w_g, w_f = sample_tasks(num_tasks, seed)
    save_tasks(path, w_g, w_f)
    return w_g, w_f


def build_sequence(
    w_g: np.ndarray,
    w_f: np.ndarray,
    rng: np.random.Generator,
    seq_type: str,
    num_pairs: int = NUM_PAIRS,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Build one interleaved sequence; labels=-100 on inputs.

    Returns input_ids, labels, and a list of raw pairs for predictors:
      comp / atomic: (x, y) local indices
      g_only: (x, z)
      f_only: (z, y)
    """
    input_ids = np.empty(2 * num_pairs, dtype=np.int64)
    labels = np.full(2 * num_pairs, -100, dtype=np.int64)
    pairs = []

    for i in range(num_pairs):
        if seq_type == "comp":
            x = int(rng.integers(X_SIZE))
            z = int(rng.choice(Z_SIZE, p=w_g[x]))
            y = int(rng.choice(Y_SIZE, p=w_f[z]))
            in_id = X_OFFSET + x
            out_id = Y_OFFSET + y
            pairs.append((x, y))
        elif seq_type == "g_only":
            x = int(rng.integers(X_SIZE))
            z = int(rng.choice(Z_SIZE, p=w_g[x]))
            in_id = X_OFFSET + x
            out_id = Z_OFFSET + z
            pairs.append((x, z))
        elif seq_type == "f_only":
            z = int(rng.integers(Z_SIZE))
            y = int(rng.choice(Y_SIZE, p=w_f[z]))
            in_id = Z_OFFSET + z
            out_id = Y_OFFSET + y
            pairs.append((z, y))
        else:
            raise ValueError(seq_type)

        input_ids[2 * i] = in_id
        input_ids[2 * i + 1] = out_id
        labels[2 * i + 1] = out_id

    return input_ids, labels, pairs


class CompUrnsIterable(IterableDataset):
    """Infinite training stream with typed mixing."""

    def __init__(self, w_g, w_f, p_mix, seed=1):
        self.w_g = w_g
        self.w_f = w_f
        self.p_mix = np.asarray(p_mix, dtype=np.float64)
        self.p_mix = self.p_mix / self.p_mix.sum()
        self.seed = int(seed)
        self.D = w_g.shape[0]

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + 17 * wid)
        while True:
            d = int(rng.integers(self.D))
            tau = SEQ_TYPES[int(rng.choice(len(SEQ_TYPES), p=self.p_mix))]
            ids, labels, _ = build_sequence(self.w_g[d], self.w_f[d], rng, tau)
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "seq_type": SEQ_TYPE_TO_ID[tau],
            }


class CompUrnsEvalDataset(Dataset):
    """Fixed eval set of typed sequences."""

    def __init__(self, w_g, w_f, seq_type, n_sequences=256, seed=1):
        self.examples = []
        rng = np.random.default_rng(seed)
        D = w_g.shape[0]
        for i in range(n_sequences):
            d = int(rng.integers(D))
            ids, labels, pairs = build_sequence(w_g[d], w_f[d], rng, seq_type)
            self.examples.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "seq_type": SEQ_TYPE_TO_ID[seq_type],
                    "task_idx": d,
                    "pairs": pairs,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_shared_prefix_comp_sequence(
    w_g: np.ndarray,
    w_f: np.ndarray,
    rng: np.random.Generator,
    query_x: int,
    num_context_pairs: int = NUM_PAIRS - 1,
) -> Tuple[np.ndarray, list, int]:
    """Context pairs + final query x (y slot filled with placeholder 0).

    Returns input_ids of length 2*(num_context_pairs+1), context (x,y) pairs,
    and the query position index (even index of final x).
    """
    ids = np.empty(2 * (num_context_pairs + 1), dtype=np.int64)
    pairs = []
    for i in range(num_context_pairs):
        x = int(rng.integers(X_SIZE))
        z = int(rng.choice(Z_SIZE, p=w_g[x]))
        y = int(rng.choice(Y_SIZE, p=w_f[z]))
        ids[2 * i] = X_OFFSET + x
        ids[2 * i + 1] = Y_OFFSET + y
        pairs.append((x, y))
    qpos = 2 * num_context_pairs
    ids[qpos] = X_OFFSET + int(query_x)
    ids[qpos + 1] = Y_OFFSET  # placeholder; model predicts this slot from qpos
    return ids, pairs, qpos
