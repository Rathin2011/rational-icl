"""Task sampling and datasets for the compositional lookup-table ICL task.

A fixed pool of D tasks {(g, f)} is drawn once from the priors and held constant.
For each training example we sample a task index and a sequence type, then emit an
interleaved sequence of (input, output) pairs. The task itself is never tokenized;
the model must infer (g, f) from the in-context prefix.

Loss is autoregressive next-token cross-entropy scored ONLY on the deterministic
output tokens. Input tokens are randomly drawn and unpredictable, so their label
positions are set to -100 (ignored by the HuggingFace causal-LM loss).
"""

import os
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


# --- Task priors ------------------------------------------------------------
def sample_tasks(num_tasks, seed):
    """Draw `num_tasks` tasks (g, f) independently from the uniform priors.

    g ~ Uniform(Z^X): for each of the |X| inputs, an element of Z.
    f ~ Uniform(Y^Z): for each of the |Z| inputs, an element of Y.

    Returns
    -------
    g : np.ndarray[int64], shape (num_tasks, X_SIZE), values in [0, Z_SIZE)
    f : np.ndarray[int64], shape (num_tasks, Z_SIZE), values in [0, Y_SIZE)
    """
    rng = np.random.default_rng(seed)
    g = rng.integers(0, Z_SIZE, size=(num_tasks, X_SIZE), dtype=np.int64)
    f = rng.integers(0, Y_SIZE, size=(num_tasks, Z_SIZE), dtype=np.int64)
    return g, f


def save_tasks(path, g, f):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, g=g, f=f)


def load_tasks(path):
    data = np.load(path)
    return data["g"], data["f"]


def get_or_create_tasks(path, num_tasks, seed, regenerate=False):
    """Load tasks from `path`, or sample and cache them if missing."""
    if os.path.exists(path) and not regenerate:
        return load_tasks(path)
    g, f = sample_tasks(num_tasks, seed)
    save_tasks(path, g, f)
    return g, f


# --- Sequence construction --------------------------------------------------
def _build_sequence(g_row, f_row, rng, seq_type):
    """Build one interleaved sequence for a single task and sequence type.

    Emits (in_1, out_1, in_2, out_2, ..., in_12, out_12). Output tokens (odd
    positions) are scored; input tokens (even positions) are masked with -100.
    """
    input_ids = np.empty(SEQ_LEN, dtype=np.int64)
    labels = np.full(SEQ_LEN, -100, dtype=np.int64)

    for i in range(NUM_PAIRS):
        if seq_type == "comp":            # x -> f(g(x))
            x = int(rng.integers(X_SIZE))
            z = int(g_row[x])
            y = int(f_row[z])
            in_id = X_OFFSET + x
            out_id = Y_OFFSET + y
        elif seq_type == "g_only":        # x -> g(x)
            x = int(rng.integers(X_SIZE))
            z = int(g_row[x])
            in_id = X_OFFSET + x
            out_id = Z_OFFSET + z
        else:                              # f_only: z -> f(z)
            z = int(rng.integers(Z_SIZE))
            y = int(f_row[z])
            in_id = Z_OFFSET + z
            out_id = Y_OFFSET + y

        input_ids[2 * i] = in_id
        input_ids[2 * i + 1] = out_id
        labels[2 * i + 1] = out_id  # score only the output token

    return input_ids, labels


# --- Datasets ---------------------------------------------------------------
class CompositionTrainDataset(IterableDataset):
    """Infinite stream of training sequences (fresh samples every draw)."""

    def __init__(self, g, f, seed):
        self.g = g
        self.f = f
        self.num_tasks = g.shape[0]
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng([self.seed, worker_id, 0xC0FFEE])
        num_types = len(SEQ_TYPES)
        while True:
            d = int(rng.integers(self.num_tasks))
            seq_type = SEQ_TYPES[int(rng.integers(num_types))]
            input_ids, labels = _build_sequence(self.g[d], self.f[d], rng, seq_type)
            yield {
                "input_ids": torch.from_numpy(input_ids),
                "labels": torch.from_numpy(labels),
            }


class CompositionEvalDataset(Dataset):
    """Fixed, deterministic evaluation set for one task pool and sequence type."""

    def __init__(self, g, f, seq_type, n_eval, seed):
        rng = np.random.default_rng([seed, SEQ_TYPE_TO_ID[seq_type]])
        self.examples = []
        num_tasks = g.shape[0]
        for _ in range(n_eval):
            d = int(rng.integers(num_tasks))
            input_ids, labels = _build_sequence(g[d], f[d], rng, seq_type)
            self.examples.append(
                (torch.from_numpy(input_ids), torch.from_numpy(labels))
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels = self.examples[idx]
        return {"input_ids": input_ids, "labels": labels}


def build_eval_sets(train_g, train_f, ood_g, ood_f, n_eval, seed):
    """Build the 6 eval sets: {in-distribution, out-of-distribution} x 3 types.

    - `id_*` sets reuse the D training tasks (measures memorization / in-dist fit).
    - `ood_*` sets use held-out tasks (measures generalization to new (g, f)).
    """
    eval_sets = {}
    for seq_type in SEQ_TYPES:
        eval_sets[f"id_{seq_type}"] = CompositionEvalDataset(
            train_g, train_f, seq_type, n_eval, seed + 1
        )
        eval_sets[f"ood_{seq_type}"] = CompositionEvalDataset(
            ood_g, ood_f, seq_type, n_eval, seed + 2
        )
    return eval_sets
