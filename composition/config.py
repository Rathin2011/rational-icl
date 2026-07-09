"""Configuration and vocabulary layout for the compositional lookup-table ICL task.

A task is a pair (g, f) of deterministic lookup tables:
    g: X -> Z   (|X| = 50, |Z| = 5)
    f: Z -> Y   (|Z| = 5,  |Y| = 50)
with composite h(x) = f(g(x)).

The vocabulary is the disjoint union X | Z | Y (105 tokens). Token ids are laid
out contiguously so that a token's identity reveals which alphabet it belongs to
(this is how the model infers the sequence type without an explicit type token):

    X : ids 0..49
    Z : ids 50..54
    Y : ids 55..104
"""

import os
import json
from dataclasses import dataclass, asdict

# --- Vocabulary layout (disjoint alphabets) --------------------------------
X_SIZE = 50
Z_SIZE = 5
Y_SIZE = 50

X_OFFSET = 0
Z_OFFSET = X_SIZE            # 50
Y_OFFSET = X_SIZE + Z_SIZE   # 55
VOCAB_SIZE = X_SIZE + Z_SIZE + Y_SIZE  # 105

# --- Sequence layout --------------------------------------------------------
NUM_PAIRS = 12
SEQ_LEN = 2 * NUM_PAIRS       # 24 tokens: interleaved (input, output) pairs

# Sequence types, sampled uniformly per training example.
SEQ_TYPES = ("comp", "g_only", "f_only")
SEQ_TYPE_TO_ID = {t: i for i, t in enumerate(SEQ_TYPES)}


def read_cache_dir(default=None):
    """Resolve the cache directory from $CACHE_DIR or the project-root .env."""
    cd = os.getenv("CACHE_DIR")
    if cd:
        return cd
    root_env = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(root_env):
        with open(root_env) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("CACHE_DIR="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


@dataclass
class CompositionConfig:
    # Task diversity (D): number of fixed (g, f) tasks sampled once from the prior.
    num_tasks: int = 64
    seed: int = 1

    # Model (GPT-NeoX backbone)
    d_model: int = 128           # hidden_size
    d_ff: int = 512              # intermediate_size
    n_layers: int = 2
    n_heads: int = 4
    max_position_embeddings: int = 128

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_steps: int = 100_000
    warmup_steps: int = 0
    lr_scheduler_type: str = "constant"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    eval_steps: int = 1000
    num_checkpoints: int = 50    # sqrt-spaced checkpoints saved during training

    # Evaluation
    num_ood_tasks: int = 256     # held-out tasks for the generalization eval
    n_eval: int = 512            # sequences per eval set

    # IO
    cache_dir: str = None

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = read_cache_dir()
        if self.cache_dir is None:
            raise ValueError(
                "cache_dir is not set; pass --cache_dir or set CACHE_DIR in .env"
            )

    # --- Derived paths ------------------------------------------------------
    @property
    def setting_dir(self):
        return os.path.join(self.cache_dir, "composition")

    @property
    def tasks_dir(self):
        return os.path.join(self.setting_dir, "tasks")

    @property
    def train_tasks_path(self):
        return os.path.join(self.tasks_dir, f"train-D{self.num_tasks}-seed{self.seed}.npz")

    @property
    def ood_tasks_path(self):
        # OOD tasks are drawn from the same prior with a disjoint seed.
        return os.path.join(
            self.tasks_dir, f"ood-D{self.num_ood_tasks}-seed{self.seed + 1000}.npz"
        )

    @property
    def run_name(self):
        return (
            f"D{self.num_tasks}-{self.n_layers}L-{self.n_heads}H-{self.d_model}d"
            f"-{self.d_ff}ff-lr{self.learning_rate}-bs{self.batch_size}"
            f"-{self.max_steps}steps-seed{self.seed}"
        )

    @property
    def run_dir(self):
        return os.path.join(self.setting_dir, "runs", self.run_name)

    @property
    def checkpoints_dir(self):
        return os.path.join(self.run_dir, "checkpoints")

    def save(self):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), "w") as fh:
            json.dump(asdict(self), fh, indent=2)
