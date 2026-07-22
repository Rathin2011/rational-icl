"""Configuration and vocabulary layout for the compositional lookup-table ICL task.

A task is a pair (g, f) of deterministic lookup tables:
    g: X -> Z   (|X| = 50, |Z| = 5)
    f: Z -> Y   (|Z| = 5, |Y| = 50)
with composite h(x) = f(g(x)).

Current default |Z|=5 is the ΔK > 0 (bottleneck / hoped C_GG) setting from the notes.
Earlier shortcut runs used |Z|=45 (ΔK < 0); task/run paths include alphabet sizes
so those caches do not collide.

The vocabulary is the disjoint union X | Z | Y. Token ids are laid out
contiguously so that a token's identity reveals which alphabet it belongs to
(this is how the model infers the sequence type without an explicit type token).
"""

import os
import json

# --- Vocabulary layout (disjoint alphabets) --------------------------------
X_SIZE = 50
Z_SIZE = 5
Y_SIZE = 50

X_OFFSET = 0
Z_OFFSET = X_SIZE
Y_OFFSET = X_SIZE + Z_SIZE
VOCAB_SIZE = X_SIZE + Z_SIZE + Y_SIZE

# --- Sequence layout --------------------------------------------------------
# Main linear-regression exp uses context_length=16; match that # of demos.
NUM_PAIRS = 16
SEQ_LEN = 2 * NUM_PAIRS       # 32 tokens: interleaved (input, output) pairs

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


class CompositionConfig:
    """Run configuration (plain class for compatibility with Python 3.6+)."""

    def __init__(
        self,
        num_tasks=64,
        seed=1,
        # Match linear-regression main exp in this repo (Wurgaft-style):
        # 8 layers, 1 head, d_model=64, mlp expansion 4 -> d_ff=256.
        d_model=64,
        d_ff=256,
        n_layers=8,
        n_heads=1,
        max_position_embeddings=128,
        batch_size=128,
        learning_rate=5e-4,
        max_steps=100_000,
        warmup_steps=500,
        lr_scheduler_type="inverse_sqrt",
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=100,
        eval_steps=1000,
        num_checkpoints=50,
        num_ood_tasks=256,
        n_eval=512,
        cache_dir=None,
    ):
        self.num_tasks = num_tasks
        self.seed = seed
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_position_embeddings = max_position_embeddings
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.num_checkpoints = num_checkpoints
        self.num_ood_tasks = num_ood_tasks
        self.n_eval = n_eval
        self.cache_dir = cache_dir if cache_dir is not None else read_cache_dir()
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
    def alphabet_tag(self):
        return f"X{X_SIZE}-Z{Z_SIZE}-Y{Y_SIZE}"

    @property
    def train_tasks_path(self):
        return os.path.join(
            self.tasks_dir,
            f"train-D{self.num_tasks}-{self.alphabet_tag}-seed{self.seed}.npz",
        )

    @property
    def ood_tasks_path(self):
        # OOD tasks are drawn from the same prior with a disjoint seed.
        return os.path.join(
            self.tasks_dir,
            f"ood-D{self.num_ood_tasks}-{self.alphabet_tag}-seed{self.seed + 1000}.npz",
        )

    @property
    def run_name(self):
        return (
            f"D{self.num_tasks}-{self.alphabet_tag}-"
            f"{self.n_layers}L-{self.n_heads}H-{self.d_model}d"
            f"-{self.d_ff}ff-lr{self.learning_rate}-bs{self.batch_size}"
            f"-{self.max_steps}steps-seed{self.seed}"
        )

    @property
    def run_dir(self):
        return os.path.join(self.setting_dir, "runs", self.run_name)

    @property
    def checkpoints_dir(self):
        return os.path.join(self.run_dir, "checkpoints")

    def to_dict(self):
        return {
            "num_tasks": self.num_tasks,
            "seed": self.seed,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "num_checkpoints": self.num_checkpoints,
            "num_ood_tasks": self.num_ood_tasks,
            "n_eval": self.n_eval,
            "cache_dir": self.cache_dir,
        }

    def save(self):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
