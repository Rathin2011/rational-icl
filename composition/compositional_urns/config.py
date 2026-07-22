"""Config and vocabulary for stochastic compositional Balls & Urns (C_GG test).

Bottleneck heuristic: |Z|=6 < |X|/2=10. Not claimed as write-up ΔK.
Architecture locked to 2L/4H/d=128 (see PROTOCOL.md).
"""

from __future__ import annotations

import json
import os

# --- Alphabets (disjoint) ---------------------------------------------------
X_SIZE = 20
Z_SIZE = 6
Y_SIZE = 20

X_OFFSET = 0
Z_OFFSET = X_SIZE
Y_OFFSET = X_SIZE + Z_SIZE
VOCAB_SIZE = X_SIZE + Z_SIZE + Y_SIZE  # 46

# --- Sequence layout --------------------------------------------------------
NUM_PAIRS = 32
SEQ_LEN = 2 * NUM_PAIRS  # 64

SEQ_TYPES = ("comp", "g_only", "f_only")
SEQ_TYPE_TO_ID = {t: i for t, i in zip(SEQ_TYPES, range(3))}

# --- Locked training defaults ----------------------------------------------
N_LAYERS = 2
N_HEADS = 4
D_MODEL = 128
D_FF = 512
MAX_POSITION_EMBEDDINGS = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
BATCH_SIZE = 256
SEED = 1

NUM_TRAIN_TASKS = 64
NUM_OOD_TASKS = 200

PHASE1_MAX_STEPS = 100_000
PHASE1_CKPT_EVERY = 5_000
PHASE1_CE_TOL = 0.05

PHASE2_STEPS = 60_000
PHASE2_SAVE_STEPS = (1000, 3000, 10000, 30000, 60000)

# Mix / loss weights
PHASE1_P_MIX = (0.0, 0.5, 0.5)  # comp, g_only, f_only
PHASE2_P_MIX = (0.85, 0.075, 0.075)
PHASE2_LOSS_WEIGHTS = (1.0, 2.0, 2.0)  # comp, g, f

# Measurement locks
PATCH_LAYER = 0
PATCH_LAYER_LOG = 1
D_REL_MARGIN = 0.05
PATCH_KL_MARGIN = 0.05
Z_DECODE_MIN = 0.4
N_PATCH_TRIPLES = 200


def read_cache_dir(default=None):
    cd = os.getenv("CACHE_DIR")
    if cd:
        return cd
    root_env = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
    if os.path.exists(root_env):
        with open(root_env) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("CACHE_DIR="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default or "/projectnb/buinlp/rathin/cache"


class CompUrnsConfig:
    """Run configuration for the two-phase C_GG experiment."""

    def __init__(
        self,
        num_tasks=NUM_TRAIN_TASKS,
        num_ood_tasks=NUM_OOD_TASKS,
        seed=SEED,
        d_model=D_MODEL,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        max_steps=PHASE1_MAX_STEPS,
        p_comp=0.0,
        p_g_only=0.5,
        p_f_only=0.5,
        w_comp=1.0,
        w_g_only=1.0,
        w_f_only=1.0,
        run_tag="",
        cache_dir=None,
    ):
        self.num_tasks = int(num_tasks)
        self.num_ood_tasks = int(num_ood_tasks)
        self.seed = int(seed)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.p_comp = float(p_comp)
        self.p_g_only = float(p_g_only)
        self.p_f_only = float(p_f_only)
        self.w_comp = float(w_comp)
        self.w_g_only = float(w_g_only)
        self.w_f_only = float(w_f_only)
        self.run_tag = str(run_tag)
        self.cache_dir = cache_dir or read_cache_dir()

        base = os.path.join(self.cache_dir, "compositional_urns")
        self.tasks_dir = os.path.join(base, "tasks")
        self.runs_dir = os.path.join(base, "runs")
        tag = f"X{X_SIZE}-Z{Z_SIZE}-Y{Y_SIZE}"
        self.train_tasks_path = os.path.join(
            self.tasks_dir, f"train-D{self.num_tasks}-seed{self.seed}-{tag}.npz"
        )
        self.ood_tasks_path = os.path.join(
            self.tasks_dir,
            f"ood-D{self.num_ood_tasks}-seed{self.seed + 1000}-{tag}.npz",
        )

    @property
    def p_mix(self):
        return (self.p_comp, self.p_g_only, self.p_f_only)

    @property
    def loss_weights(self):
        return (self.w_comp, self.w_g_only, self.w_f_only)

    def run_name(self, phase):
        parts = [
            f"phase{phase}",
            f"D{self.num_tasks}",
            f"X{X_SIZE}Z{Z_SIZE}Y{Y_SIZE}",
            f"{self.n_layers}L{self.n_heads}H{self.d_model}d",
            f"lr{self.learning_rate}",
            f"bs{self.batch_size}",
            f"seed{self.seed}",
        ]
        if self.run_tag:
            parts.append(self.run_tag)
        return "-".join(parts)

    def run_dir(self, phase):
        return os.path.join(self.runs_dir, self.run_name(phase))

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
