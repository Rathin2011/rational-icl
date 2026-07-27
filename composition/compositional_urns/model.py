"""GPT-NeoX model builder for compositional urns (locked 2L/4H/128d)."""

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from config import VOCAB_SIZE


def build_model(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model_config = GPTNeoXConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        intermediate_size=cfg.d_ff,
        max_position_embeddings=cfg.max_position_embeddings,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_bias=False,
        use_parallel_residual=False,
        use_cache=False,
    )
    return GPTNeoXForCausalLM(model_config)
