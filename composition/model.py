"""Model for the compositional lookup-table ICL task.

A small decoder-only GPT-NeoX language model over the 105-token vocabulary.
Using the causal-LM head means the standard shifted next-token cross-entropy is
computed internally, and label positions set to -100 (the random input tokens)
are ignored automatically -- i.e. loss is scored only on the deterministic
output tokens.
"""

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from config import VOCAB_SIZE


def build_model(cfg):
    """Initialize a GPTNeoXForCausalLM from a CompositionConfig."""
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
    model = GPTNeoXForCausalLM(model_config)
    return model
