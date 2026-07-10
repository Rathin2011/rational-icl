"""Train a transformer on the compositional lookup-table ICL task.

Training uses comp-only (x, y) sequences with y = f(g(x)). Eval additionally
logs g_only and f_only metrics. Autoregressive next-token cross-entropy (scored
on output tokens only), AdamW, sweeping task diversity D.

Usage (from the composition/ directory):
    python train.py --num_tasks 64
    python train.py --num_tasks 4 --max_steps 200 --eval_steps 50   # quick run
"""

import os
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)

from config import CompositionConfig
from data import get_or_create_tasks, CompositionTrainDataset, build_eval_sets
from model import build_model


def create_sqrt_checkpoint_schedule(total_steps, num_checkpoints, start_step=20):
    """Checkpoint steps spaced ~linearly in sqrt(step), to capture early dynamics."""
    if num_checkpoints <= 0:
        return [total_steps]
    start_step = max(1, min(start_step, total_steps - 1))
    sqrt_space = np.linspace(np.sqrt(start_step), np.sqrt(total_steps), num_checkpoints)
    steps = sorted(set(int(s ** 2) for s in sqrt_space))
    if steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


class SqrtSaveCallback(TrainerCallback):
    """Save model weights at sqrt-spaced steps (plus the final step)."""

    def __init__(self, steps, output_dir, model):
        self.steps = set(steps)
        self.output_dir = output_dir
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps or state.global_step == args.max_steps:
            path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            self.model.save_pretrained(path)
            print(f"Saved checkpoint at step {state.global_step} -> {path}")
        return control


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to per-token (logprob-at-label, correct, mask) to save memory.

    Mirrors the causal-LM shift: token at position t is predicted from position t-1.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]

    logprobs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp_min(0)
    tok_logprob = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    preds = shift_logits.argmax(dim=-1)
    correct = (preds == shift_labels).float()
    mask = (shift_labels != -100).float()
    # (batch, seq_len - 1, 3)
    return torch.stack([tok_logprob, correct, mask], dim=-1)


def compute_metrics(eval_pred):
    """Cross-entropy and token accuracy over scored (output) positions."""
    preds = eval_pred.predictions  # (N, seq_len - 1, 3)
    tok_logprob = preds[..., 0]
    correct = preds[..., 1]
    mask = preds[..., 2].astype(bool)
    n = int(mask.sum())
    if n == 0:
        return {"ce": float("nan"), "accuracy": float("nan")}
    ce = float(-tok_logprob[mask].sum() / n)
    accuracy = float(correct[mask].sum() / n)
    return {"ce": ce, "accuracy": accuracy}


def run_training(cfg):
    print(f"=== Run: {cfg.run_name} ===")
    print(f"CUDA available: {torch.cuda.is_available()}")

    cfg.save()

    # Tasks: D training tasks (fixed) + held-out OOD tasks from the same prior.
    train_g, train_f = get_or_create_tasks(
        cfg.train_tasks_path, cfg.num_tasks, cfg.seed
    )
    ood_g, ood_f = get_or_create_tasks(
        cfg.ood_tasks_path, cfg.num_ood_tasks, cfg.seed + 1000
    )

    train_dataset = CompositionTrainDataset(train_g, train_f, seed=cfg.seed)
    eval_sets = build_eval_sets(
        train_g, train_f, ood_g, ood_f, n_eval=cfg.n_eval, seed=cfg.seed
    )

    model = build_model(cfg)
    print(f"Model parameters: {model.num_parameters():,}")

    training_args = TrainingArguments(
        output_dir=cfg.checkpoints_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="no",  # handled by SqrtSaveCallback
        seed=cfg.seed,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_sets,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    schedule = create_sqrt_checkpoint_schedule(cfg.max_steps, cfg.num_checkpoints)
    trainer.add_callback(SqrtSaveCallback(schedule, cfg.checkpoints_dir, model))

    trainer.train()

    # Dump the full metric log to CSV.
    logs = trainer.state.log_history
    os.makedirs(cfg.run_dir, exist_ok=True)
    all_keys = sorted(set().union(*(d.keys() for d in logs))) if logs else []
    logs_path = os.path.join(cfg.run_dir, "logs.csv")
    with open(logs_path, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, restval="NA")
        writer.writeheader()
        writer.writerows(logs)
    print(f"Wrote logs to {logs_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train on the compositional lookup task.")
    p.add_argument("--num_tasks", "-D", type=int, default=64, help="Task diversity D")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--lr_scheduler_type", type=str, default="constant")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--num_checkpoints", type=int, default=50)
    p.add_argument("--num_ood_tasks", type=int, default=256)
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--max_position_embeddings", type=int, default=128)
    p.add_argument("--cache_dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = CompositionConfig(
        num_tasks=args.num_tasks,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        num_checkpoints=args.num_checkpoints,
        num_ood_tasks=args.num_ood_tasks,
        n_eval=args.n_eval,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_position_embeddings=args.max_position_embeddings,
        cache_dir=args.cache_dir,
    )
    run_training(cfg)
