"""Train two-phase / baseline compositional urns models.

Usage (from compositional_urns/):
  python train.py --phase 1
  python train.py --phase 2 --resume_from /path/to/phase1/checkpoint-XXXX
  python train.py --phase baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    GPTNeoXForCausalLM,
)

from config import (
    CompUrnsConfig,
    SEQ_TYPES,
    PHASE1_MAX_STEPS,
    PHASE1_CKPT_EVERY,
    PHASE1_P_MIX,
    PHASE2_STEPS,
    PHASE2_P_MIX,
    PHASE2_LOSS_WEIGHTS,
    N_LAYERS,
    N_HEADS,
    D_MODEL,
    D_FF,
)
from data import get_or_create_tasks, CompUrnsIterable, CompUrnsEvalDataset
from model import build_model
from phase1_gate import evaluate_phase1_gate


class WeightedSeqTypeTrainer(Trainer):
    def __init__(self, *args, loss_weights=(1.0, 1.0, 1.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = tuple(float(w) for w in loss_weights)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        seq_type_id = inputs.pop("seq_type_id", None)
        inputs.pop("task_idx", None)
        inputs.pop("pairs", None)
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.size(-1)
        tok_loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        mask = (shift_labels != -100).float()

        if seq_type_id is None:
            loss = (tok_loss * mask).sum() / mask.sum().clamp_min(1.0)
            return (loss, outputs) if return_outputs else loss

        per_ex = (tok_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        if seq_type_id.dim() > 1:
            seq_type_id = seq_type_id.view(-1)

        loss = per_ex.new_zeros(())
        any_type = False
        for t, w in enumerate(self.loss_weights):
            if w == 0.0:
                continue
            m = seq_type_id == t
            if m.any():
                loss = loss + w * per_ex[m].mean()
                any_type = True
        if not any_type:
            loss = per_ex.mean()
        return (loss, outputs) if return_outputs else loss


class SaveStepsCallback(TrainerCallback):
    def __init__(self, steps, output_dir, model):
        self.steps = set(int(s) for s in steps)
        self.output_dir = output_dir
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", self.model)
        if state.global_step in self.steps or state.global_step == args.max_steps:
            path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            model.save_pretrained(path)
            print(f"Saved checkpoint at step {state.global_step} -> {path}")
        return control


class Phase1GateCallback(TrainerCallback):
    """Every gate_every steps, check Bayes CE gate; stop early on first pass."""

    def __init__(
        self,
        model,
        g_eval,
        f_eval,
        output_dir,
        gate_every,
        device,
        g_concentration=1.0,
        f_concentration=1.0,
    ):
        self.model = model
        self.g_eval = g_eval
        self.f_eval = f_eval
        self.output_dir = output_dir
        self.gate_every = int(gate_every)
        self.device = device
        self.g_concentration = g_concentration
        self.f_concentration = f_concentration
        self.passed_step = None
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0 or (step % self.gate_every) != 0:
            return control
        model = kwargs.get("model", self.model)
        ok, metrics = evaluate_phase1_gate(
            model,
            self.g_eval,
            self.f_eval,
            self.device,
            g_concentration=self.g_concentration,
            f_concentration=self.f_concentration,
        )
        metrics["step"] = step
        self.history.append(metrics)
        path = os.path.join(self.output_dir, "phase1_gate_history.json")
        with open(path, "w") as fh:
            json.dump(self.history, fh, indent=2)
        print(
            f"[Phase1 gate @ {step}] "
            f"gap_g={metrics['gap_g']:.4f} gap_f={metrics['gap_f']:.4f} "
            f"passed={ok}"
        )
        if ok and self.passed_step is None:
            self.passed_step = step
            ckpt = os.path.join(self.output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt)
            marker = os.path.join(self.output_dir, "phase1_passed.json")
            with open(marker, "w") as fh:
                json.dump({"step": step, "checkpoint": ckpt, **metrics}, fh, indent=2)
            print(f"Phase-1 gate PASSED at step {step}. Stopping. ckpt={ckpt}")
            control.should_training_stop = True
        return control


def _data_collator(features):
    """Stack tensors; drop non-tensor keys like pairs."""
    batch = {}
    keys = [k for k in features[0] if k in ("input_ids", "labels", "seq_type_id")]
    for k in keys:
        batch[k] = torch.stack(
            [
                f[k] if torch.is_tensor(f[k]) else torch.tensor(f[k], dtype=torch.long)
                for f in features
            ]
        )
    return batch


def run_training(cfg: CompUrnsConfig):
    os.makedirs(cfg.run_path, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    cfg.save()
    print(f"=== Run: {cfg.run_name()} ===")
    print(f"CUDA: {torch.cuda.is_available()}  phase={cfg.phase}")
    print(f"p_mix={cfg.p_mix}  loss_weights={cfg.loss_weights}")

    train_w_g, train_w_f = get_or_create_tasks(
        cfg.train_tasks_path,
        cfg.num_tasks,
        cfg.seed,
        g_concentration=cfg.g_concentration,
        f_concentration=cfg.f_concentration,
    )
    ood_w_g, ood_w_f = get_or_create_tasks(
        cfg.ood_tasks_path,
        cfg.num_ood_tasks,
        cfg.seed + 1000,
        g_concentration=cfg.g_concentration,
        f_concentration=cfg.f_concentration,
    )

    train_dataset = CompUrnsIterable(train_w_g, train_w_f, cfg.p_mix, seed=cfg.seed)

    if cfg.resume_from:
        print(f"Loading weights from {cfg.resume_from}")
        model = GPTNeoXForCausalLM.from_pretrained(cfg.resume_from)
    else:
        model = build_model(cfg)
    print(f"Parameters: {model.num_parameters():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_args = TrainingArguments(
        output_dir=cfg.checkpoints_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        seed=cfg.seed,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = WeightedSeqTypeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_data_collator,
        loss_weights=cfg.loss_weights,
    )

    if cfg.phase == "1":
        save_steps = list(
            range(PHASE1_CKPT_EVERY, cfg.max_steps + 1, PHASE1_CKPT_EVERY)
        )
        if cfg.max_steps not in save_steps:
            save_steps.append(cfg.max_steps)
        trainer.add_callback(SaveStepsCallback(save_steps, cfg.checkpoints_dir, model))
        g_eval = CompUrnsEvalDataset(
            train_w_g, train_w_f, "g_only", n_sequences=cfg.n_eval, seed=cfg.seed
        )
        f_eval = CompUrnsEvalDataset(
            train_w_g, train_w_f, "f_only", n_sequences=cfg.n_eval, seed=cfg.seed + 1
        )
        gate_cb = Phase1GateCallback(
            model,
            g_eval,
            f_eval,
            cfg.checkpoints_dir,
            gate_every=PHASE1_CKPT_EVERY,
            device=device,
            g_concentration=cfg.g_concentration,
            f_concentration=cfg.f_concentration,
        )
        trainer.add_callback(gate_cb)
    elif cfg.phase in ("2", "baseline"):
        trainer.add_callback(
            SaveStepsCallback(cfg.resolved_save_steps, cfg.checkpoints_dir, model)
        )
    else:
        raise ValueError(f"Unknown phase: {cfg.phase}")

    trainer.train()

    logs = trainer.state.log_history
    logs_path = os.path.join(cfg.run_path, "logs.csv")
    all_keys = sorted(set().union(*(d.keys() for d in logs))) if logs else []
    with open(logs_path, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, restval="NA")
        writer.writeheader()
        writer.writerows(logs)
    print(f"Wrote {logs_path}")

    # Final gate / note for phase 1
    if cfg.phase == "1":
        marker = os.path.join(cfg.checkpoints_dir, "phase1_passed.json")
        if os.path.exists(marker):
            print(f"Phase 1 SUCCESS: {marker}")
        else:
            print(
                "Phase 1 INCONCLUSIVE: gate never passed within "
                f"{cfg.max_steps} steps. See phase1_gate_history.json"
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=["1", "2", "baseline"],
        default="1",
        help="1=g+f pretrain+gate; 2=comp+replay from resume; baseline=scratch comp-only",
    )
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--num_tasks", "-D", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument(
        "--g_concentration", type=float, default=1.0,
        help="Dirichlet concentration for true w_g rows; <1 sharpens (less diffuse)",
    )
    p.add_argument(
        "--f_concentration", type=float, default=1.0,
        help="Dirichlet concentration for true w_f rows; <1 sharpens (less diffuse)",
    )
    p.add_argument("--n_layers", type=int, default=None, help="override locked arch (default 2)")
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument(
        "--save_every", type=int, default=None,
        help="save a checkpoint every N steps (plus final); default: auto log-spaced schedule",
    )
    return p.parse_args()


def config_for_phase(args) -> CompUrnsConfig:
    if args.phase == "1":
        p_comp, p_g, p_f = PHASE1_P_MIX
        w = (1.0, 1.0, 1.0)
        max_steps = args.max_steps or PHASE1_MAX_STEPS
        resume = args.resume_from
        tag = args.run_tag or "gf"
    elif args.phase == "2":
        p_comp, p_g, p_f = PHASE2_P_MIX
        w = PHASE2_LOSS_WEIGHTS
        max_steps = args.max_steps or PHASE2_STEPS
        resume = args.resume_from
        if not resume:
            raise SystemExit("--phase 2 requires --resume_from <phase1 checkpoint>")
        tag = args.run_tag or "comp-replay"
    else:  # baseline
        p_comp, p_g, p_f = 1.0, 0.0, 0.0
        w = (1.0, 0.0, 0.0)
        max_steps = args.max_steps or PHASE2_STEPS
        resume = None
        tag = args.run_tag or "comp-only"

    if args.save_every:
        save_steps = list(range(args.save_every, max_steps + 1, args.save_every))
        if not save_steps or save_steps[-1] != max_steps:
            save_steps.append(max_steps)
    else:
        save_steps = None

    return CompUrnsConfig(
        num_tasks=args.num_tasks,
        seed=args.seed,
        max_steps=max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_layers=args.n_layers if args.n_layers is not None else N_LAYERS,
        n_heads=args.n_heads if args.n_heads is not None else N_HEADS,
        d_model=args.d_model if args.d_model is not None else D_MODEL,
        d_ff=args.d_ff if args.d_ff is not None else D_FF,
        g_concentration=args.g_concentration,
        f_concentration=args.f_concentration,
        save_steps=save_steps,
        p_comp=p_comp,
        p_g_only=p_g,
        p_f_only=p_f,
        w_comp=w[0],
        w_g_only=w[1],
        w_f_only=w[2],
        run_tag=tag,
        cache_dir=args.cache_dir,
        phase=args.phase if args.phase != "baseline" else "baseline",
        resume_from=resume,
        n_eval=args.n_eval,
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = config_for_phase(args)
    run_training(cfg)
