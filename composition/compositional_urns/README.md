# Stochastic compositional Balls & Urns (two-phase C_GG)

Implements the design-review-locked protocol for a single-point $\mathrm{C}_{GG}$
test. See **[PROTOCOL.md](PROTOCOL.md)** for the full experimental spec.

## What's here

| File | Role |
|------|------|
| `PROTOCOL.md` | Locked settings, predictor algorithms, success criteria |
| `config.py` | Alphabets, arch (2L/4H/128), phase mixes, margins |
| `data.py` | Dirichlet row-stochastic tasks + typed sequences |
| `predictors.py` | $g$, $f$, $M$, atomic $G$, approx/exact $\mathrm{C}_{GG}$ |
| `phase1_gate.py` | CE-to-Bayes gate helpers |
| `relative_distance.py` | Soft 3-way $d_Q^{\mathrm{rel}}$ (symmetrized KL) |
| `probe.py` | Shared-prefix patching + $z$_decode |
| `test_predictors.py` | Unit tests (approx vs exact on short $L$) |

Training loop / SCC launchers are included.

## Train (SCC)

```bash
cd /projectnb/buinlp/rathin/rational-icl/composition/compositional_urns
mkdir -p logs
bash run_launch.sh          # Phase-1 + baseline
# After Phase-1 gate passes:
qsub -N cu_phase2 train_scc.sh 2
```

Local:
```bash
python train.py --phase 1
python train.py --phase baseline
python train.py --phase 2 --resume_from /path/to/checkpoint-XXXX
python eval_checkpoint.py --checkpoint /path/to/ckpt --out_json metrics.json
```

## Critical design locks (from review)

1. **Approx $\mathrm{C}_{GG}$** on comp prefixes (exact only for short $L$ unit tests).
2. Phase-1 gate uses **$g$/$f$ Dirichlet–Categorical CE**, tolerance 0.05 nats.
3. Success needs **patch margin and $z$_decode ≥ 0.4** (patch alone can fake flat G).
4. Soft $d^{\mathrm{rel}}$ on **ID and OOD**; margin $\varepsilon=0.05$.
5. Patch **layer 0**, shared prefix, patch $h_a$ into $x_b$ run (routing → $f(g(x_a))$).
