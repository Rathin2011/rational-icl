"""Print small concrete examples of g, f, h, and training sequences.

Run from the composition/ directory:
    python generate_sample.py
    python generate_sample.py --num_tasks 2 --num_examples 2

Requires only numpy (no torch). For training, use the transformer conda env.
"""

import argparse

import numpy as np

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
)


def sample_tasks(num_tasks, seed):
    rng = np.random.RandomState(seed)
    g = rng.randint(0, Z_SIZE, size=(num_tasks, X_SIZE))
    f = rng.randint(0, Y_SIZE, size=(num_tasks, Z_SIZE))
    return g, f


def build_sequence(g_row, f_row, rng, seq_type):
    input_ids = np.empty(SEQ_LEN, dtype=np.int64)
    labels = np.full(SEQ_LEN, -100, dtype=np.int64)

    for i in range(NUM_PAIRS):
        if seq_type == "comp":
            x = int(rng.randint(X_SIZE))
            z = int(g_row[x])
            y = int(f_row[z])
            in_id = X_OFFSET + x
            out_id = Y_OFFSET + y
        elif seq_type == "g_only":
            x = int(rng.randint(X_SIZE))
            z = int(g_row[x])
            in_id = X_OFFSET + x
            out_id = Z_OFFSET + z
        else:
            z = int(rng.randint(Z_SIZE))
            y = int(f_row[z])
            in_id = Z_OFFSET + z
            out_id = Y_OFFSET + y

        input_ids[2 * i] = in_id
        input_ids[2 * i + 1] = out_id
        labels[2 * i + 1] = out_id

    return input_ids, labels


def token_name(token_id):
    """Decode a token id to a human-readable label."""
    if X_OFFSET <= token_id < Z_OFFSET:
        return f"x{token_id - X_OFFSET}"
    if Z_OFFSET <= token_id < Y_OFFSET:
        return f"z{token_id - Z_OFFSET}"
    if Y_OFFSET <= token_id < Y_OFFSET + Y_SIZE:
        return f"y{token_id - Y_OFFSET}"
    return f"?{token_id}"


def print_lookup_table(name, table, input_prefix, output_prefix):
    """Pretty-print a lookup table mapping input indices to output indices."""
    print(f"{name} (lookup table):")
    for i, out in enumerate(table):
        print(f"  {input_prefix}{i} -> {output_prefix}{out}")
    print()


def print_composite_h(g_row, f_row, x_indices):
    """Show h(x) = f(g(x)) for a few x values."""
    print("Composite function h(x) = f(g(x)):")
    for x in x_indices:
        z = int(g_row[x])
        y = int(f_row[z])
        print(f"  x{x} -> g(x{x}) = z{z} -> f(z{z}) = y{y}")
    print()


def print_sequence_example(g_row, f_row, rng, seq_type, example_idx):
    """Build and print one training sequence with labels/masking explained."""
    input_ids, labels = build_sequence(g_row, f_row, rng, seq_type)

    print(f"Example {example_idx}  [type: {seq_type}]")
    print(f"  token ids : {input_ids.tolist()}")
    print(f"  labels    : {labels.tolist()}  (-100 = masked, not scored in loss)")
    print("  pairs:")

    for i in range(NUM_PAIRS):
        in_tok = int(input_ids[2 * i])
        out_tok = int(input_ids[2 * i + 1])
        label = int(labels[2 * i + 1])
        scored = "scored" if label != -100 else "masked"
        print(
            f"    pair {i + 1:2d}: {token_name(in_tok):>4} -> {token_name(out_tok):>4}"
            f"   (output label={label}, {scored})"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show what g, f, h are and what training examples look like."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=1, help="per sequence type")
    parser.add_argument(
        "--show_full_tables",
        action="store_true",
        help="print all 50 g-entries (default: first 8 only)",
    )
    args = parser.parse_args()

    g, f = sample_tasks(args.num_tasks, seed=args.seed)
    rng = np.random.RandomState(args.seed + 42)

    print("=" * 72)
    print("COMPOSITIONAL LOOKUP-TABLE TASK: g, f, and h")
    print("=" * 72)
    print()
    print("Setup:")
    print(f"  |X| = {X_SIZE}, |Z| = {Z_SIZE}, |Y| = {Y_SIZE}")
    print(f"  g: X -> Z   (each x maps to some z)")
    print(f"  f: Z -> Y   (each z maps to some y)")
    print(f"  h: X -> Y   defined by h(x) = f(g(x))")
    print(f"  vocab: x0..x{X_SIZE - 1} (ids {X_OFFSET}..{Z_OFFSET - 1}), "
          f"z0..z{Z_SIZE - 1} (ids {Z_OFFSET}..{Y_OFFSET - 1}), "
          f"y0..y{Y_SIZE - 1} (ids {Y_OFFSET}..{Y_OFFSET + Y_SIZE - 1})")
    print()

    task_idx = 0
    g_row = g[task_idx]
    f_row = f[task_idx]

    print(f"Sampled task d = {task_idx} (from prior: g ~ Uniform(Z^{X_SIZE}), "
          f"f ~ Uniform(Y^{Z_SIZE}))")
    print()

    n_show = X_SIZE if args.show_full_tables else min(8, X_SIZE)
    print_lookup_table("g", g_row[:n_show], "x", "z")
    if not args.show_full_tables and X_SIZE > n_show:
        print(f"  ... ({X_SIZE - n_show} more x entries omitted; use --show_full_tables)")
        print()
    print_lookup_table("f", f_row, "z", "y")

    x_demo = [0, 1, 2, 7, 17]
    print_composite_h(g_row, f_row, x_demo)

    print("=" * 72)
    print("TRAINING EXAMPLES (comp) AND EVAL-ONLY TYPES (g_only, f_only)")
    print("=" * 72)
    print()
    print("Each training example:")
    print("  1. picks one fixed task (g, f) from the pool of D tasks")
    print("  2. emits 16 interleaved (x, y) pairs with y = f(g(x))  [comp only]")
    print("  3. loss is scored only on output tokens (inputs masked with -100)")
    print()
    print("Training uses comp sequences only. The examples below also show")
    print("g_only and f_only formats used at evaluation time:")
    print()

    for seq_type in SEQ_TYPES:
        print("-" * 72)
        print(f"Type: {seq_type}")
        print("-" * 72)
        for ex in range(args.num_examples):
            print_sequence_example(g_row, f_row, rng, seq_type, ex + 1)

    if args.num_tasks > 1:
        print("=" * 72)
        print("OTHER TASKS IN THE POOL (summary)")
        print("=" * 72)
        for d in range(1, args.num_tasks):
            print(f"task {d}: g[0]={g[d, 0]}, g[1]={g[d, 1]}, ...  "
                  f"f[0]={f[d, 0]}, f[1]={f[d, 1]}, ...")
        print()


if __name__ == "__main__":
    main()
