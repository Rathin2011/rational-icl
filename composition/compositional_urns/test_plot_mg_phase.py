"""Input/output contract tests for plot_mg_phase.py.

aggregate_grid is pure pandas/numpy -- tested directly against synthetic
rows. load_all does path resolution + file I/O -- tested against small
hand-written CSVs under a temp cache dir, exercising both the found-file and
missing-file paths.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CompUrnsConfig
from plot_mg_phase import aggregate_grid, load_all


def _row(num_tasks, seed, step, split, L, tv_mg, retained, d_rel_m):
    return {
        "num_tasks": num_tasks,
        "seed": seed,
        "step": step,
        "split": split,
        "L": L,
        "tv_mg": tv_mg,
        "retained": retained,
        "d_rel_m": d_rel_m,
    }


def test_aggregate_grid_basic_shape_and_values():
    rows = [
        _row(4, 1, 100, "ood", 32, 0.2, True, 0.9),  # memorizing-ish... high d_rel_m
        _row(4, 1, 100, "ood", 32, 0.3, True, 0.7),
        _row(64, 1, 100, "ood", 32, 0.2, True, 0.1),
        _row(64, 1, 200, "ood", 32, 0.2, True, 0.2),
    ]
    df = pd.DataFrame(rows)
    steps, d_values, grid, retention = aggregate_grid(df, split="ood", L=32)

    assert d_values == [4, 64]
    assert steps == [100, 200]
    # D=4, step=100: mean of [0.9, 0.7] = 0.8
    assert np.isclose(grid[d_values.index(4), steps.index(100)], 0.8)
    # D=64, step=200: single value 0.2
    assert np.isclose(grid[d_values.index(64), steps.index(200)], 0.2)
    # D=64, step=100 has one row; D=4, step=200 has no rows -> NaN
    assert np.isnan(grid[d_values.index(4), steps.index(200)])


def test_aggregate_grid_filters_split_and_L():
    rows = [
        _row(4, 1, 100, "ood", 32, 0.2, True, 0.9),
        _row(4, 1, 100, "id", 32, 0.2, True, 0.1),  # wrong split, must be excluded
        _row(4, 1, 100, "ood", 8, 0.2, True, 0.5),  # wrong L, must be excluded
    ]
    df = pd.DataFrame(rows)
    steps, d_values, grid, retention = aggregate_grid(df, split="ood", L=32)
    assert grid.shape == (1, 1)
    assert np.isclose(grid[0, 0], 0.9)


def test_aggregate_grid_excludes_unretained_from_mean_but_not_from_retention():
    rows = [
        _row(4, 1, 100, "ood", 32, 0.01, False, 0.9),  # not retained (TV too low)
        _row(4, 1, 100, "ood", 32, 0.20, True, 0.3),
    ]
    df = pd.DataFrame(rows)
    steps, d_values, grid, retention = aggregate_grid(df, split="ood", L=32)
    # mean should only reflect the retained row (0.3), not the excluded 0.9
    assert np.isclose(grid[0, 0], 0.3)
    # retention_rate is computed over ALL rows (1 of 2 retained = 0.5)
    assert np.isclose(retention.loc[(4, 100)], 0.5)


def test_load_all_reads_existing_and_skips_missing():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = CompUrnsConfig(num_tasks=4, seed=1, run_tag="comp-only", phase="baseline", cache_dir=tmp)
        os.makedirs(cfg.run_path, exist_ok=True)
        pd.DataFrame([_row(4, 1, 100, "ood", 32, 0.2, True, 0.5)]).to_csv(
            os.path.join(cfg.run_path, "mg_sweep.csv"), index=False
        )
        # D=64 has no csv on disk -- should be skipped, not raise.
        df = load_all(
            d_values=[4, 64],
            seeds=[1],
            run_tag="comp-only",
            phase="baseline",
            cache_dir=tmp,
            batch_size=256,
            learning_rate=1e-3,
        )
        assert len(df) == 1
        assert df.iloc[0]["num_tasks"] == 4


def test_load_all_raises_if_nothing_found():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_all(
                d_values=[999],
                seeds=[1],
                run_tag="comp-only",
                phase="baseline",
                cache_dir=tmp,
                batch_size=256,
                learning_rate=1e-3,
            )
            assert False, "expected SystemExit"
        except SystemExit:
            pass


if __name__ == "__main__":
    test_aggregate_grid_basic_shape_and_values()
    test_aggregate_grid_filters_split_and_L()
    test_aggregate_grid_excludes_unretained_from_mean_but_not_from_retention()
    test_load_all_reads_existing_and_skips_missing()
    test_load_all_raises_if_nothing_found()
    print("All plot_mg_phase tests passed.")
