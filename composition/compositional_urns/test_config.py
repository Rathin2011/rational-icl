"""Tests for config.py's new concentration/save-schedule parameters.

Focus: backward compatibility (default-concentration runs must be byte-for-
byte unaffected -- same cache paths, same run names, same save schedule) and
correctness of the new behavior when non-defaults are passed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CompUrnsConfig, PHASE2_SAVE_STEPS, default_save_steps


def test_default_save_steps_matches_legacy_tuple_at_60k():
    assert default_save_steps(60_000) == PHASE2_SAVE_STEPS


def test_default_save_steps_spans_full_range_when_extended():
    steps = default_save_steps(300_000)
    assert steps[0] >= 1000
    assert steps[-1] == 300_000
    assert list(steps) == sorted(steps)
    assert len(set(steps)) == len(steps)  # no duplicates


def test_default_concentration_run_name_and_paths_unchanged():
    """Backward compatibility: default concentration must not alter the
    locked cache path / run name scheme at all."""
    cfg = CompUrnsConfig(num_tasks=64, seed=1, phase="baseline", run_tag="comp-only")
    assert "gc" not in cfg.run_name()
    assert "gc" not in cfg.train_tasks_path
    assert cfg.run_name() == "phasebaseline-D64-X20Z6Y20-2L4H128d-lr0.001-bs256-seed1-comp-only"


def test_nondefault_concentration_gets_distinct_paths():
    cfg_default = CompUrnsConfig(num_tasks=64, seed=1, phase="baseline")
    cfg_sharp = CompUrnsConfig(
        num_tasks=64, seed=1, phase="baseline", g_concentration=0.1, f_concentration=0.1
    )
    assert cfg_default.run_name() != cfg_sharp.run_name()
    assert cfg_default.train_tasks_path != cfg_sharp.train_tasks_path
    assert "gc0.1" in cfg_sharp.run_name()


def test_resolved_save_steps_override():
    cfg = CompUrnsConfig(
        num_tasks=64, seed=1, phase="baseline", max_steps=10_000, save_steps=(2000, 5000, 10000)
    )
    assert cfg.resolved_save_steps == (2000, 5000, 10000)


def test_deeper_architecture_gets_distinct_run_name():
    cfg2 = CompUrnsConfig(num_tasks=64, seed=1, phase="baseline", n_layers=2)
    cfg4 = CompUrnsConfig(num_tasks=64, seed=1, phase="baseline", n_layers=4)
    assert "2L" in cfg2.run_name()
    assert "4L" in cfg4.run_name()
    assert cfg2.run_name() != cfg4.run_name()


if __name__ == "__main__":
    test_default_save_steps_matches_legacy_tuple_at_60k()
    test_default_save_steps_spans_full_range_when_extended()
    test_default_concentration_run_name_and_paths_unchanged()
    test_nondefault_concentration_gets_distinct_paths()
    test_resolved_save_steps_override()
    test_deeper_architecture_gets_distinct_run_name()
    print("All config.py tests passed.")
