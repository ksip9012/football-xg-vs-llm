"""compare_llm のテスト（作り物のデータで確かめる）。"""

from pathlib import Path

import numpy as np
import pytest

from football_xg_vs_llm.compare_context import aligned
from football_xg_vs_llm.compare_llm import (
    bootstrap_differences,
    calibration_figure,
    difference_map,
    paired,
)
from football_xg_vs_llm.figures import style
from football_xg_vs_llm.models import Shot


def _shot(shot_id: int, *, match: int, x: int, y: int, goal: bool) -> Shot:
    return Shot.model_validate(
        {
            "shot_id": shot_id,
            "competition": "England",
            "match_id": match,
            "match_period": "1H",
            "event_sec": 1.0,
            "team_id": 1,
            "player_id": 1,
            "shot_type": "shot",
            "x": x,
            "y": y,
            "body_part": "right_foot",
            "is_counter_attack": False,
            "prev_event": None,
            "prev_seconds_before": None,
            "is_goal": goal,
        }
    )


def _data(
    n: int = 2000,
) -> tuple[list[Shot], dict[int, float], dict[int, float]]:
    rng = np.random.default_rng(3)
    shots: list[Shot] = []
    stat: dict[int, float] = {}
    llm: dict[int, float] = {}
    for i in range(n):
        p = float(rng.uniform(0.02, 0.5))
        shots.append(
            _shot(
                i,
                match=i % 40,
                x=int(rng.integers(75, 100)),
                y=int(rng.integers(30, 70)),
                goal=bool(rng.random() < p),
            )
        )
        stat[i] = p
        llm[i] = float(np.clip(p + rng.normal(0, 0.05), 0.001, 0.999))
    return shots, stat, llm


def test_paired_keeps_shots_with_both_probabilities() -> None:
    shots, stat, llm = _data(10)
    del llm[3]

    data = paired(shots, stat, llm)

    assert len(data.goals) == 9
    assert 3 not in {s.shot_id for s in data.shots}


def test_bootstrap_is_repeatable_and_the_range_holds_the_estimate() -> None:
    data = paired(*_data())

    first = bootstrap_differences(data, rounds=200, seed=1)
    second = bootstrap_differences(data, rounds=200, seed=1)

    assert first == second
    assert [d.name for d in first] == ["ログ損失", "ブライアスコア", "AUC"]
    for d in first:
        assert d.low <= d.estimate <= d.high


def test_bootstrap_of_the_same_model_gives_zero() -> None:
    shots, stat, _ = _data(500)
    data = paired(shots, stat, dict(stat))

    for d in bootstrap_differences(data, rounds=50, seed=1):
        assert d.estimate == pytest.approx(0)
        assert d.low == pytest.approx(0)
        assert d.high == pytest.approx(0)


def test_figures_are_written(tmp_path: Path) -> None:
    data = paired(*_data())
    style()

    calibration_figure(data, tmp_path / "calibration.png")
    difference_map(data, tmp_path / "map.png")

    for name in ("calibration.png", "map.png"):
        assert (tmp_path / name).read_bytes()[:4] == b"\x89PNG"


def test_aligned_keeps_shots_that_every_model_answered() -> None:
    shots = [
        _shot(i, match=i % 3, x=90, y=50, goal=i % 2 == 0) for i in range(1, 5)
    ]
    xgs = {
        "a": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
        "b": {1: 0.5, 2: 0.6, 4: 0.7},
    }

    rows, goals, probs = aligned(shots, xgs)

    assert [s.shot_id for s in rows] == [1, 2, 4]
    assert list(goals) == [0.0, 1.0, 1.0]
    assert list(probs["b"]) == [0.5, 0.6, 0.7]
