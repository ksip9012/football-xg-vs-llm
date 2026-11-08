"""xg と libraries のテスト（作り物のデータで確かめる）。

物差しは scikit-learn の関数と比べる。scikit-learn には型の情報がないので、
このファイルでは「型がわからない」の指摘を外す。
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

import math

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from football_xg_vs_llm.libraries import fit_sklearn, fit_statsmodels
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.xg import (
    Floats,
    brier_score,
    build_dataset,
    calibration,
    distance_and_angle,
    fit_numpy,
    log_loss,
    predict,
    roc_auc,
    split_by_match,
)


def _arr(*values: float) -> Floats:
    return np.array(values, dtype=np.float64)


def test_distance_and_angle_in_front_of_the_goal() -> None:
    # ゴールの正面 11m 弱（x=89.5 → 105 × 0.895 = 93.975m）
    distance, angle = distance_and_angle(_arr(89.5), _arr(50))

    assert distance[0] == pytest.approx(105 - 93.975)
    expected = 2 * math.atan((7.32 / 2) / (105 - 93.975))
    assert angle[0] == pytest.approx(expected)


def test_angle_is_pi_between_the_posts_and_small_from_the_side() -> None:
    _, on_line = distance_and_angle(_arr(100), _arr(50))
    _, wide = distance_and_angle(_arr(99), _arr(0))

    assert on_line[0] == pytest.approx(math.pi)
    assert 0 < wide[0] < 0.1


def test_angle_is_the_same_on_both_sides() -> None:
    _, angles = distance_and_angle(_arr(85, 85), _arr(30, 70))

    assert angles[0] == pytest.approx(angles[1])


def _shot(shot_id: int, *, x: int, match: int, goal: bool, head: bool) -> Shot:
    return Shot(
        shot_id=shot_id,
        competition="England",
        match_id=match,
        match_period="1H",
        event_sec=1.0,
        team_id=1,
        player_id=1,
        shot_type="shot",
        x=x,
        y=50,
        body_part="head_or_body" if head else "right_foot",
        is_counter_attack=False,
        prev_event=None,
        prev_seconds_before=None,
        is_goal=goal,
    )


def test_build_dataset_keeps_normal_shots_from_the_attacking_half() -> None:
    shots = [
        _shot(1, x=90, match=1, goal=True, head=True),
        _shot(2, x=49, match=1, goal=True, head=False),  # 自陣は外す
        _shot(3, x=80, match=2, goal=False, head=False).model_copy(
            update={"shot_type": "penalty"}  # PK は外す
        ),
        _shot(4, x=75, match=2, goal=False, head=False),
    ]

    data = build_dataset(shots, with_head=True)

    assert data.names == ("distance", "angle", "head")
    assert data.goals.tolist() == [1.0, 0.0]
    assert data.features[:, 2].tolist() == [1.0, 0.0]
    assert data.match_ids.tolist() == [1, 2]


def test_split_keeps_each_match_on_one_side_and_is_repeatable() -> None:
    match_ids = np.repeat(np.arange(100, dtype=np.int64), 3)

    train, test = split_by_match(match_ids)
    train2, _ = split_by_match(match_ids)

    assert set(match_ids[train]).isdisjoint(set(match_ids[test]))
    assert len(set(match_ids[test])) == 20
    assert (train == train2).all()


def _synthetic(n: int = 2000) -> tuple[Floats, Floats]:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 2))
    true_coef = _arr(-1.0, 0.8, -0.5)
    p = 1 / (1 + np.exp(-(true_coef[0] + features @ true_coef[1:])))
    goals = (rng.random(n) < p).astype(np.float64)
    return features, goals


def test_three_ways_give_the_same_coefficients() -> None:
    features, goals = _synthetic()

    own = fit_numpy(features, goals)
    sk = fit_sklearn(features, goals, features)
    sm = fit_statsmodels(features, goals, features)

    assert own == pytest.approx(sm.coef, abs=1e-8)
    assert own == pytest.approx(sk.coef, abs=1e-3)
    assert predict(own, features) == pytest.approx(sm.test_probs, abs=1e-8)
    assert sm.std_err is not None
    assert (sm.std_err > 0).all()


def test_metrics_match_scikit_learn() -> None:
    features, goals = _synthetic()
    probs = predict(fit_numpy(features, goals), features)

    assert log_loss(goals, probs) == pytest.approx(
        sk_metrics.log_loss(goals, probs)
    )
    assert brier_score(goals, probs) == pytest.approx(
        sk_metrics.brier_score_loss(goals, probs)
    )
    assert roc_auc(goals, probs) == pytest.approx(
        sk_metrics.roc_auc_score(goals, probs)
    )


def test_auc_counts_ties_as_half() -> None:
    goals = _arr(1, 0, 1, 0)
    probs = _arr(0.5, 0.5, 0.9, 0.1)

    assert roc_auc(goals, probs) == pytest.approx(0.875)


def test_calibration_splits_into_equal_bins() -> None:
    goals = _arr(0, 0, 1, 1)
    probs = _arr(0.1, 0.2, 0.7, 0.8)

    bins = calibration(goals, probs, bins=2)

    assert [b.count for b in bins] == [2, 2]
    assert bins[0].mean_prob == pytest.approx(0.15)
    assert bins[1].goal_rate == 1.0
