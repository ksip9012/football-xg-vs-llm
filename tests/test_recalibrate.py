"""recalibrate のテスト（作り物のデータで確かめる）。"""

import numpy as np
import pytest

from football_xg_vs_llm.cv import assign_folds
from football_xg_vs_llm.recalibrate import (
    fit_platt,
    fit_shift,
    logit,
    recalibrate,
)
from football_xg_vs_llm.xg import roc_auc, sigmoid


def _data(n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """本当の確率より、対数オッズで 0.5 低く、傾き 0.8 で答える作り物。"""
    rng = np.random.default_rng(seed)
    true = rng.uniform(0.02, 0.6, size=n)
    goals = (rng.uniform(size=n) < true).astype(np.float64)
    answer = sigmoid(-0.5 + 0.8 * logit(true))
    return answer, goals


def test_fit_platt_finds_the_inverse_of_the_distortion() -> None:
    answer, goals = _data(200_000)

    a, b = fit_platt(answer, goals)

    # 答え = σ(−0.5 + 0.8 z) を z に戻すと、a = 0.5 / 0.8・b = 1 / 0.8
    assert a == pytest.approx(0.5 / 0.8, abs=0.05)
    assert b == pytest.approx(1 / 0.8, abs=0.05)


def test_fit_shift_matches_the_mean_goal_rate() -> None:
    answer, goals = _data(5_000, seed=1)

    a = fit_shift(answer, goals)

    # 切片だけの最尤推定では、直した確率の合計がゴールの数と一致する
    assert np.sum(sigmoid(logit(answer) + a)) == pytest.approx(goals.sum())
    assert a > 0


def test_recalibrate_keeps_the_order_and_uses_other_matches() -> None:
    answer, goals = _data(4_000, seed=2)
    match_ids = np.arange(4_000, dtype=np.int64) // 10

    fixed = recalibrate(answer, goals, match_ids)

    assert len(fixed.platt_coefs) == 5
    # 同じ組の中では順位を変えない。組ごとに数が少し違うので、組をまたぐと
    # 順位がわずかに入れ替わり、全体の AUC は少しだけ変わる
    fold = assign_folds(match_ids)
    for k in range(5):
        part = fold == k
        before = roc_auc(goals[part], answer[part])
        assert roc_auc(goals[part], fixed.platt[part]) == pytest.approx(before)
        assert roc_auc(goals[part], fixed.shift[part]) == pytest.approx(before)
    assert roc_auc(goals, fixed.platt) == pytest.approx(
        roc_auc(goals, answer), abs=0.005
    )
    # 低めに出していたので、直すと平均が実際の率に近づく
    gap_before = abs(answer.mean() - goals.mean())
    gap_after = abs(fixed.platt.mean() - goals.mean())
    assert gap_after < gap_before
