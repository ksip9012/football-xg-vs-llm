"""最初の xG を 3通り（numpy・scikit-learn・statsmodels）で作って比べる。

同じ特徴・同じ分け方・同じ物差しで動かし、違いが作り方だけになるようにする。
結果は Markdown の表で標準出力に出す（`mise run xg`）。
決めごとは decisions/0004。
"""

import statistics
import sys
import time
from collections.abc import Callable

import numpy as np

from football_xg_vs_llm.libraries import Fit, fit_sklearn, fit_statsmodels
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import (
    MIN_X,
    Dataset,
    Floats,
    brier_score,
    build_dataset,
    calibration,
    fit_numpy,
    log_loss,
    predict,
    roc_auc,
    split_by_match,
)

REPEATS = 5

Fitter = Callable[[Floats, Floats, Floats], Fit]


def _fit_numpy(train: Floats, goals: Floats, test: Floats) -> Fit:
    coef = fit_numpy(train, goals)
    return Fit(coef=coef, test_probs=predict(coef, test))


FITTERS: dict[str, Fitter] = {
    "numpy": _fit_numpy,
    "scikit-learn": fit_sklearn,
    "statsmodels": fit_statsmodels,
}


def _timed(
    fitter: Fitter, train: Floats, goals: Floats, test: Floats
) -> tuple[Fit, float]:
    """`REPEATS` 回当てはめ、最後の結果と、時間の中央値（ミリ秒）を返す。"""
    times: list[float] = []
    fit: Fit | None = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        fit = fitter(train, goals, test)
        times.append((time.perf_counter() - start) * 1000)
    assert fit is not None
    return fit, statistics.median(times)


def _metrics(goals: Floats, probs: Floats) -> str:
    return (
        f"{log_loss(goals, probs):.5f} | {brier_score(goals, probs):.5f} | "
        f"{roc_auc(goals, probs):.4f}"
    )


def compare(shots: list[Shot], *, with_head: bool) -> None:
    """1つの特徴の組で、3通りを当てはめて表にする。"""
    data: Dataset = build_dataset(shots, with_head=with_head)
    train, test = split_by_match(data.match_ids)
    x_train, y_train = data.features[train], data.goals[train]
    x_test, y_test = data.features[test], data.goals[test]
    names = ("intercept", *data.names)
    print(f"\n## 特徴: {', '.join(data.names)}\n")

    results = {
        name: _timed(f, x_train, y_train, x_test)
        for name, f in FITTERS.items()
    }
    base = results["numpy"][0]

    print(
        "| 作り方 | "
        + " | ".join(names)
        + " | 係数の差の最大 | 確率の差の最大 | 時間（ms） |"
    )
    print(
        "| --- | " + " | ".join("---" for _ in names) + " | --- | --- | --- |"
    )
    for name, (fit, ms) in results.items():
        coef = " | ".join(f"{c:.6f}" for c in fit.coef)
        coef_diff = float(np.max(np.abs(fit.coef - base.coef)))
        prob_diff = float(np.max(np.abs(fit.test_probs - base.test_probs)))
        diffs = f"{coef_diff:.1e} | {prob_diff:.1e}"
        print(f"| {name} | {coef} | {diffs} | {ms:.1f} |")

    sm_fit = results["statsmodels"][0]
    if sm_fit.std_err is not None and sm_fit.p_values is not None:
        assert sm_fit.conf_low is not None and sm_fit.conf_high is not None
        print("\nstatsmodels の係数（95% 信頼区間・p 値）\n")
        print("| 項目 | 係数 | 標準誤差 | 95% 信頼区間 | p 値 |")
        print("| --- | --- | --- | --- | --- |")
        for i, n in enumerate(names):
            print(
                f"| {n} | {sm_fit.coef[i]:.4f} | {sm_fit.std_err[i]:.4f} | "
                f"{sm_fit.conf_low[i]:.4f}〜{sm_fit.conf_high[i]:.4f} | "
                f"{sm_fit.p_values[i]:.1e} |"
            )

    rate = y_train.mean()
    constant = np.full(len(y_test), rate)
    print("\nテストの物差し\n")
    print("| モデル | ログ損失 | ブライアスコア | AUC |")
    print("| --- | --- | --- | --- |")
    print(f"| 平均の率だけ（{rate:.4f}） | {_metrics(y_test, constant)} |")
    for name, (fit, _) in results.items():
        print(f"| {name} | {_metrics(y_test, fit.test_probs)} |")

    print("\n確率の当たり具合（numpy。テストを確率の低い順に 10等分）\n")
    print("| 段階 | 本数 | 出した確率の平均 | 実際のゴールの率 |")
    print("| --- | --- | --- | --- |")
    for i, b in enumerate(calibration(y_test, base.test_probs), start=1):
        print(f"| {i} | {b.count} | {b.mean_prob:.3f} | {b.goal_rate:.3f} |")


def main() -> int:
    """リポジトリの `data/shots.csv` で 3通りを比べる（`mise run xg`）。"""
    shots = load_shots_csv()
    data = build_dataset(shots, with_head=False)
    train, test = split_by_match(data.match_ids)
    normal = [s for s in shots if s.shot_type == "shot"]
    print("# 最初の xG（ロジスティック回帰）の 3通りの比べ\n")
    print(
        f"- 普通のシュート {len(normal)}本のうち、自陣（x<{MIN_X}）の "
        f"{len(normal) - len(data.goals)}本を外した {len(data.goals)}本を使う"
    )
    for label, rows in (("学習", train), ("テスト", test)):
        matches = len(np.unique(data.match_ids[rows]))
        print(
            f"- {label}: {int(rows.sum())}本・{matches}試合"
            f"（ゴール {data.goals[rows].mean():.4f}）"
        )
    print(f"- 時間は {REPEATS}回の中央値")
    for with_head in (False, True):
        compare(shots, with_head=with_head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
