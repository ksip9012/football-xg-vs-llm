"""テストの 8,626本で、3つの情報と広い情報の 4つのモデルを比べる。

決めごとは decisions/0007。

- 4つ: 統計モデル（交差検証の xG）と Gemini の、それぞれ「距離・角度・
  頭か足か」だけのものと、直前のプレーの組とカウンターも足したもの
- 差の 95% の幅（試合単位のブートストラップ。decisions/0006 と同じ）を 3つ:
  統計モデルの広い − 3つ、Gemini の広い − 3つ、Gemini の広い − 統計モデルの広い
- 組ごと・カウンターの有無ごとに、実際の率と 4つの確率の平均
- 広い情報の統計モデルを全部のシュートで当てはめた係数（組ごとの効き方を読む）

`mise run compare-context`（先に `mise run xg-cv` と
`mise run llm-test-context`）。
"""

import sys
from collections.abc import Sequence

import numpy as np

from football_xg_vs_llm.compare_llm import (
    BOOTSTRAP,
    METRICS,
    Paired,
    bootstrap_differences,
)
from football_xg_vs_llm.cv import CONTEXT_XG_PATH, DEFAULT_XG_PATH, load_xg
from football_xg_vs_llm.llm_test import CONTEXT_XG_PATH as GEMINI_CONTEXT_PATH
from football_xg_vs_llm.llm_test import XG_PATH as GEMINI_PATH
from football_xg_vs_llm.llm_test import test_shots
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import (
    GROUP_ORDER,
    Floats,
    build_dataset,
    fit_numpy,
    prev_group,
)

LABELS = {
    "stat": "統計モデル（3つ）",
    "stat_context": "統計モデル（広い）",
    "gemini": "Gemini（3つ）",
    "gemini_context": "Gemini（広い）",
}
# 差を出す組（後ろ − 前）
PAIRS = (
    ("stat", "stat_context"),
    ("gemini", "gemini_context"),
    ("stat_context", "gemini_context"),
)


def aligned(
    shots: Sequence[Shot], xgs: dict[str, dict[int, float]]
) -> tuple[list[Shot], Floats, dict[str, Floats]]:
    """4つすべてに確率のあるシュートだけを、同じ並びにする。"""
    rows = [s for s in shots if all(s.shot_id in xg for xg in xgs.values())]
    goals = np.array([s.is_goal for s in rows], dtype=np.float64)
    probs = {
        name: np.array([xg[s.shot_id] for s in rows])
        for name, xg in xgs.items()
    }
    return rows, goals, probs


def pair_of(
    rows: Sequence[Shot], goals: Floats, before: Floats, after: Floats
) -> Paired:
    """`bootstrap_differences` に渡す形（差は after − before）。"""
    return Paired(
        shots=list(rows),
        goals=goals,
        stat=before,
        llm=after,
        match_ids=np.array([s.match_id for s in rows], dtype=np.int64),
    )


def report(
    rows: Sequence[Shot], goals: Floats, probs: dict[str, Floats]
) -> None:
    """物差し・差の幅・組ごとの表を Markdown で出す。"""
    match_ids = np.array([s.match_id for s in rows], dtype=np.int64)
    print("# テストの 8,626本で、3つの情報と広い情報を比べる\n")
    print(
        f"- {len(goals)}本・{len(np.unique(match_ids))}試合・"
        f"ゴール {int(goals.sum())}（{goals.mean():.4f}）\n"
    )
    print("| モデル | ログ損失 | ブライアスコア | AUC | 確率の平均 |")
    print("| --- | --- | --- | --- | --- |")
    for name, label in LABELS.items():
        values = " | ".join(
            f"{metric(goals, probs[name]):.4f}" for metric in METRICS.values()
        )
        print(f"| {label} | {values} | {probs[name].mean():.4f} |")

    print(
        f"\n差と 95% の幅（試合単位のブートストラップ {BOOTSTRAP}回。"
        "ログ損失・ブライアスコアはマイナス、AUC はプラスなら後ろがよい）\n"
    )
    print("| 比べるもの | 物差し | 差 | 95% の幅 |")
    print("| --- | --- | --- | --- |")
    for before, after in PAIRS:
        data = pair_of(rows, goals, probs[before], probs[after])
        for d in bootstrap_differences(data):
            print(
                f"| {LABELS[after]} − {LABELS[before]} | {d.name} | "
                f"{d.estimate:+.4f} | {d.low:+.4f}〜{d.high:+.4f} |"
            )

    groups = np.array([prev_group(s) for s in rows])
    counter = np.array([s.is_counter_attack for s in rows])
    print("\n直前のプレーの組・カウンターごとの、実際の率と確率の平均\n")
    header = " | ".join(LABELS.values())
    print(f"| 組 | 本数 | 実際の率 | {header} |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in LABELS) + " |")
    parts = [(g, groups == g) for g in GROUP_ORDER]
    parts += [("counter", counter), ("counter なし", ~counter)]
    for label, mask in parts:
        if not mask.any():
            continue
        means = " | ".join(
            f"{probs[name][mask].mean():.3f}" for name in LABELS
        )
        print(
            f"| {label} | {int(mask.sum())} | {goals[mask].mean():.3f} | "
            f"{means} |"
        )


def coefficients(shots: Sequence[Shot]) -> None:
    """広い情報の統計モデルを、全部のシュートで当てはめた係数を出す。"""
    data = build_dataset(shots, with_head=True, with_context=True)
    coef = fit_numpy(data.features, data.goals)
    print(
        "\n広い情報の統計モデルの係数（全部の "
        f"{len(data.goals):,}本で当てはめた対数オッズ。"
        "組は基準の pass との差）\n"
    )
    print("| 特徴 | 係数 | オッズ比 |")
    print("| --- | --- | --- |")
    for name, value in zip(("intercept", *data.names), coef, strict=True):
        print(f"| {name} | {value:+.3f} | {np.exp(value):.2f} |")


def main() -> int:
    """4つを比べて表を出す（`mise run compare-context`）。"""
    shots = load_shots_csv()
    xgs = {
        "stat": load_xg(DEFAULT_XG_PATH),
        "stat_context": load_xg(CONTEXT_XG_PATH),
        "gemini": load_xg(GEMINI_PATH),
        "gemini_context": load_xg(GEMINI_CONTEXT_PATH),
    }
    rows, goals, probs = aligned(test_shots(shots), xgs)
    report(rows, goals, probs)
    coefficients(shots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
