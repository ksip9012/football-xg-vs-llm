"""同じ 2,000本ほどで、Claude・Gemini・統計モデルを比べる（decisions/0009）。

- 6つ: 統計モデル・Gemini・Claude の、それぞれ 3つの情報と広い情報
- 高さだけを直したもの（decisions/0008 と同じ。この 2,000本の中で試合単位の
  5分割で学び、別の試合に当てる）も、Gemini と Claude で並べる
- 差の 95% の幅（試合単位のブートストラップ）: Claude − Gemini、Claude −
  統計モデル、Claude の広い − 3つ
- Claude と Gemini の確率の相関

`mise run compare-claude`（先に `mise run llm-claude` と
`mise run llm-claude-context`）。
"""

import sys
from collections.abc import Sequence

import numpy as np

from football_xg_vs_llm.compare_context import aligned, pair_of
from football_xg_vs_llm.compare_llm import (
    BOOTSTRAP,
    METRICS,
    bootstrap_differences,
)
from football_xg_vs_llm.cv import CONTEXT_XG_PATH, DEFAULT_XG_PATH, load_xg
from football_xg_vs_llm.llm_claude import XG_PATHS, sample_shots
from football_xg_vs_llm.llm_test import CONTEXT_XG_PATH as GEMINI_CONTEXT_PATH
from football_xg_vs_llm.llm_test import XG_PATH as GEMINI_PATH
from football_xg_vs_llm.llm_test import test_shots
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.recalibrate import recalibrate
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import GROUP_ORDER, Floats, prev_group

LABELS = {
    "stat": "統計モデル（3つ）",
    "gemini": "Gemini（3つ）",
    "claude": "Claude（3つ）",
    "stat_context": "統計モデル（広い）",
    "gemini_context": "Gemini（広い）",
    "claude_context": "Claude（広い）",
}
# 差を出す組（後ろ − 前）
PAIRS = (
    ("gemini", "claude"),
    ("stat", "claude"),
    ("gemini_context", "claude_context"),
    ("stat_context", "claude_context"),
    ("claude", "claude_context"),
    ("gemini_shift", "claude_shift"),
    ("stat_context", "claude_context_shift"),
)


def report(
    rows: Sequence[Shot], goals: Floats, probs: dict[str, Floats]
) -> None:
    """物差し・差の幅・相関・組ごとの表を Markdown で出す。"""
    match_ids = np.array([s.match_id for s in rows], dtype=np.int64)
    table = dict(probs)
    labels = dict(LABELS)
    for name in ("gemini", "claude", "gemini_context", "claude_context"):
        table[f"{name}_shift"] = recalibrate(
            probs[name], goals, match_ids
        ).shift
        labels[f"{name}_shift"] = f"{LABELS[name]}・高さを直す"

    print("# 同じシュートで、Claude・Gemini・統計モデルを比べる\n")
    print(
        f"- {len(goals)}本・{len(np.unique(match_ids))}試合・"
        f"ゴール {int(goals.sum())}（{goals.mean():.4f}）\n"
    )
    print("| モデル | ログ損失 | ブライアスコア | AUC | 確率の平均 |")
    print("| --- | --- | --- | --- | --- |")
    for name, label in labels.items():
        values = " | ".join(
            f"{metric(goals, table[name]):.4f}" for metric in METRICS.values()
        )
        print(f"| {label} | {values} | {table[name].mean():.4f} |")

    print("\n確率の相関\n")
    print("| | 3つの情報 | 広い情報 |")
    print("| --- | --- | --- |")
    for a, b in (("claude", "gemini"), ("claude", "stat"), ("gemini", "stat")):
        narrow = np.corrcoef(table[a], table[b])[0, 1]
        wide = np.corrcoef(table[f"{a}_context"], table[f"{b}_context"])[0, 1]
        print(f"| {a} と {b} | {narrow:.3f} | {wide:.3f} |")

    print(
        f"\n差と 95% の幅（試合単位のブートストラップ {BOOTSTRAP}回。"
        "ログ損失・ブライアスコアはマイナス、AUC はプラスなら後ろがよい）\n"
    )
    print("| 比べるもの | 物差し | 差 | 95% の幅 |")
    print("| --- | --- | --- | --- |")
    for before, after in PAIRS:
        data = pair_of(rows, goals, table[before], table[after])
        for d in bootstrap_differences(data):
            print(
                f"| {labels[after]} − {labels[before]} | {d.name} | "
                f"{d.estimate:+.4f} | {d.low:+.4f}〜{d.high:+.4f} |"
            )

    groups = np.array([prev_group(s) for s in rows])
    print("\n組ごとの、実際の率と確率の平均（広い情報）\n")
    print("| 組 | 本数 | 実際の率 | 統計モデル | Gemini | Claude |")
    print("| --- | --- | --- | --- | --- | --- |")
    for g in GROUP_ORDER:
        mask = groups == g
        if not mask.any():
            continue
        means = " | ".join(
            f"{table[n][mask].mean():.3f}"
            for n in ("stat_context", "gemini_context", "claude_context")
        )
        print(
            f"| {g} | {int(mask.sum())} | {goals[mask].mean():.3f} | {means} |"
        )


def main() -> int:
    """比べて表を出す（`mise run compare-claude`）。"""
    xgs = {
        "stat": load_xg(DEFAULT_XG_PATH),
        "gemini": load_xg(GEMINI_PATH),
        "claude": load_xg(XG_PATHS[False]),
        "stat_context": load_xg(CONTEXT_XG_PATH),
        "gemini_context": load_xg(GEMINI_CONTEXT_PATH),
        "claude_context": load_xg(XG_PATHS[True]),
    }
    shots = sample_shots(test_shots(load_shots_csv()))
    rows, goals, probs = aligned(shots, xgs)
    report(rows, goals, probs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
