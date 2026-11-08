"""Gemini で 500本を試し、考える量と揺れを見て、統計モデルと並べる。

- シュートは、最初の xG と同じテスト（試合単位の 2割）から 500本を選ぶ
- 考える量（LOW・MEDIUM・HIGH）を変えた 3回と、LOW を順番の混ぜ方だけ
  変えてもう 1回（2回聞いたときの揺れを見るため）動かす
- 統計モデルは、交差検証の xG（`data/xg_stat.csv`）の同じ 500本

結果のファイルは `data/llm_runs/trial/<名前>.jsonl`（続きから始められる）。
決めごとは decisions/0006（`mise run llm-trial`）。
"""

import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from google import genai

from football_xg_vs_llm.cv import DEFAULT_XG_PATH, load_xg
from football_xg_vs_llm.llm import RunConfig, RunResult, features_of, run
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import (
    Floats,
    brier_score,
    build_dataset,
    is_modeled,
    log_loss,
    roc_auc,
    split_by_match,
)

TRIAL_DIR = Path("data/llm_runs/trial")
TRIAL_SIZE = 500
SEED = 2026

# gemini-3.8-flash は MINIMAL を受け付けない（2026-11-08 に 400 が返った）
RUNS = [
    RunConfig(name="low", thinking_level="LOW"),
    RunConfig(name="medium", thinking_level="MEDIUM"),
    RunConfig(name="high", thinking_level="HIGH"),
    RunConfig(name="low_again", thinking_level="LOW", seed=SEED + 1),
]


def trial_shots(shots: Sequence[Shot]) -> list[Shot]:
    """テストのシュートから `TRIAL_SIZE` 本を選ぶ。

    テストは、最初の xG と同じ分け方（試合単位の 2割）。
    """
    data = build_dataset(shots, with_head=True)
    _, test = split_by_match(data.match_ids)
    test_ids = {int(s) for s in data.shot_ids[test]}
    pool = [s for s in shots if is_modeled(s) and s.shot_id in test_ids]
    return sorted(
        random.Random(SEED).sample(pool, TRIAL_SIZE), key=lambda s: s.shot_id
    )


def _row(name: str, goals: Floats, probs: Floats) -> str:
    return (
        f"| {name} | {log_loss(goals, probs):.4f} | "
        f"{brier_score(goals, probs):.4f} | {roc_auc(goals, probs):.4f} | "
        f"{probs.mean():.3f} |"
    )


def report(
    shots: Sequence[Shot],
    stat: dict[int, float],
    results: dict[str, RunResult],
) -> None:
    """結果を Markdown の表で出す（答えのそろったシュートだけで比べる）。"""
    ids = [
        s.shot_id
        for s in shots
        if all(s.shot_id in r.xg for r in results.values())
    ]
    goal_of = {s.shot_id: float(s.is_goal) for s in shots}
    goals = np.array([goal_of[i] for i in ids])
    print("# Gemini の 500本の試し\n")
    print(
        f"- シュート {len(shots)}本（ゴール {int(sum(goal_of.values()))}）。"
    )
    print(f"- すべての試しで答えのそろった {len(ids)}本で比べる\n")
    print("| モデル | ログ損失 | ブライアスコア | AUC | 確率の平均 |")
    print("| --- | --- | --- | --- | --- |")
    print(_row("平均の率だけ", goals, np.full(len(ids), goals.mean())))
    stat_probs = np.array([stat[i] for i in ids])
    print(_row("統計モデル（交差検証の xG）", goals, stat_probs))
    for name, r in results.items():
        print(_row(f"Gemini {name}", goals, np.array([r.xg[i] for i in ids])))

    print(
        "\n| 試し | 呼び出し | 答えなし | 入力 | 出力 | 思考 | 費用（$） "
        "| 時間（秒） | 統計モデルとの相関 |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        probs = np.array([r.xg[i] for i in ids])
        corr = float(np.corrcoef(probs, stat_probs)[0, 1])
        print(
            f"| {name} | {r.calls} | {len(r.missing)} | {r.prompt_tokens} | "
            f"{r.output_tokens} | {r.thought_tokens} | {r.cost:.4f} | "
            f"{r.seconds:.0f} | {corr:.3f} |"
        )

    if "low" in results and "low_again" in results:
        a = np.array([results["low"].xg[i] for i in ids])
        b = np.array([results["low_again"].xg[i] for i in ids])
        print(
            f"\n2回聞いたときの揺れ（LOW、順番の混ぜ方だけ変えた）: 相関 "
            f"{float(np.corrcoef(a, b)[0, 1]):.3f}・差の絶対値の平均 "
            f"{float(np.mean(np.abs(a - b))):.4f}・最大 "
            f"{float(np.max(np.abs(a - b))):.4f}"
        )


def main() -> int:
    """500本の試しを動かす（`mise run llm-trial`）。"""
    shots = trial_shots(load_shots_csv())
    features = features_of(shots)
    client = genai.Client()
    results = {
        config.name: run(
            client, features, config, TRIAL_DIR / f"{config.name}.jsonl"
        )
        for config in RUNS
    }
    report(shots, load_xg(DEFAULT_XG_PATH), results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
