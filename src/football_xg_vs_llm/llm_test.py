"""テストのシュート（最初の xG と同じ分け方、8,626本）を Gemini に答えさせる。

500本の試し（notes/llm-trial.md）で決めた、考える量 LOW・1回に 50本で動かす。
答えは `data/llm_runs/test/low.jsonl` に 1回ごとに残し（続きから始められる）、
確率は `shot_id, xg` の CSV（`data/xg_gemini.csv`）に置く。この形なので、
decisions/0005 の集計と図に、そのまま渡せる。

`mise run llm-test`（料金がかかる。答えのファイルがあれば、その回は呼ばない）。
"""

import sys
from collections.abc import Sequence
from pathlib import Path

from google import genai

from football_xg_vs_llm.cv import write_xg
from football_xg_vs_llm.llm import RunConfig, features_of, run
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import build_dataset, is_modeled, split_by_match

RUN_PATH = Path("data/llm_runs/test/low.jsonl")
XG_PATH = Path("data/xg_gemini.csv")
CONFIG = RunConfig(name="test_low", thinking_level="LOW", batch_size=50)


def test_shots(shots: Sequence[Shot]) -> list[Shot]:
    """最初の xG と同じテストのシュート（試合単位の 2割）。"""
    data = build_dataset(shots, with_head=True)
    _, test = split_by_match(data.match_ids)
    test_ids = {int(s) for s in data.shot_ids[test]}
    return [s for s in shots if is_modeled(s) and s.shot_id in test_ids]


def main() -> int:
    """テストのシュートを Gemini に答えさせ、確率の CSV を書く。"""
    shots = test_shots(load_shots_csv())
    result = run(genai.Client(), features_of(shots), CONFIG, RUN_PATH)
    write_xg(result.xg, XG_PATH)
    print(
        f"書いた: {XG_PATH}（{len(result.xg)}本、"
        f"答えなし {len(result.missing)}本）"
    )
    print(
        f"呼び出し {result.calls}回・入力 {result.prompt_tokens}・"
        f"出力 {result.output_tokens}・思考 {result.thought_tokens}・"
        f"費用 ${result.cost:.4f}・時間 {result.seconds:.0f}秒"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
