"""同じシュートを Claude（Opus 5.5）に見せて、ゴールになる確率を答えさせる。

Gemini の回（`llm`）と同じ説明・同じ JSON・同じ 50本ずつで送り、答えは決まった
形の JSON（構造化出力）で受け取る。渡す情報も同じで、`--context` を付けると
直前のプレーの組とカウンターも渡す。

- シュート: テストの中から、試合単位で 2,000本ほどを選ぶ（`sample_shots`）
- 考える量: Opus 5.5 は考えるのを止められないので、いちばん低い `low` にする
- 断られた回（`refusal`）は答えなしとして記録し、最後にもう一度だけ送る。
  別のモデルへの切り替え（フォールバック）は入れない。別のモデルの答えが
  混ざると、比べが崩れるため
- 答えは 1回ごとに `data/llm_runs/claude/` の JSONL に残す（続きから
  始められる）

決めごとは decisions/0009。`mise run llm-claude`・`mise run llm-claude-context`
（料金がかかる。API キーは手元の `mise.local.toml` の `ANTHROPIC_API_KEY`）。
"""

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anthropic
import numpy as np

from football_xg_vs_llm.cv import write_xg
from football_xg_vs_llm.llm import (
    Features,
    Predictions,
    RunConfig,
    features_of,
    make_record,
    payload_json,
    run,
    system_prompt,
)
from football_xg_vs_llm.llm_test import test_shots
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv

MODEL = "claude-opus-5-5"
# 100万トークンあたりの単価（ドル）。Anthropic の価格表（2026-10-06 時点の
# 写し）。考えたトークンも出力として数える（`usage.output_tokens` に入る）
PRICE_INPUT_PER_M = 4.0
PRICE_OUTPUT_PER_M = 20.0
EFFORT = "low"
MAX_TOKENS = 16000
SAMPLE_SIZE = 2000
SEED = 2026

RUN_DIR = Path("data/llm_runs/claude")
CONFIGS = {
    False: RunConfig(name="claude_low", thinking_level=EFFORT, batch_size=50),
    True: RunConfig(
        name="claude_low_context",
        thinking_level=EFFORT,
        batch_size=50,
        context=True,
    ),
}
RUN_PATHS = {
    False: RUN_DIR / "opus_low.jsonl",
    True: RUN_DIR / "opus_low_context.jsonl",
}
XG_PATHS = {
    False: Path("data/xg_claude.csv"),
    True: Path("data/xg_claude_context.csv"),
}


def sample_shots(
    shots: Sequence[Shot], *, size: int = SAMPLE_SIZE, seed: int = SEED
) -> list[Shot]:
    """試合を順番を混ぜて選び、シュートが `size` 本以上になったところで止める。

    同じ試合のシュートはまとめて入る（差の幅を試合単位で出すため）。
    """
    matches = sorted({s.match_id for s in shots})
    order = np.random.default_rng(seed).permutation(len(matches))
    count_of: dict[int, int] = {}
    for s in shots:
        count_of[s.match_id] = count_of.get(s.match_id, 0) + 1
    picked: set[int] = set()
    total = 0
    for i in order:
        if total >= size:
            break
        match = matches[int(i)]
        picked.add(match)
        total += count_of[match]
    return [s for s in shots if s.match_id in picked]


def call_claude(
    client: anthropic.Anthropic,
    batch: Sequence[Features],
    *,
    context: bool,
) -> dict[str, Any]:
    """Claude を 1回呼び、記録の形（`make_record`）で返す。"""
    start = time.perf_counter()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt(context=context),
        messages=[{"role": "user", "content": payload_json(batch)}],
        output_config={"effort": EFFORT},
        output_format=Predictions,
    )
    seconds = time.perf_counter() - start
    error = None
    if response.stop_reason == "refusal":
        category = (
            response.stop_details.category if response.stop_details else None
        )
        error = f"断られた: {category}"
    elif response.stop_reason == "max_tokens":
        error = "max_tokens で切れた"
    text = next((b.text for b in response.content if b.type == "text"), None)
    return make_record(
        batch,
        text,
        prompt_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        thought_tokens=0,
        seconds=seconds,
        error=error,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """選んだシュートを Claude に答えさせ、確率の CSV を書く。"""
    args = sys.argv[1:] if argv is None else list(argv)
    context = "--context" in args
    shots = sample_shots(test_shots(load_shots_csv()))
    features = features_of(shots, with_context=context)
    client = anthropic.Anthropic(max_retries=5)

    def caller(batch: Sequence[Features]) -> dict[str, Any]:
        return call_claude(client, batch, context=context)

    result = run(
        None,
        features,
        CONFIGS[context],
        RUN_PATHS[context],
        caller=caller,
        prices=(PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M),
    )
    write_xg(result.xg, XG_PATHS[context])
    print(
        f"書いた: {XG_PATHS[context]}（{len(result.xg)}本・"
        f"{len({s.match_id for s in shots})}試合、"
        f"答えなし {len(result.missing)}本）"
    )
    print(
        f"呼び出し {result.calls}回・入力 {result.prompt_tokens}・"
        f"出力（考えた分を含む） {result.output_tokens}・"
        f"費用 ${result.cost:.4f}・時間 {result.seconds:.0f}秒"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
