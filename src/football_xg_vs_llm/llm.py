"""同じシュートを LLM（Gemini）に見せて、ゴールになる確率を答えさせる。

- 渡すのは統計モデルと同じ情報だけ（距離・ゴールの見える角度・頭か足か）。
  選手の名前・チーム・試合の日付は渡さない（有名な試合の結果を LLM が
  覚えていて、答えが漏れるおそれがあるため）
- 順番を混ぜて `batch_size` 本ずつ送り、`shot_id` は送るときだけ 1 からの
  連番に置き換える（同じ試合のシュートや元データの ID の並びから、答えが
  漏れないように）
- 答えは決まった形の JSON で受け取り、pydantic で確かめる
- 1回ごとの結果を JSONL に残し、終わった回は飛ばして続きから始める
  （同じシュートに 2回料金を払わない）。答えのなかったシュートは、
  最後にもう一度だけ送る

決めごとは decisions/0006。
"""

import json
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.xg import distance_and_angle

MODEL = "gemini-3.8-flash"
# 100万トークンあたりの単価（ドル）。Step2 で 2026-11-03 に確かめた値で、
# 2026-12-31 までの価格。思考のトークンは出力として数える
PRICE_INPUT_PER_M = 0.75
PRICE_OUTPUT_PER_M = 3.75

SYSTEM_PROMPT = """\
You are a football (soccer) analyst. You will receive a list of shots from \
professional men's football matches. Penalties and direct free kicks are \
excluded. For each shot, estimate the probability that it results in a goal \
(its expected goals, xG), as a number between 0 and 1.

Each shot has:
- id: an identifier for this list only
- distance_m: distance from where the shot was taken to the centre of the \
goal, in metres
- angle_deg: the angle, in degrees, between the lines from the shot location \
to the two goal posts (a larger angle means more of the goal is visible; \
180 on the goal line between the posts)
- body: "foot" or "head" (head also covers other body parts)

Return one prediction for every id, and nothing else."""


class ShotInput(BaseModel):
    """LLM に渡すシュート 1本。"""

    model_config = ConfigDict(frozen=True)

    id: int
    distance_m: float
    angle_deg: float
    body: Literal["foot", "head"]


class Prediction(BaseModel):
    """LLM の答え 1本。"""

    id: int
    xg: float = Field(ge=0, le=1)


class Predictions(BaseModel):
    """LLM の答え（1回の呼び出しぶん）。"""

    predictions: list[Prediction]


@dataclass(frozen=True)
class Features:
    """LLM に渡す前の、シュート 1本の情報（ID は元データのもの）。"""

    shot_id: int
    distance_m: float
    angle_deg: float
    body: Literal["foot", "head"]


def features_of(shots: Sequence[Shot]) -> list[Features]:
    """シュートの表の行から、LLM に渡す情報を作る（小数 1桁）。"""
    x = np.array([s.x for s in shots], dtype=np.float64)
    y = np.array([s.y for s in shots], dtype=np.float64)
    distance, angle = distance_and_angle(x, y)
    return [
        Features(
            shot_id=s.shot_id,
            distance_m=round(float(d), 1),
            angle_deg=round(float(np.degrees(a)), 1),
            body="head" if s.body_part == "head_or_body" else "foot",
        )
        for s, d, a in zip(shots, distance, angle, strict=True)
    ]


class _Usage(Protocol):
    prompt_token_count: int | None
    candidates_token_count: int | None
    thoughts_token_count: int | None


class _Response(Protocol):
    @property
    def text(self) -> str | None: ...

    @property
    def usage_metadata(self) -> _Usage | None: ...


class _Models(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> _Response: ...


class Client(Protocol):
    """`google.genai.Client` のうち、使う部分の形（テストでは偽物を渡す）。"""

    @property
    def models(self) -> _Models: ...


@dataclass(frozen=True)
class RunConfig:
    """1回の試しの設定。

    Attributes:
        name: 試しの名前（結果のファイルの名前になる）。
        thinking_level: 考える量（`MINIMAL`・`LOW`・`MEDIUM`・`HIGH`）。
        batch_size: 1回の呼び出しに入れる本数。
        seed: 順番を混ぜる乱数の種。
    """

    name: str
    thinking_level: str
    batch_size: int = 50
    seed: int = 2026


def _config(level: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=Predictions,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(level)
        ),
        temperature=1.0,
    )


def batches(
    features: Sequence[Features], *, batch_size: int, seed: int
) -> Iterator[list[Features]]:
    """順番を混ぜて、`batch_size` 本ずつに分ける。"""
    shuffled = list(features)
    random.Random(seed).shuffle(shuffled)
    for i in range(0, len(shuffled), batch_size):
        yield shuffled[i : i + batch_size]


def _payload(batch: Sequence[Features]) -> str:
    """1回ぶんのシュートを、連番の ID を付けた JSON にする。"""
    items = [
        ShotInput(
            id=i,
            distance_m=f.distance_m,
            angle_deg=f.angle_deg,
            body=f.body,
        ).model_dump()
        for i, f in enumerate(batch, start=1)
    ]
    return json.dumps({"shots": items}, ensure_ascii=False)


def call_batch(
    client: Client, batch: Sequence[Features], level: str
) -> dict[str, Any]:
    """1回呼び、結果を記録の形（JSONL の 1行）で返す。

    答えの形が崩れていても止まらず、`error` に理由を書いて返す。
    """
    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL, contents=_payload(batch), config=_config(level)
    )
    seconds = time.perf_counter() - start
    usage = response.usage_metadata
    record: dict[str, Any] = {
        "shot_ids": [f.shot_id for f in batch],
        "raw": response.text,
        "predictions": {},
        "prompt_tokens": (usage.prompt_token_count if usage else None) or 0,
        "output_tokens": (usage.candidates_token_count if usage else None)
        or 0,
        "thought_tokens": (usage.thoughts_token_count if usage else None) or 0,
        "seconds": round(seconds, 3),
        "error": None,
    }
    try:
        parsed = Predictions.model_validate_json(response.text or "")
    except ValidationError as e:
        record["error"] = f"形が崩れている: {e.error_count()}件"
        return record
    by_local = {f_id: f.shot_id for f_id, f in enumerate(batch, start=1)}
    record["predictions"] = {
        str(by_local[p.id]): p.xg
        for p in parsed.predictions
        if p.id in by_local
    }
    return record


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[record["batch"]] = record
    return records


def _predictions(record: dict[str, Any]) -> dict[int, float]:
    """記録の 1行の答え（`shot_id` の文字列 → xG）を、数の対応にする。"""
    raw = cast("dict[str, float]", record["predictions"])
    return {int(k): float(v) for k, v in raw.items()}


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class RunResult:
    """1回の試しの結果。"""

    xg: dict[int, float]
    missing: list[int]
    calls: int
    prompt_tokens: int
    output_tokens: int
    thought_tokens: int
    seconds: float

    @property
    def cost(self) -> float:
        """費用（ドル）。思考のトークンも出力として数える。"""
        out = self.output_tokens + self.thought_tokens
        return (
            self.prompt_tokens * PRICE_INPUT_PER_M + out * PRICE_OUTPUT_PER_M
        ) / 1_000_000


def run(
    client: Client,
    features: Sequence[Features],
    config: RunConfig,
    path: Path,
) -> RunResult:
    """試しを 1回動かす。`path` の JSONL に、終わった回があれば飛ばす。

    最後に、答えのなかったシュートだけをまとめて、もう一度だけ送る。
    """
    records = _load_records(path)
    planned = list(
        batches(features, batch_size=config.batch_size, seed=config.seed)
    )
    for i, batch in enumerate(planned):
        key = f"{i:04d}"
        if key not in records:
            record: dict[str, Any] = {
                "batch": key,
                **call_batch(client, batch, config.thinking_level),
            }
            _append(path, record)
            records[key] = record

    answered: dict[int, float] = {}
    for r in records.values():
        answered.update(_predictions(r))
    missing = [f for f in features if f.shot_id not in answered]
    if missing and "retry" not in records:
        retry: dict[str, Any] = {
            "batch": "retry",
            **call_batch(client, missing, config.thinking_level),
        }
        _append(path, retry)
        records["retry"] = retry
        answered.update(_predictions(retry))

    return RunResult(
        xg={
            f.shot_id: answered[f.shot_id]
            for f in features
            if f.shot_id in answered
        },
        missing=[f.shot_id for f in features if f.shot_id not in answered],
        calls=len(records),
        prompt_tokens=sum(r["prompt_tokens"] for r in records.values()),
        output_tokens=sum(r["output_tokens"] for r in records.values()),
        thought_tokens=sum(r["thought_tokens"] for r in records.values()),
        seconds=sum(r["seconds"] for r in records.values()),
    )
