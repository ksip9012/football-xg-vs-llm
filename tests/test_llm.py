"""llm のテスト。API は呼ばず、偽のクライアントで確かめる。"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from google.genai import types

from football_xg_vs_llm.llm import (
    SYSTEM_PROMPT,
    Features,
    RunConfig,
    batches,
    call_batch,
    features_of,
    run,
)
from football_xg_vs_llm.models import Shot


@dataclass
class _Usage:
    prompt_token_count: int | None = 100
    candidates_token_count: int | None = 20
    thoughts_token_count: int | None = 5


@dataclass
class _Response:
    text: str | None
    usage_metadata: _Usage | None = field(default_factory=_Usage)


@dataclass
class _Models:
    """送られた JSON を覚え、ID ごとに 0.1 を返す。

    `skip` の ID は答えない。`broken` なら JSON でない答えを返す。
    """

    skip: set[int] = field(default_factory=set[int])
    broken: bool = False
    sent: list[dict[str, list[dict[str, object]]]] = field(
        default_factory=list[dict[str, list[dict[str, object]]]]
    )
    configs: list[types.GenerateContentConfig] = field(
        default_factory=list[types.GenerateContentConfig]
    )

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> _Response:
        payload = json.loads(contents)
        self.sent.append(payload)
        self.configs.append(config)
        if self.broken:
            return _Response(text="not json")
        predictions = [
            {"id": s["id"], "xg": 0.1}
            for s in payload["shots"]
            if s["id"] not in self.skip
        ]
        return _Response(text=json.dumps({"predictions": predictions}))


@dataclass
class _Client:
    models: _Models = field(default_factory=_Models)


def _features(n: int) -> list[Features]:
    return [
        Features(
            shot_id=1000 + i, distance_m=10.0, angle_deg=30.0, body="foot"
        )
        for i in range(n)
    ]


def _shot(x: int, y: int, body: str) -> Shot:
    return Shot.model_validate(
        {
            "shot_id": 1,
            "competition": "England",
            "match_id": 1,
            "match_period": "1H",
            "event_sec": 1.0,
            "team_id": 1,
            "player_id": 1,
            "shot_type": "shot",
            "x": x,
            "y": y,
            "body_part": body,
            "is_counter_attack": False,
            "prev_event": None,
            "prev_seconds_before": None,
            "is_goal": False,
        }
    )


def test_features_are_meters_degrees_and_foot_or_head() -> None:
    (f,) = features_of([_shot(90, 50, "head_or_body")])

    assert f.distance_m == pytest.approx(10.5)
    assert f.angle_deg == pytest.approx(38.4, abs=0.1)
    assert f.body == "head"


def test_batches_mix_the_order_and_keep_every_shot() -> None:
    features = _features(120)

    parts = list(batches(features, batch_size=50, seed=1))

    assert [len(p) for p in parts] == [50, 50, 20]
    ids = [f.shot_id for p in parts for f in p]
    assert sorted(ids) == [f.shot_id for f in features]
    assert ids != [f.shot_id for f in features]


def test_payload_uses_local_ids_and_no_source_ids() -> None:
    client = _Client()
    batch = _features(3)

    record = call_batch(client, batch, "LOW")

    sent = client.models.sent[0]["shots"]
    assert [s["id"] for s in sent] == [1, 2, 3]
    assert set(sent[0]) == {"id", "distance_m", "angle_deg", "body"}
    assert record["predictions"] == {"1000": 0.1, "1001": 0.1, "1002": 0.1}
    assert record["prompt_tokens"] == 100
    config = client.models.configs[0]
    assert config.system_instruction == SYSTEM_PROMPT
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_broken_answer_is_recorded_not_raised() -> None:
    client = _Client(models=_Models(broken=True))

    record = call_batch(client, _features(2), "LOW")

    assert record["predictions"] == {}
    assert "形が崩れている" in record["error"]


def test_run_retries_missing_shots_once(tmp_path: Path) -> None:
    client = _Client(models=_Models(skip={2}))
    features = _features(4)
    path = tmp_path / "run.jsonl"

    result = run(
        client, features, RunConfig("t", "LOW", batch_size=2, seed=0), path
    )

    # 2本ずつの 2回と、答えのなかった 2本（各回の ID 2）をまとめたもう 1回
    assert result.calls == 3
    assert len(client.models.sent[2]["shots"]) == 2
    # もう 1回でも ID 2 は答えないので、1本は答えなしのまま
    assert len(result.xg) == 3
    assert len(result.missing) == 1
    assert result.cost > 0


def test_run_resumes_from_the_file(tmp_path: Path) -> None:
    features = _features(4)
    path = tmp_path / "run.jsonl"
    config = RunConfig("t", "LOW", batch_size=2, seed=0)
    run(_Client(), features, config, path)

    again = _Client()
    result = run(again, features, config, path)

    assert again.models.sent == []
    assert len(result.xg) == 4
    assert len(path.read_text().splitlines()) == 2
