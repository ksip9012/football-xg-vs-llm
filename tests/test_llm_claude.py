"""llm_claude のテスト。API は呼ばず、偽のクライアントで確かめる。"""

import json
from dataclasses import dataclass, field
from typing import Any, cast

import anthropic

from football_xg_vs_llm.llm import Features, system_prompt
from football_xg_vs_llm.llm_claude import (
    EFFORT,
    MODEL,
    call_claude,
    sample_shots,
)
from football_xg_vs_llm.models import Shot


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 300
    output_tokens: int = 80


@dataclass
class _StopDetails:
    category: str | None


@dataclass
class _Message:
    content: list[_Block]
    stop_reason: str = "end_turn"
    stop_details: _StopDetails | None = None
    usage: _Usage = field(default_factory=_Usage)


@dataclass
class _Messages:
    """送られた引数を覚え、ID ごとに 0.2 を返す（`refuse` なら断る）。"""

    refuse: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def parse(self, **kwargs: Any) -> _Message:
        self.calls.append(kwargs)
        if self.refuse:
            return _Message(
                content=[],
                stop_reason="refusal",
                stop_details=_StopDetails(category="cyber"),
            )
        shots = json.loads(kwargs["messages"][0]["content"])["shots"]
        answer = {"predictions": [{"id": s["id"], "xg": 0.2} for s in shots]}
        return _Message(content=[_Block(text=json.dumps(answer))])


@dataclass
class _Client:
    messages: _Messages = field(default_factory=_Messages)


def _batch() -> list[Features]:
    return [
        Features(shot_id=500 + i, distance_m=12.0, angle_deg=25.0, body="foot")
        for i in range(3)
    ]


def _shot(shot_id: int, match: int) -> Shot:
    return Shot.model_validate(
        {
            "shot_id": shot_id,
            "competition": "England",
            "match_id": match,
            "match_period": "1H",
            "event_sec": 1.0,
            "team_id": 1,
            "player_id": 1,
            "shot_type": "shot",
            "x": 90,
            "y": 50,
            "body_part": "right_foot",
            "is_counter_attack": False,
            "prev_event": None,
            "prev_seconds_before": None,
            "is_goal": False,
        }
    )


def test_call_claude_sends_the_same_prompt_and_maps_ids_back() -> None:
    client = _Client()

    record = call_claude(
        cast("anthropic.Anthropic", client), _batch(), context=False
    )

    (sent,) = client.messages.calls
    assert sent["model"] == MODEL
    assert sent["system"] == system_prompt(context=False)
    assert sent["output_config"] == {"effort": EFFORT}
    assert record["predictions"] == {"500": 0.2, "501": 0.2, "502": 0.2}
    assert record["error"] is None
    assert record["prompt_tokens"] == 300


def test_refusal_is_recorded_as_no_answer() -> None:
    client = _Client(messages=_Messages(refuse=True))

    record = call_claude(
        cast("anthropic.Anthropic", client), _batch(), context=True
    )

    assert record["predictions"] == {}
    assert record["error"] == "断られた: cyber"


def test_sample_shots_takes_whole_matches_until_the_size() -> None:
    shots = [_shot(m * 10 + i, m) for m in range(10) for i in range(4)]

    picked = sample_shots(shots, size=10, seed=1)

    matches = {s.match_id for s in picked}
    # 1試合 4本なので、10本以上になるのは 3試合（12本）
    assert len(matches) == 3
    assert len(picked) == 12
    assert sample_shots(shots, size=10, seed=1) == picked
