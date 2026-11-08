"""shots のテスト。

速いテストは作り物のイベントで、slow は元データで確かめる。
"""

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from football_xg_vs_llm.shots import (
    COLUMNS,
    competition_of,
    extract_shots,
    read_shots,
    write_shots,
)


def _event(
    event_id: int,
    sub_event_id: int | str,
    *,
    id_: int,
    team: int = 1,
    sec: float = 0.0,
    period: str = "1H",
    match: int = 100,
    tags: tuple[int, ...] = (),
    name: str = "Simple pass",
    x: int = 80,
    y: int = 50,
) -> dict[str, Any]:
    return {
        "id": id_,
        "eventId": event_id,
        "subEventId": sub_event_id,
        "eventName": "Pass",
        "subEventName": name,
        "matchId": match,
        "matchPeriod": period,
        "eventSec": sec,
        "teamId": team,
        "playerId": 7,
        "positions": [{"x": x, "y": y}, {"x": 100, "y": 100}],
        "tags": [{"id": t} for t in tags],
    }


def _shot(**kwargs: Any) -> dict[str, Any]:
    return _event(10, 100, name="Shot", **kwargs)


def test_extracts_three_kinds_and_skips_other_events() -> None:
    events = [
        _event(8, 85, id_=1),
        _shot(id_=2, tags=(402, 101)),
        _event(3, 33, id_=3, name="Free kick shot", tags=(401,)),
        _event(3, 35, id_=4, name="Penalty", tags=(402, 101)),
        _event(3, 31, id_=5, name="Free kick cross"),
    ]

    shots = list(extract_shots(events, "England"))

    assert [(s.shot_id, s.shot_type) for s in shots] == [
        (2, "shot"),
        (3, "free_kick"),
        (4, "penalty"),
    ]
    assert [s.is_goal for s in shots] == [True, False, True]
    assert [s.body_part for s in shots] == [
        "right_foot",
        "left_foot",
        "right_foot",
    ]


def test_previous_event_skips_the_other_team() -> None:
    events = [
        _event(8, 80, id_=1, team=1, sec=10.0, name="Cross"),
        _event(1, 12, id_=2, team=2, sec=11.0, name="Ground defending duel"),
        _shot(id_=3, team=1, sec=11.5),
    ]

    (shot,) = extract_shots(events, "England")

    assert shot.prev_event == "Cross"
    assert shot.prev_seconds_before == 1.5


def test_previous_event_resets_at_a_new_period_or_match() -> None:
    events = [
        _event(8, 85, id_=1, team=1, sec=2700.0, period="1H"),
        _shot(id_=2, team=1, sec=3.0, period="2H"),
        _event(8, 85, id_=3, team=1, sec=5.0, period="2H", match=100),
        _shot(id_=4, team=1, sec=1.0, period="2H", match=101),
    ]

    shots = list(extract_shots(events, "England"))

    assert [(s.prev_event, s.prev_seconds_before) for s in shots] == [
        ("", None),
        ("", None),
    ]


def test_rebound_shot_has_shot_as_previous_event() -> None:
    events = [_shot(id_=1, sec=20.0), _shot(id_=2, sec=21.2)]

    shots = list(extract_shots(events, "England"))

    assert shots[1].prev_event == "Shot"
    assert shots[1].prev_seconds_before == 1.2


def test_keeps_only_the_start_position_and_no_post_shot_information() -> None:
    # 201 Opportunity・1801 Accurate・2101 Blocked・1201（ゴールの位置）は
    # 表に入れない
    events = [_shot(id_=1, x=88, y=40, tags=(201, 1801, 2101, 1201, 1901))]

    (shot,) = extract_shots(events, "England")

    assert (shot.x, shot.y) == (88, 40)
    assert shot.is_counter_attack is True
    for word in ("opportunity", "accurate", "blocked", "end_x", "end_y"):
        assert not any(word in column for column in COLUMNS)


def test_competition_name_comes_from_the_file_name() -> None:
    assert competition_of("events_World_Cup.json") == "World_Cup"


def test_read_and_write_round_trip(tmp_path: Path) -> None:
    events_path = tmp_path / "events.zip"
    with zipfile.ZipFile(events_path, "w") as z:
        spain = [_shot(id_=9, tags=(101,))]
        z.writestr("events_Spain.json", json.dumps(spain))
        z.writestr("events_England.json", json.dumps([_shot(id_=8)]))
    output = tmp_path / "out" / "shots.csv"

    write_shots(read_shots(events_path), output)

    with output.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == COLUMNS
    assert [(r["competition"], r["shot_id"], r["is_goal"]) for r in rows] == [
        ("England", "8", "0"),
        ("Spain", "9", "1"),
    ]
    assert rows[0]["prev_seconds_before"] == ""


@pytest.mark.slow
def test_counts_match_the_source_data() -> None:
    # 本数とゴール数は notes/wyscout-data-columns.md で数えた値
    events_path = Path("data/raw/events.zip")
    if not events_path.exists():
        pytest.skip("元データがない（mise run fetch で落とす）")

    shots = read_shots(events_path)

    assert Counter(s.shot_type for s in shots) == {
        "shot": 43078,
        "free_kick": 2209,
        "penalty": 658,
    }
    assert Counter(s.shot_type for s in shots if s.is_goal) == {
        "shot": 4492,
        "free_kick": 137,
        "penalty": 477,
    }
    assert all(s.body_part for s in shots)
