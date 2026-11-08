"""shots のテスト。

速いテストは作り物のイベントで、slow は元データで確かめる。
"""

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from football_xg_vs_llm.models import COLUMNS, Event
from football_xg_vs_llm.shots import (
    DEFAULT_OUTPUT_PATH,
    competition_of,
    extract_shots,
    load_shots_csv,
    read_events,
    read_shots,
    write_shots,
)

BODY_PARTS = {401, 402, 403}


def _raw(
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
    """元データの形（camelCase の辞書）のイベント。"""
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


def _raw_shot(*, tags: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Any]:
    """普通のシュート。体の部位のタグがなければ右足を付ける。"""
    if not BODY_PARTS & set(tags):
        tags = (*tags, 402)
    return _raw(10, 100, name="Shot", tags=tags, **kwargs)


def _event(*args: Any, **kwargs: Any) -> Event:
    return Event.model_validate(_raw(*args, **kwargs))


def _shot(**kwargs: Any) -> Event:
    return Event.model_validate(_raw_shot(**kwargs))


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
        (None, None),
        (None, None),
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


def test_event_model_evens_out_the_source_types() -> None:
    raw = _raw(8, "", id_=1)
    raw["teamId"] = "1609"
    raw["eventSec"] = 3

    (event,) = read_events(json.dumps([raw]).encode())

    assert event.sub_event_id is None
    assert event.team_id == 1609
    assert event.event_sec == 3.0


def test_event_model_rejects_a_missing_field() -> None:
    raw = _raw(8, 85, id_=1)
    del raw["teamId"]

    with pytest.raises(ValidationError):
        read_events(json.dumps([raw]).encode())


def test_shot_without_a_body_part_tag_is_rejected() -> None:
    events = [_event(10, 100, id_=1, name="Shot")]

    with pytest.raises(ValidationError):
        list(extract_shots(events, "England"))


def test_shot_outside_the_pitch_is_rejected() -> None:
    events = [_shot(id_=1, x=101)]

    with pytest.raises(ValidationError):
        list(extract_shots(events, "England"))


def test_unknown_competition_is_rejected() -> None:
    with pytest.raises(ValidationError):
        list(extract_shots([_shot(id_=1)], "Netherlands"))


def test_read_write_and_load_round_trip(tmp_path: Path) -> None:
    events_path = tmp_path / "events.zip"
    with zipfile.ZipFile(events_path, "w") as z:
        spain = [_raw_shot(id_=9, tags=(101,))]
        z.writestr("events_Spain.json", json.dumps(spain))
        z.writestr("events_England.json", json.dumps([_raw_shot(id_=8)]))
    output = tmp_path / "out" / "shots.csv"
    shots = read_shots(events_path)

    write_shots(shots, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert lines[1].endswith(",shot,80,50,right_foot,0,,,0")
    assert load_shots_csv(output) == shots


def test_load_rejects_a_broken_row(tmp_path: Path) -> None:
    path = tmp_path / "shots.csv"
    header = ",".join(COLUMNS)
    row = "1,England,100,1H,3.0,1,7,shot,80,50,right_foot,0,Cross,,0"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    # prev_event があるのに prev_seconds_before がない
    with pytest.raises(ValidationError):
        load_shots_csv(path)


def test_load_rejects_unexpected_columns(tmp_path: Path) -> None:
    path = tmp_path / "shots.csv"
    path.write_text("shot_id,is_goal\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="列が違う"):
        load_shots_csv(path)


def test_repository_table_follows_the_model() -> None:
    shots = load_shots_csv(DEFAULT_OUTPUT_PATH)

    assert len(shots) == 45945


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
