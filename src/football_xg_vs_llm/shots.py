"""イベント（`events.zip`）からシュートを抜き出し、1行 1本の表にする。

表の形の決めごとは decisions/0002、列の説明は notes/shots-table.md。
シュートのあとに決まる情報（ゴールのどこに飛んだか、枠内か、ブロックか、
終わりの位置、タグ 201 Opportunity）は、答えが漏れるので表に入れない。
"""

import csv
import json
import logging
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_PATH = Path("data/raw/events.zip")
DEFAULT_OUTPUT_PATH = Path("data/shots.csv")

# (eventId, subEventId) → シュートの種類
SHOT_KINDS: dict[tuple[int, int], str] = {
    (10, 100): "shot",  # Shot / Shot
    (3, 33): "free_kick",  # Free Kick / Free kick shot（直接フリーキック）
    (3, 35): "penalty",  # Free Kick / Penalty
}

TAG_GOAL = 101
TAG_COUNTER_ATTACK = 1901
BODY_PART_TAGS: dict[int, str] = {
    401: "left_foot",
    402: "right_foot",
    403: "head_or_body",
}


@dataclass(frozen=True)
class Shot:
    """シュート 1本。列の意味は notes/shots-table.md。"""

    shot_id: int
    competition: str
    match_id: int
    match_period: str
    event_sec: float
    team_id: int
    player_id: int
    shot_type: str
    x: int
    y: int
    body_part: str
    is_counter_attack: bool
    prev_event: str
    prev_seconds_before: float | None
    is_goal: bool


COLUMNS: list[str] = [f.name for f in fields(Shot)]


def _tag_ids(event: dict[str, Any]) -> set[int]:
    tags = cast("list[dict[str, Any]]", event.get("tags", []))
    return {int(tag["id"]) for tag in tags}


def _kind(event: dict[str, Any]) -> str | None:
    """シュートの種類。シュートでなければ None。"""
    sub_event_id = event.get("subEventId")
    if sub_event_id in (None, ""):
        return None
    return SHOT_KINDS.get((int(event["eventId"]), int(sub_event_id)))


def _event_name(event: dict[str, Any]) -> str:
    """イベントの細かい種類の名前（なければ大きい種類の名前）。"""
    return str(event.get("subEventName") or event["eventName"])


def _body_part(tag_ids: set[int]) -> str:
    found = sorted(tag_ids & BODY_PART_TAGS.keys())
    parts = [BODY_PART_TAGS[t] for t in found]
    return parts[0] if len(parts) == 1 else ""


def extract_shots(
    events: Iterable[dict[str, Any]], competition: str
) -> Iterator[Shot]:
    """1大会分のイベントからシュートを抜き出す。

    イベントは、試合・前後半・時刻の順に並んでいるものとする（元データの並び）。
    直前のイベントは、相手チームのイベントを飛ばした、同じ試合・同じ前後半の
    中の同じチームの 1つ前のイベント（decisions/0002）。

    Args:
        events: 大会の JSON の中身（イベントの並び）。
        competition: 大会の名前（`events_England.json` なら `England`）。

    Yields:
        シュート。元データの並びの順。
    """
    current: tuple[int, str] | None = None
    last_by_team: dict[int, dict[str, Any]] = {}
    for event in events:
        key = (int(event["matchId"]), str(event["matchPeriod"]))
        if key != current:
            current = key
            last_by_team = {}
        team_id = int(event["teamId"])
        kind = _kind(event)
        if kind is not None:
            tag_ids = _tag_ids(event)
            start = cast("list[dict[str, Any]]", event["positions"])[0]
            event_sec = float(event["eventSec"])
            prev = last_by_team.get(team_id)
            yield Shot(
                shot_id=int(event["id"]),
                competition=competition,
                match_id=key[0],
                match_period=key[1],
                event_sec=round(event_sec, 3),
                team_id=team_id,
                player_id=int(event["playerId"]),
                shot_type=kind,
                x=int(start["x"]),
                y=int(start["y"]),
                body_part=_body_part(tag_ids),
                is_counter_attack=TAG_COUNTER_ATTACK in tag_ids,
                prev_event="" if prev is None else _event_name(prev),
                prev_seconds_before=(
                    None
                    if prev is None
                    else round(event_sec - float(prev["eventSec"]), 3)
                ),
                is_goal=TAG_GOAL in tag_ids,
            )
        last_by_team[team_id] = event


def competition_of(member: str) -> str:
    """zip の中のファイル名から大会の名前を出す。

    例: `events_England.json` → `England`
    """
    return Path(member).stem.removeprefix("events_")


def read_shots(events_path: Path = DEFAULT_EVENTS_PATH) -> list[Shot]:
    """`events.zip` を展開せずに読み、全大会のシュートを抜き出す。

    大会は zip の中のファイル名の順。大会ごとに JSON を 1つずつメモリに読む。
    """
    shots: list[Shot] = []
    with zipfile.ZipFile(events_path) as z:
        for member in sorted(z.namelist()):
            events = cast("list[dict[str, Any]]", json.loads(z.read(member)))
            shots.extend(extract_shots(events, competition_of(member)))
    return shots


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def write_shots(shots: Iterable[Shot], output_path: Path) -> None:
    """シュートの表を CSV に書く（真偽は 0/1、値がなければ空欄）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for shot in shots:
            row = asdict(shot)
            writer.writerow([_csv_value(row[c]) for c in COLUMNS])


def main() -> int:
    """`data/raw/events.zip` から `data/shots.csv` を作る。

    `mise run shots` で動かす。
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    shots = read_shots()
    write_shots(shots, DEFAULT_OUTPUT_PATH)
    counts = Counter(s.shot_type for s in shots)
    goals = Counter(s.shot_type for s in shots if s.is_goal)
    for kind in SHOT_KINDS.values():
        logger.info("%s: %d 本（ゴール %d）", kind, counts[kind], goals[kind])
    logger.info("書いた: %s（%d 行）", DEFAULT_OUTPUT_PATH, len(shots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
