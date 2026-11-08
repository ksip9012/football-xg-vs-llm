"""イベント（`events.zip`）からシュートを抜き出し、1行 1本の表にする。

表の形の決めごとは decisions/0002、列の説明は notes/shots-table.md。
シュートのあとに決まる情報（ゴールのどこに飛んだか、枠内か、ブロックか、
終わりの位置、タグ 201 Opportunity）は、答えが漏れるので表に入れない。
イベントと表の 1行は、models の pydantic のモデルで確かめる。
"""

import csv
import logging
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import TypeAdapter

from football_xg_vs_llm.models import COLUMNS, Event, Shot, ShotType

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_PATH = Path("data/raw/events.zip")
DEFAULT_OUTPUT_PATH = Path("data/shots.csv")

# (eventId, subEventId) → シュートの種類
SHOT_KINDS: dict[tuple[int, int], ShotType] = {
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

_EVENTS = TypeAdapter(list[Event])


def _kind(event: Event) -> ShotType | None:
    """シュートの種類。シュートでなければ None。"""
    if event.sub_event_id is None:
        return None
    return SHOT_KINDS.get((event.event_id, event.sub_event_id))


def _body_part(tag_ids: frozenset[int]) -> str | None:
    """体の部位のタグがちょうど 1つなら、その名前。"""
    found = sorted(tag_ids & BODY_PART_TAGS.keys())
    parts = [BODY_PART_TAGS[t] for t in found]
    return parts[0] if len(parts) == 1 else None


def extract_shots(events: Iterable[Event], competition: str) -> Iterator[Shot]:
    """1大会分のイベントからシュートを抜き出す。

    イベントは、試合・前後半・時刻の順に並んでいるものとする（元データの並び）。
    直前のイベントは、相手チームのイベントを飛ばした、同じ試合・同じ前後半の
    中の同じチームの 1つ前のイベント（decisions/0002）。

    Args:
        events: 大会のイベントの並び。
        competition: 大会の名前（`events_England.json` なら `England`）。

    Yields:
        シュート。元データの並びの順。

    Raises:
        pydantic.ValidationError: シュートの行が決まりに合わない場合
            （体の部位のタグがない、位置が 0〜100 の外など）。
    """
    current: tuple[int, str] | None = None
    last_by_team: dict[int, Event] = {}
    for event in events:
        key = (event.match_id, event.match_period)
        if key != current:
            current = key
            last_by_team = {}
        kind = _kind(event)
        if kind is not None:
            tag_ids = event.tag_ids
            start = event.positions[0]
            prev = last_by_team.get(event.team_id)
            yield Shot.model_validate(
                {
                    "shot_id": event.id,
                    "competition": competition,
                    "match_id": event.match_id,
                    "match_period": event.match_period,
                    "event_sec": round(event.event_sec, 3),
                    "team_id": event.team_id,
                    "player_id": event.player_id,
                    "shot_type": kind,
                    "x": start.x,
                    "y": start.y,
                    "body_part": _body_part(tag_ids),
                    "is_counter_attack": TAG_COUNTER_ATTACK in tag_ids,
                    "prev_event": None if prev is None else prev.name,
                    "prev_seconds_before": (
                        None
                        if prev is None
                        else round(event.event_sec - prev.event_sec, 3)
                    ),
                    "is_goal": TAG_GOAL in tag_ids,
                }
            )
        last_by_team[event.team_id] = event


def competition_of(member: str) -> str:
    """zip の中のファイル名から大会の名前を出す。

    例: `events_England.json` → `England`
    """
    return Path(member).stem.removeprefix("events_")


def read_events(raw_json: bytes) -> list[Event]:
    """大会の JSON をそのまま pydantic で読み、イベントの並びにする。"""
    return _EVENTS.validate_json(raw_json)


def read_shots(events_path: Path = DEFAULT_EVENTS_PATH) -> list[Shot]:
    """`events.zip` を展開せずに読み、全大会のシュートを抜き出す。

    大会は zip の中のファイル名の順。大会ごとに JSON を 1つずつメモリに読む。
    """
    shots: list[Shot] = []
    with zipfile.ZipFile(events_path) as z:
        for member in sorted(z.namelist()):
            events = read_events(z.read(member))
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
            row = shot.model_dump()
            writer.writerow([_csv_value(row[c]) for c in COLUMNS])


def load_shots_csv(path: Path = DEFAULT_OUTPUT_PATH) -> list[Shot]:
    """シュートの表の CSV を読み戻し、1行ずつモデルで確かめる。

    Raises:
        ValueError: 列の並びが表の決まりと違う場合。
        pydantic.ValidationError: 行が決まりに合わない場合。
    """
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path} の列が違う: {reader.fieldnames}")
        return [Shot.model_validate(row) for row in reader]


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
