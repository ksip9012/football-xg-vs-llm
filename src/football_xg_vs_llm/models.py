"""元データのイベントと、シュートの表の 1行のモデル（pydantic）。

- `Event`: Wyscout のイベント 1件。使う項目だけを持つ。シュート以外のイベントも
  通るので、型をそろえるだけにして、値の範囲は縛らない（位置は -1〜101 がある）
- `Shot`: シュートの表（`data/shots.csv`）の 1行。決まった値・範囲を確かめる。
  表を書くときと、CSV を読み戻すときの両方で使う

列の意味は notes/shots-table.md、表の決めごとは decisions/0002。
"""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.alias_generators import to_camel


def _empty_to_none(value: object) -> object:
    """空文字を「値なし」にする（元データの `subEventId`、CSV の空欄）。"""
    return None if value == "" else value


EmptyAsNone = BeforeValidator(_empty_to_none)


class Position(BaseModel):
    """イベントの位置（ピッチの縦横を 0〜100 にした割合）。

    元データには -1 や 101 もあるので、範囲は縛らない。
    """

    model_config = ConfigDict(frozen=True)

    x: int
    y: int


class Tag(BaseModel):
    """イベントのタグ（番号の意味は `tags2name.csv`）。"""

    model_config = ConfigDict(frozen=True)

    id: int


class Event(BaseModel):
    """Wyscout のイベント 1件。項目の名前は元データの camelCase で読む。"""

    model_config = ConfigDict(frozen=True, alias_generator=to_camel)

    id: int
    event_id: int
    sub_event_id: Annotated[int | None, EmptyAsNone]
    event_name: str
    sub_event_name: str
    match_id: int
    match_period: str
    event_sec: float
    team_id: int
    player_id: int
    positions: list[Position] = Field(min_length=1)
    tags: list[Tag]

    @property
    def tag_ids(self) -> frozenset[int]:
        """付いているタグの番号。"""
        return frozenset(tag.id for tag in self.tags)

    @property
    def name(self) -> str:
        """細かい種類の名前（なければ大きい種類の名前）。"""
        return self.sub_event_name or self.event_name


ShotType = Literal["shot", "free_kick", "penalty"]
BodyPart = Literal["left_foot", "right_foot", "head_or_body"]
MatchPeriod = Literal["1H", "2H", "E1", "E2", "P"]
Competition = Literal[
    "England",
    "European_Championship",
    "France",
    "Germany",
    "Italy",
    "Spain",
    "World_Cup",
]


class Shot(BaseModel):
    """シュートの表の 1行。列の意味は notes/shots-table.md。"""

    model_config = ConfigDict(frozen=True)

    shot_id: int
    competition: Competition
    match_id: int
    match_period: MatchPeriod
    event_sec: float = Field(ge=0)
    team_id: int
    player_id: int
    shot_type: ShotType
    x: int = Field(ge=0, le=100)
    y: int = Field(ge=0, le=100)
    body_part: BodyPart
    is_counter_attack: bool
    prev_event: Annotated[str | None, EmptyAsNone]
    prev_seconds_before: Annotated[float | None, EmptyAsNone] = Field(ge=0)
    is_goal: bool

    @model_validator(mode="after")
    def _prev_columns_go_together(self) -> Self:
        """直前のイベントの 2列は、両方あるか、両方ないか。"""
        if (self.prev_event is None) != (self.prev_seconds_before is None):
            raise ValueError(
                "prev_event と prev_seconds_before は、両方あるか両方ないか"
            )
        return self


COLUMNS: list[str] = list(Shot.model_fields)
