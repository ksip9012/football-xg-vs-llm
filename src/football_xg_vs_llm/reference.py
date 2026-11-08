"""選手・チーム・試合の元データを、集計に使う形で読む（pydantic）。

- 選手の名前: `players.json` の `shortName`（例: `H. Kane`）
- チームの名前: `teams.json` の `name`
- 試合の日付と大会: `matches.zip` の大会ごとの JSON の `dateutc`

どれも使う項目だけをモデルにし、ID の型の揺れはモデルでそろえる。
"""

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, TypeAdapter
from pydantic.alias_generators import to_camel

DEFAULT_PLAYERS_PATH = Path("data/raw/players.json")
DEFAULT_TEAMS_PATH = Path("data/raw/teams.json")
DEFAULT_MATCHES_PATH = Path("data/raw/matches.zip")


_ESCAPED = re.compile(r"\\u([0-9a-fA-F]{4})")


def _unescape(value: object) -> object:
    r"""`ü` のように文字のまま残ったエスケープを、文字に戻す。

    元データの名前には、JSON として読んだあとも `Bayern München` の形の
    文字列が残っている（二重にエスケープされていると推定）。
    """
    if isinstance(value, str):
        return _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), value)
    return value


Name = Annotated[str, BeforeValidator(_unescape)]


class _Source(BaseModel):
    model_config = ConfigDict(frozen=True, alias_generator=to_camel)


class Player(_Source):
    """選手（`players.json` の 1件）。"""

    wy_id: int
    short_name: Name


class Team(_Source):
    """チーム（`teams.json` の 1件）。"""

    wy_id: int
    name: Name


class Match(_Source):
    """試合（`matches_<大会>.json` の 1件）。"""

    wy_id: int
    dateutc: datetime


def load_player_names(path: Path = DEFAULT_PLAYERS_PATH) -> dict[int, str]:
    """選手の ID → 短い名前。"""
    players = TypeAdapter(list[Player]).validate_json(path.read_bytes())
    return {p.wy_id: p.short_name for p in players}


def load_team_names(path: Path = DEFAULT_TEAMS_PATH) -> dict[int, str]:
    """チームの ID → 名前。"""
    teams = TypeAdapter(list[Team]).validate_json(path.read_bytes())
    return {t.wy_id: t.name for t in teams}


def load_match_dates(
    path: Path = DEFAULT_MATCHES_PATH,
) -> dict[int, tuple[str, datetime]]:
    """試合の ID → （大会の名前, 日時 UTC）。

    大会の名前は、zip の中のファイル名から出す（`matches_England.json` →
    `England`）。シュートの表の `competition` と同じ書き方。
    """
    adapter = TypeAdapter(list[Match])
    dates: dict[int, tuple[str, datetime]] = {}
    with zipfile.ZipFile(path) as z:
        for member in sorted(z.namelist()):
            competition = Path(member).stem.removeprefix("matches_")
            raw = json.loads(z.read(member))
            for match in adapter.validate_python(raw):
                dates[match.wy_id] = (competition, match.dateutc)
    return dates
