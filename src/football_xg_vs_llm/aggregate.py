"""シュートごとの xG を、選手・チーム・大会ごとに集計する。

- 対象は xG のモデルと同じ普通のシュート（PK・直接フリーキック・自陣を除く）。
  ゴールもこの範囲で数えるので、公式のゴール数とは合わない
- xG は `shot_id → xg` の対応で受け取る（`data/xg_stat.csv` など）。
  LLM の予測も同じ形にすれば、同じ集計で並べられる
- 決定力が本物か: 大会ごとに日付の順に並べた試合を奇数と偶数に分け、両方で
  `MIN_SHOTS` 本以上打った選手の「1本あたりのゴール − xG」が、もう片方でも
  続くか（相関）を見る

決めごとは decisions/0005。
"""

import csv
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from football_xg_vs_llm.cv import DEFAULT_XG_PATH, load_xg
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.reference import (
    load_match_dates,
    load_player_names,
    load_team_names,
)
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import is_modeled

DEFAULT_OUTPUT_DIR = Path("data/aggregates")
MIN_SHOTS = 20


@dataclass(frozen=True)
class Summary:
    """1つの集まり（選手・チーム・大会）の集計。"""

    key: int | str
    label: str
    shots: int
    goals: int
    xg: float

    @property
    def goals_minus_xg(self) -> float:
        """ゴール − xG（プラスなら、xG より多く決めた）。"""
        return self.goals - self.xg

    @property
    def xg_per_shot(self) -> float:
        """1本あたりの xG（いい場所から打てているか）。"""
        return self.xg / self.shots

    @property
    def finishing_per_shot(self) -> float:
        """1本あたりのゴール − xG。"""
        return self.goals_minus_xg / self.shots


@dataclass(frozen=True)
class ScoredShot:
    """xG を付けた、集計の対象のシュート。"""

    shot: Shot
    xg: float


def scored_shots(
    shots: Iterable[Shot], xg: Mapping[int, float]
) -> list[ScoredShot]:
    """対象のシュートに xG を付ける。

    Raises:
        KeyError: 対象のシュートに xG がない場合（xG の CSV が古いなど）。
    """
    rows: list[ScoredShot] = []
    for shot in shots:
        if not is_modeled(shot):
            continue
        if shot.shot_id not in xg:
            raise KeyError(f"シュート {shot.shot_id} の xG がない")
        rows.append(ScoredShot(shot=shot, xg=xg[shot.shot_id]))
    return rows


def summarize(
    rows: Iterable[ScoredShot],
    key: Callable[[Shot], int | str],
    label: Callable[[int | str], str],
) -> list[Summary]:
    """`key` ごとにまとめる。xG の合計の大きい順。"""
    groups: dict[int | str, list[ScoredShot]] = defaultdict(list)
    for row in rows:
        groups[key(row.shot)].append(row)
    summaries = [
        Summary(
            key=k,
            label=label(k),
            shots=len(group),
            goals=sum(r.shot.is_goal for r in group),
            xg=sum(r.xg for r in group),
        )
        for k, group in groups.items()
    ]
    return sorted(summaries, key=lambda s: (-s.xg, str(s.key)))


def match_halves(
    match_dates: Mapping[int, tuple[str, datetime]],
) -> dict[int, int]:
    """試合の ID → 0 か 1（大会ごとに日付の順に並べ、交互に分ける）。"""
    by_competition: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
    for match_id, (competition, date) in match_dates.items():
        by_competition[competition].append((date, match_id))
    halves: dict[int, int] = {}
    for matches in by_competition.values():
        for i, (_, match_id) in enumerate(sorted(matches)):
            halves[match_id] = i % 2
    return halves


@dataclass(frozen=True)
class SplitHalf:
    """選手ごとの、2つに分けた試合のそれぞれの集計。"""

    player_id: int
    label: str
    first: Summary
    second: Summary


def split_half(
    rows: Sequence[ScoredShot],
    halves: Mapping[int, int],
    label: Callable[[int | str], str],
    *,
    min_shots: int = MIN_SHOTS,
) -> list[SplitHalf]:
    """両方の組で `min_shots` 本以上打った選手の、組ごとの集計。"""
    parts = [
        {
            s.key: s
            for s in summarize(
                (r for r in rows if halves[r.shot.match_id] == h),
                key=lambda shot: shot.player_id,
                label=label,
            )
        }
        for h in (0, 1)
    ]
    result = [
        SplitHalf(
            player_id=int(key),
            label=first.label,
            first=first,
            second=parts[1][key],
        )
        for key, first in parts[0].items()
        if key in parts[1]
        and first.shots >= min_shots
        and parts[1][key].shots >= min_shots
    ]
    return sorted(result, key=lambda r: r.player_id)


def correlation(pairs: Sequence[tuple[float, float]]) -> float:
    """ピアソンの相関係数。"""
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return float(np.corrcoef(a, b)[0, 1])


def write_summaries(summaries: Iterable[Summary], path: Path) -> None:
    """集計を CSV に書く。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["key", "label", "shots", "goals", "xg", "goals_minus_xg"]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.key,
                    s.label,
                    s.shots,
                    s.goals,
                    f"{s.xg:.3f}",
                    f"{s.goals_minus_xg:.3f}",
                ]
            )


def _table(title: str, summaries: Sequence[Summary]) -> None:
    print(f"\n### {title}\n")
    print("| 名前 | 本数 | ゴール | xG | ゴール − xG | 1本あたりの xG |")
    print("| --- | --- | --- | --- | --- | --- |")
    for s in summaries:
        print(
            f"| {s.label} | {s.shots} | {s.goals} | {s.xg:.1f} | "
            f"{s.goals_minus_xg:+.1f} | {s.xg_per_shot:.3f} |"
        )


def main() -> int:
    """`data/xg_stat.csv` で集計し、CSV と Markdown を出す。

    `mise run aggregate` で動かす。
    """
    shots = load_shots_csv()
    rows = scored_shots(shots, load_xg(DEFAULT_XG_PATH))
    players = load_player_names()
    teams = load_team_names()

    def player_label(k: int | str) -> str:
        return players.get(int(k), f"(id {k})")

    def team_label(k: int | str) -> str:
        return teams.get(int(k), f"(id {k})")

    by_player = summarize(rows, lambda s: s.player_id, player_label)
    by_team = summarize(rows, lambda s: s.team_id, team_label)
    by_competition = summarize(rows, lambda s: s.competition, str)
    for name, summaries in (
        ("players", by_player),
        ("teams", by_team),
        ("competitions", by_competition),
    ):
        write_summaries(summaries, DEFAULT_OUTPUT_DIR / f"{name}_stat.csv")

    total_goals = sum(r.shot.is_goal for r in rows)
    total_xg = sum(r.xg for r in rows)
    print("## 集計（PK・直接フリーキック・自陣を除く普通のシュート）\n")
    print(f"- {len(rows)}本・ゴール {total_goals}・xG {total_xg:.1f}")
    print(
        f"- 選手 {len(by_player)}人・チーム {len(by_team)}・"
        f"大会 {len(by_competition)}"
    )

    _table("大会ごと", by_competition)
    _table("チームごと（xG の多い順に 10）", by_team[:10])
    _table("選手ごと（xG の多い順に 15）", by_player[:15])
    many = [s for s in by_player if s.shots >= 50]
    many.sort(key=lambda s: s.goals_minus_xg, reverse=True)
    _table("ゴール − xG の大きい選手（50本以上）", many[:10])
    _table("ゴール − xG の小さい選手（50本以上）", many[-10:][::-1])

    halves = match_halves(load_match_dates())
    pairs = split_half(rows, halves, player_label)
    finishing = correlation(
        [
            (p.first.finishing_per_shot, p.second.finishing_per_shot)
            for p in pairs
        ]
    )
    quality = correlation(
        [(p.first.xg_per_shot, p.second.xg_per_shot) for p in pairs]
    )
    print("\n## 決定力が本物か（試合を奇数と偶数に分けて比べる）\n")
    print(f"- 両方の組で {MIN_SHOTS}本以上打った選手: {len(pairs)}人")
    print(f"- 1本あたりのゴール − xG の相関: {finishing:.3f}")
    print(f"- 1本あたりの xG（打つ場所のよさ）の相関: {quality:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
