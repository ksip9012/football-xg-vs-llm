"""cv・aggregate・reference・figures のテスト（作り物のデータで確かめる）。"""

import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from football_xg_vs_llm.aggregate import (
    ScoredShot,
    correlation,
    match_halves,
    scored_shots,
    split_half,
    summarize,
)
from football_xg_vs_llm.cv import (
    assign_folds,
    load_xg,
    out_of_fold_xg,
    write_xg,
)
from football_xg_vs_llm.figures import make_figures
from football_xg_vs_llm.models import Shot, ShotType
from football_xg_vs_llm.reference import (
    load_match_dates,
    load_player_names,
    load_team_names,
)


def _shot(
    shot_id: int,
    *,
    player: int = 1,
    team: int = 10,
    match: int = 100,
    x: int = 90,
    y: int = 50,
    goal: bool = False,
    head: bool = False,
    shot_type: ShotType = "shot",
) -> Shot:
    return Shot(
        shot_id=shot_id,
        competition="England",
        match_id=match,
        match_period="1H",
        event_sec=1.0,
        team_id=team,
        player_id=player,
        shot_type=shot_type,
        x=x,
        y=y,
        body_part="head_or_body" if head else "right_foot",
        is_counter_attack=False,
        prev_event=None,
        prev_seconds_before=None,
        is_goal=goal,
    )


def _name(k: int | str) -> str:
    return f"p{k}"


def test_folds_keep_each_match_together_and_are_repeatable() -> None:
    match_ids = np.repeat(np.arange(50, dtype=np.int64), 4)

    folds = assign_folds(match_ids, folds=5)

    for m in range(50):
        assert len(set(folds[match_ids == m].tolist())) == 1
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
    assert (folds == assign_folds(match_ids, folds=5)).all()


def test_out_of_fold_xg_covers_every_modeled_shot() -> None:
    rng = np.random.default_rng(1)
    shots = [
        _shot(
            i,
            match=i // 10,
            x=int(rng.integers(60, 100)),
            y=int(rng.integers(20, 80)),
            goal=bool(rng.random() < 0.2),
            head=i % 4 == 0,
        )
        for i in range(400)
    ]
    shots.append(_shot(999, shot_type="penalty"))

    xg = out_of_fold_xg(shots)

    assert set(xg) == set(range(400))
    assert all(0 < v < 1 for v in xg.values())


def test_xg_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "xg.csv"

    write_xg({2: 0.25, 1: 0.1234567}, path)

    assert path.read_text().splitlines() == [
        "shot_id,xg",
        "1,0.123457",
        "2,0.250000",
    ]
    assert load_xg(path) == {1: 0.123457, 2: 0.25}


def test_xg_csv_rejects_values_outside_0_to_1(tmp_path: Path) -> None:
    path = tmp_path / "xg.csv"
    path.write_text("shot_id,xg\n1,1.5\n")

    with pytest.raises(ValidationError):
        load_xg(path)


def test_scored_shots_keeps_modeled_shots_and_needs_their_xg() -> None:
    shots = [
        _shot(1),
        _shot(2, shot_type="penalty"),
        _shot(3, x=40),
    ]

    rows = scored_shots(shots, {1: 0.3})

    assert [r.shot.shot_id for r in rows] == [1]
    with pytest.raises(KeyError):
        scored_shots([_shot(4)], {1: 0.3})


def test_summarize_adds_up_by_key() -> None:
    rows = [
        ScoredShot(_shot(1, player=7, goal=True), 0.4),
        ScoredShot(_shot(2, player=7), 0.2),
        ScoredShot(_shot(3, player=8, goal=True), 0.1),
    ]

    by_player = summarize(rows, lambda s: s.player_id, _name)

    assert [(s.label, s.shots, s.goals) for s in by_player] == [
        ("p7", 2, 1),
        ("p8", 1, 1),
    ]
    assert by_player[0].xg == pytest.approx(0.6)
    assert by_player[0].goals_minus_xg == pytest.approx(0.4)
    assert by_player[1].finishing_per_shot == pytest.approx(0.9)


def test_match_halves_alternate_by_date_within_each_competition() -> None:
    dates = {
        30: ("England", datetime(2018, 1, 3)),
        10: ("England", datetime(2018, 1, 1)),
        20: ("England", datetime(2018, 1, 2)),
        40: ("Spain", datetime(2017, 1, 1)),
    }

    halves = match_halves(dates)

    assert halves == {10: 0, 20: 1, 30: 0, 40: 0}


def test_split_half_needs_enough_shots_in_both_halves() -> None:
    rows = [
        ScoredShot(_shot(i, player=1, match=i % 2), 0.1) for i in range(40)
    ] + [ScoredShot(_shot(100 + i, player=2, match=0), 0.1) for i in range(40)]

    pairs = split_half(rows, {0: 0, 1: 1}, _name, min_shots=20)

    assert [(p.player_id, p.first.shots, p.second.shots) for p in pairs] == [
        (1, 20, 20)
    ]


def test_correlation() -> None:
    assert correlation([(1, 2), (2, 4), (3, 6)]) == pytest.approx(1.0)


def test_reference_names_are_unescaped(tmp_path: Path) -> None:
    players = tmp_path / "players.json"
    players.write_text(
        json.dumps([{"wyId": "5", "shortName": "L. Su\\u00e1rez", "role": {}}])
    )
    teams = tmp_path / "teams.json"
    teams.write_text(json.dumps([{"wyId": 7, "name": "Bayern M\\u00fcnchen"}]))

    assert load_player_names(players) == {5: "L. Suárez"}
    assert load_team_names(teams) == {7: "Bayern München"}


def test_match_dates_come_with_the_competition(tmp_path: Path) -> None:
    path = tmp_path / "matches.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "matches_World_Cup.json",
            json.dumps([{"wyId": 1, "dateutc": "2018-06-14 15:00:00"}]),
        )

    assert load_match_dates(path) == {
        1: ("World_Cup", datetime(2018, 6, 14, 15, 0))
    }


def test_make_figures_writes_four_pngs(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    rows = [
        ScoredShot(
            _shot(
                i,
                player=i % 30,
                match=i % 7,  # 選手の周期（30）と素にして、両方の組に出す
                x=int(rng.integers(70, 100)),
                y=int(rng.integers(30, 70)),
                goal=bool(rng.random() < 0.15),
            ),
            float(rng.uniform(0.02, 0.4)),
        )
        for i in range(3000)
    ]
    players = summarize(rows, lambda s: s.player_id, _name)
    pairs = split_half(rows, {m: m % 2 for m in range(10)}, _name)

    paths = make_figures(rows, players, pairs, tmp_path / "figures")

    assert [p.name for p in paths] == [
        "calibration.png",
        "shot_map.png",
        "players_xg_goals.png",
        "finishing_split_half.png",
    ]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
