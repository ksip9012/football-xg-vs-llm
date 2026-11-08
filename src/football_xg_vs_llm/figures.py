"""xG のモデルを確かめる図と、集計の図を PNG で作る（`mise run figures`）。

- `calibration.png`: 確率の当たり具合（10段階）
- `shot_map.png`: ハーフピッチ上の、場所ごとの実際のゴールの率と平均の xG
- `players_xg_goals.png`: 選手ごとの xG の合計と実際のゴール
- `finishing_split_half.png`: 試合を 2つに分けたときの、決定力と打つ場所のよさ

ラベルは英語（どの PC・CI でも同じ図になるように。decisions/0005）。
色はカテゴリの 1番目の青と、青 1色の濃淡だけを使う（dataviz の既定の
パレット）。

matplotlib は `**kwargs` の型を書いていないので、pyright の strict の
「型が一部わからない」の指摘は、このモジュールだけで外す。
"""
# pyright: reportUnknownMemberType=false

import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Rectangle

from football_xg_vs_llm.aggregate import (
    MIN_SHOTS,
    ScoredShot,
    SplitHalf,
    Summary,
    correlation,
    match_halves,
    scored_shots,
    split_half,
    summarize,
)
from football_xg_vs_llm.cv import DEFAULT_XG_PATH, load_xg
from football_xg_vs_llm.reference import (
    load_match_dates,
    load_player_names,
)
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    calibration,
    to_meters,
)

DEFAULT_FIGURES_DIR = Path("notes/figures")

# dataviz の既定のパレット（明るい面）
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#9a9994"
MASKED = "#f0efec"
SERIES = "#2a78d6"
BLUES = LinearSegmentedColormap.from_list(
    "blues",
    [
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
)

# シュートの地図のマス（メートル）と、率を出す本数の下限
CELL = 3.0
MIN_CELL_SHOTS = 30
MAP_MAX = 0.5

# 選手の散布図に入れる本数の下限と、名前を付ける人数
SCATTER_MIN_SHOTS = 20
LABELS_EACH_SIDE = 3


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _dots(ax: Axes, x: Sequence[float], y: Sequence[float]) -> None:
    """系列の点（青、表面の色の輪）。"""
    ax.scatter(
        x, y, s=36, color=SERIES, edgecolors=SURFACE, linewidths=1.0, zorder=3
    )


def _note(ax: Axes, text: str) -> None:
    ax.text(
        0.0,
        -0.16,
        text,
        transform=ax.transAxes,
        fontsize=8,
        color=TEXT_SECONDARY,
        va="top",
    )


def calibration_figure(rows: Sequence[ScoredShot], path: Path) -> None:
    """確率の当たり具合（シュートを xG の低い順に 10等分）。"""
    goals = np.array([r.shot.is_goal for r in rows], dtype=np.float64)
    xg = np.array([r.xg for r in rows], dtype=np.float64)
    bins = calibration(goals, xg)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    top = max(max(b.mean_prob for b in bins), max(b.goal_rate for b in bins))
    limit = float(np.ceil(top * 10) / 10)
    ax.plot([0, limit], [0, limit], color=REFERENCE, linewidth=1.0, zorder=1)
    ax.plot(
        [b.mean_prob for b in bins],
        [b.goal_rate for b in bins],
        color=SERIES,
        linewidth=2,
        solid_capstyle="round",
        zorder=2,
    )
    _dots(ax, [b.mean_prob for b in bins], [b.goal_rate for b in bins])
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted xG (mean in each decile)")
    ax.set_ylabel("Actual goal rate")
    ax.set_title(
        "xG calibration (out-of-fold, 10 equal-size bins)", loc="left"
    )
    _note(
        ax,
        f"{len(rows):,} non-penalty shots from the attacking half. "
        "Grey line: perfect calibration.",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _box(ax: Axes, x: float, y: float, width: float, height: float) -> None:
    """ピッチの線の四角（塗らない）。"""
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            edgecolor=REFERENCE,
            linewidth=1.0,
            zorder=3,
        )
    )


def half_pitch(ax: Axes) -> None:
    """攻める側のハーフピッチの線（ゴールが上。メートル）。"""
    half = PITCH_LENGTH / 2
    mid = PITCH_WIDTH / 2
    _box(ax, 0, half, PITCH_WIDTH, half)
    _box(ax, mid - 20.16, PITCH_LENGTH - 16.5, 40.32, 16.5)  # ペナルティエリア
    _box(ax, mid - 9.16, PITCH_LENGTH - 5.5, 18.32, 5.5)  # ゴールエリア
    _box(ax, mid - 3.66, PITCH_LENGTH, 7.32, 1.2)  # ゴール
    ax.add_patch(
        Arc(
            (mid, PITCH_LENGTH - 11),
            18.3,
            18.3,
            theta1=233,
            theta2=307,
            color=REFERENCE,
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.plot(mid, PITCH_LENGTH - 11, "o", color=REFERENCE, markersize=2)
    ax.set_xlim(-1, PITCH_WIDTH + 1)
    ax.set_ylim(half - 1, PITCH_LENGTH + 2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def shot_map_figure(rows: Sequence[ScoredShot], path: Path) -> None:
    """場所ごとの実際のゴールの率と平均の xG（同じ目盛り）。"""
    x = np.array([r.shot.x for r in rows], dtype=np.float64)
    y = np.array([r.shot.y for r in rows], dtype=np.float64)
    length, width = to_meters(x, y)
    goals = np.array([r.shot.is_goal for r in rows], dtype=np.float64)
    xg = np.array([r.xg for r in rows], dtype=np.float64)
    y_edges = np.arange(PITCH_LENGTH / 2, PITCH_LENGTH + CELL, CELL)
    x_edges = np.arange(0, PITCH_WIDTH + CELL, CELL)
    counts, _, _ = np.histogram2d(width, length, bins=[x_edges, y_edges])
    goal_sum, _, _ = np.histogram2d(
        width, length, bins=[x_edges, y_edges], weights=goals
    )
    xg_sum, _, _ = np.histogram2d(
        width, length, bins=[x_edges, y_edges], weights=xg
    )
    enough = counts >= MIN_CELL_SHOTS
    safe = np.where(counts > 0, counts, 1)
    panels = [
        ("Actual goal rate", np.where(enough, goal_sum / safe, np.nan)),
        ("Mean xG", np.where(enough, xg_sum / safe, np.nan)),
    ]
    cmap = BLUES.with_extremes(bad=MASKED, over=BLUES(1.0))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), layout="constrained")
    image = None
    for ax, (title, values) in zip(axes, panels, strict=True):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(values.T),
            cmap=cmap,
            vmin=0,
            vmax=MAP_MAX,
            edgecolors=SURFACE,
            linewidth=0.5,
            zorder=1,
        )
        half_pitch(ax)
        ax.set_title(title, loc="left")
    assert image is not None
    bar = fig.colorbar(image, ax=axes, shrink=0.8, extend="max", aspect=30)
    bar.outline.set_visible(False)  # pyright: ignore[reportAttributeAccessIssue]
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    fig.suptitle(
        "Where shots are scored: actual goal rate vs xG",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=12,
        fontweight="bold",
    )
    fig.supxlabel(
        f"{CELL:.0f} m cells, goal at the top. Grey cells have fewer than "
        f"{MIN_CELL_SHOTS} shots. Same colour scale in both panels "
        f"(capped at {MAP_MAX:.1f}).",
        x=0.01,
        ha="left",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _label_points(ax: Axes, points: Sequence[Summary]) -> None:
    """選んだ選手に名前を付ける。

    xG の大きい（右にいる）選手から順に置く。名前はふつう点の右に置き、
    すでに名前を置いた点の近くなら左に置く。近い 2人の名前は、互いに
    離れる向きになる。
    """
    placed: list[tuple[float, float]] = []
    for p in sorted(points, key=lambda s: -s.xg):
        near = any(
            abs(p.xg - x) < 4 and abs(p.goals - y) < 2 for x, y in placed
        )
        ax.annotate(
            p.label,
            (p.xg, p.goals),
            xytext=(-6, 3) if near else (6, 3),
            textcoords="offset points",
            ha="right" if near else "left",
            fontsize=8,
            color=TEXT_PRIMARY,
        )
        placed.append((p.xg, p.goals))


def players_figure(players: Sequence[Summary], path: Path) -> None:
    """選手ごとの xG の合計と実際のゴール（`SCATTER_MIN_SHOTS` 本以上）。"""
    shown = [p for p in players if p.shots >= SCATTER_MIN_SHOTS]
    fig, ax = plt.subplots(figsize=(5.6, 6.0), layout="constrained")
    top = max(max(p.xg for p in shown), max(p.goals for p in shown))
    limit = float(np.ceil(top / 5) * 5)
    ax.plot([0, limit], [0, limit], color=REFERENCE, linewidth=1.0, zorder=1)
    _dots(ax, [p.xg for p in shown], [p.goals for p in shown])
    ordered = sorted(shown, key=lambda p: p.goals_minus_xg)
    _label_points(
        ax, [*ordered[:LABELS_EACH_SIDE], *ordered[-LABELS_EACH_SIDE:]]
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("xG (sum)")
    ax.set_ylabel("Goals")
    ax.set_title("Goals vs xG by player", loc="left")
    _note(
        ax,
        f"{len(shown):,} players with {SCATTER_MIN_SHOTS}+ "
        "non-penalty shots.\n"
        "Above the grey line: more goals than xG.\n"
        f"Labelled: the {LABELS_EACH_SIDE} highest and lowest goals − xG.",
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def split_half_figure(pairs: Sequence[SplitHalf], path: Path) -> None:
    """試合を 2つに分けたときの、決定力と打つ場所のよさ（2面）。

    Raises:
        ValueError: 比べられる選手がいない（2人より少ない）場合。
    """
    if len(pairs) < 2:
        raise ValueError(f"比べられる選手が {len(pairs)}人しかいない")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    panels = [
        (
            "Finishing: (goals − xG) per shot",
            [p.first.finishing_per_shot for p in pairs],
            [p.second.finishing_per_shot for p in pairs],
            "zero",
        ),
        (
            "Shot quality: xG per shot",
            [p.first.xg_per_shot for p in pairs],
            [p.second.xg_per_shot for p in pairs],
            "diagonal",
        ),
    ]
    for ax, (title, first, second, guide) in zip(axes, panels, strict=True):
        r = correlation(list(zip(first, second, strict=True)))
        low = min(*first, *second)
        high = max(*first, *second)
        pad = (high - low) * 0.05
        lo, hi = low - pad, high + pad
        if guide == "zero":
            ax.axhline(0, color=REFERENCE, linewidth=1.0, zorder=1)
            ax.axvline(0, color=REFERENCE, linewidth=1.0, zorder=1)
        else:
            ax.plot(
                [lo, hi], [lo, hi], color=REFERENCE, linewidth=1.0, zorder=1
            )
        _dots(ax, first, second)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Odd matches")
        ax.set_ylabel("Even matches")
        ax.set_title(f"{title}   r = {r:.2f}", loc="left")
    fig.suptitle(
        "Does it repeat? Same players, matches split in two",
        x=0.02,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.01,
        f"{len(pairs)} players with {MIN_SHOTS}+ non-penalty shots in both "
        "halves. Matches are ordered by date within each competition and "
        "alternated.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(
    rows: Sequence[ScoredShot],
    players: Sequence[Summary],
    pairs: Sequence[SplitHalf],
    out_dir: Path,
) -> list[Path]:
    """4つの図を作り、置いた場所を返す。"""
    style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "calibration.png",
        out_dir / "shot_map.png",
        out_dir / "players_xg_goals.png",
        out_dir / "finishing_split_half.png",
    ]
    calibration_figure(rows, paths[0])
    shot_map_figure(rows, paths[1])
    players_figure(players, paths[2])
    split_half_figure(pairs, paths[3])
    return paths


def main() -> int:
    """`data/xg_stat.csv` から図を作る（`mise run figures`）。"""
    rows = scored_shots(load_shots_csv(), load_xg(DEFAULT_XG_PATH))
    names = load_player_names()

    def label(k: int | str) -> str:
        return names.get(int(k), f"(id {k})")

    players = summarize(rows, lambda s: s.player_id, label)
    pairs = split_half(rows, match_halves(load_match_dates()), label)
    for path in make_figures(rows, players, pairs, DEFAULT_FIGURES_DIR):
        print(f"書いた: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
