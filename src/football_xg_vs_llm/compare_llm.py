"""テストの 8,626本で、Gemini と統計モデルを比べる（decisions/0006）。

- 物差し: ログ損失・ブライアスコア・AUC（平均の率だけのモデルも並べる）
- 差の 95% の幅: 試合単位のブートストラップ（試合を選び直して、差を何千回も
  数え直す）。差は「Gemini − 統計モデル」で、ログ損失とブライアスコアは
  マイナスなら Gemini がよく、AUC はプラスなら Gemini がよい
- どこで違うか: 距離ごとの確率の平均と実際の率、場所ごとの差の地図
- 図: 確率の当たり具合を 2つ重ねたもの（`notes/figures/llm_calibration.png`）
  と、差の地図（`notes/figures/llm_difference_map.png`）

統計モデルは交差検証の xG（`data/xg_stat.csv`）、Gemini は `data/xg_gemini.csv`
（`mise run llm-test` で作る）。`mise run compare-llm`。
"""
# pyright: reportUnknownMemberType=false

import itertools
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

from football_xg_vs_llm.cv import DEFAULT_XG_PATH, load_xg
from football_xg_vs_llm.figures import (
    CELL,
    DEFAULT_FIGURES_DIR,
    MIN_CELL_SHOTS,
    REFERENCE,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    half_pitch,
    plt,
    style,
)
from football_xg_vs_llm.llm_test import XG_PATH, test_shots
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    Floats,
    brier_score,
    calibration,
    distance_and_angle,
    log_loss,
    roc_auc,
    to_meters,
)

BOOTSTRAP = 2000
SEED = 2026
DISTANCE_BANDS = (0.0, 6.0, 12.0, 18.0, 24.0, 200.0)

# dataviz の既定のパレット: 統計モデルはカテゴリの 1番目（青）。LLM は
# ブランドの色に寄せて、Gemini を 3番目（アクア）、Claude を 2番目（オレンジ）
# にする（2026-11-15 にユーザーが決めた。ほかの図でも同じ色にする）
STAT_COLOR = "#2a78d6"
GEMINI_COLOR = "#1baf7a"
CLAUDE_COLOR = "#eb6834"
# 正負の 2色（青 ↔ 赤）と灰色の真ん中
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b33130"]
DIFF_MAX = 0.1

Metric = Callable[[Floats, Floats], float]
METRICS: dict[str, Metric] = {
    "ログ損失": log_loss,
    "ブライアスコア": brier_score,
    "AUC": roc_auc,
}


@dataclass(frozen=True)
class Paired:
    """同じシュートの、2つのモデルの確率と結果。"""

    shots: list[Shot]
    goals: Floats
    stat: Floats
    llm: Floats
    match_ids: NDArray[np.int64]


def paired(
    shots: Sequence[Shot], stat: dict[int, float], llm: dict[int, float]
) -> Paired:
    """両方のモデルに確率のあるシュートだけを並べる。"""
    rows = [s for s in shots if s.shot_id in stat and s.shot_id in llm]
    return Paired(
        shots=rows,
        goals=np.array([s.is_goal for s in rows], dtype=np.float64),
        stat=np.array([stat[s.shot_id] for s in rows]),
        llm=np.array([llm[s.shot_id] for s in rows]),
        match_ids=np.array([s.match_id for s in rows], dtype=np.int64),
    )


@dataclass(frozen=True)
class Difference:
    """物差しの差（Gemini − 統計モデル）と、その 95% の幅。"""

    name: str
    estimate: float
    low: float
    high: float


def bootstrap_differences(
    data: Paired, *, rounds: int = BOOTSTRAP, seed: int = SEED
) -> list[Difference]:
    """試合単位のブートストラップで、物差しの差の 95% の幅を出す。

    試合を、同じ数だけ重複ありで選び直し、選ばれた試合のシュートで差を
    数え直す。これを `rounds` 回くり返し、2.5% と 97.5% の値を幅にする。
    """
    matches, inverse = np.unique(data.match_ids, return_inverse=True)
    rows_of: list[NDArray[np.intp]] = [
        np.flatnonzero(inverse == i) for i in range(len(matches))
    ]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in METRICS}
    for _ in range(rounds):
        picked = rng.integers(0, len(matches), size=len(matches))
        parts: list[NDArray[np.intp]] = [rows_of[int(i)] for i in picked]
        rows = np.concatenate(parts)
        goals = data.goals[rows]
        for name, metric in METRICS.items():
            samples[name].append(
                metric(goals, data.llm[rows]) - metric(goals, data.stat[rows])
            )
    return [
        Difference(
            name=name,
            estimate=metric(data.goals, data.llm)
            - metric(data.goals, data.stat),
            low=float(np.percentile(samples[name], 2.5)),
            high=float(np.percentile(samples[name], 97.5)),
        )
        for name, metric in METRICS.items()
    ]


def _distance(data: Paired) -> Floats:
    x = np.array([s.x for s in data.shots], dtype=np.float64)
    y = np.array([s.y for s in data.shots], dtype=np.float64)
    distance, _ = distance_and_angle(x, y)
    return distance


def report(data: Paired, differences: Sequence[Difference]) -> None:
    """物差し・差の幅・距離ごとの違いを Markdown で出す。"""
    print("# テストの 8,626本で、Gemini と統計モデルを比べる\n")
    print(
        f"- {len(data.goals)}本・{len(np.unique(data.match_ids))}試合・"
        f"ゴール {int(data.goals.sum())}（{data.goals.mean():.4f}）"
    )
    corr = float(np.corrcoef(data.llm, data.stat)[0, 1])
    print(f"- Gemini と統計モデルの確率の相関: {corr:.3f}\n")

    print("| モデル | ログ損失 | ブライアスコア | AUC | 確率の平均 |")
    print("| --- | --- | --- | --- | --- |")
    constant = np.full(len(data.goals), data.goals.mean())
    for label, probs in (
        ("平均の率だけ", constant),
        ("統計モデル（交差検証の xG）", data.stat),
        ("Gemini 3.8 Flash（LOW）", data.llm),
    ):
        values = " | ".join(
            f"{metric(data.goals, probs):.4f}" for metric in METRICS.values()
        )
        print(f"| {label} | {values} | {probs.mean():.4f} |")

    print(
        "\n差（Gemini − 統計モデル）と 95% の幅"
        f"（試合単位のブートストラップ {BOOTSTRAP}回）\n"
    )
    print("| 物差し | 差 | 95% の幅 | 読み方 |")
    print("| --- | --- | --- | --- |")
    for d in differences:
        better = (
            "マイナスなら Gemini がよい"
            if d.name != "AUC"
            else "プラスなら Gemini がよい"
        )
        print(
            f"| {d.name} | {d.estimate:+.4f} | {d.low:+.4f}〜{d.high:+.4f} | "
            f"{better} |"
        )

    distance = _distance(data)
    print("\n距離ごとの確率の平均と、実際のゴールの率\n")
    print("| 距離（m） | 本数 | 実際の率 | 統計モデル | Gemini |")
    print("| --- | --- | --- | --- | --- |")
    for lo, hi in itertools.pairwise(DISTANCE_BANDS):
        rows = (distance >= lo) & (distance < hi)
        label = f"{lo:.0f}〜{hi:.0f}" if hi < 100 else f"{lo:.0f} 以上"
        print(
            f"| {label} | {int(rows.sum())} | {data.goals[rows].mean():.3f} | "
            f"{data.stat[rows].mean():.3f} | {data.llm[rows].mean():.3f} |"
        )


def calibration_figure(data: Paired, path: Path) -> None:
    """確率の当たり具合を 2つ重ねる（どちらも 10等分）。"""
    fig, ax = plt.subplots(figsize=(5.4, 5.2), layout="constrained")
    series = [
        ("Statistical model", data.stat, STAT_COLOR),
        ("Gemini 3.8 Flash", data.llm, GEMINI_COLOR),
    ]
    bins_of = {name: calibration(data.goals, p) for name, p, _ in series}
    top = max(
        max(max(b.mean_prob, b.goal_rate) for b in bins)
        for bins in bins_of.values()
    )
    # 右上の点の名前が、枠やタイトルにかからないよう、上に余白を取る
    limit = float(np.ceil((top + 0.03) * 20) / 20)
    ax.plot([0, limit], [0, limit], color=REFERENCE, linewidth=1.0, zorder=1)
    # 右上の点の名前は、高いほうを点の上、低いほうを点の下に置いて重ねない
    highest = max(bins_of, key=lambda n: bins_of[n][-1].goal_rate)
    for name, _, color in series:
        bins = bins_of[name]
        xs = [b.mean_prob for b in bins]
        ys = [b.goal_rate for b in bins]
        ax.plot(xs, ys, color=color, linewidth=2, label=name, zorder=2)
        ax.scatter(
            xs,
            ys,
            s=36,
            color=color,
            edgecolors=SURFACE,
            linewidths=1.0,
            zorder=3,
        )
        ax.annotate(
            name,
            (xs[-1], ys[-1]),
            xytext=(-8, 6) if name == highest else (8, -12),
            textcoords="offset points",
            ha="right" if name == highest else "left",
            fontsize=8,
            color=TEXT_PRIMARY,
        )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability (mean in each decile)")
    ax.set_ylabel("Actual goal rate")
    ax.set_title("Calibration on the same test shots", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(
        0.0,
        -0.14,
        f"{len(data.goals):,} non-penalty test shots "
        f"({len(np.unique(data.match_ids))} matches). Grey line: perfect "
        "calibration.",
        transform=ax.transAxes,
        fontsize=8,
        color=TEXT_SECONDARY,
        va="top",
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def difference_map(data: Paired, path: Path) -> None:
    """場所ごとの、Gemini − 統計モデルの確率の平均（同じ目盛りの正負）。"""
    x = np.array([s.x for s in data.shots], dtype=np.float64)
    y = np.array([s.y for s in data.shots], dtype=np.float64)
    length, width = to_meters(x, y)
    y_edges = np.arange(PITCH_LENGTH / 2, PITCH_LENGTH + CELL, CELL)
    x_edges = np.arange(0, PITCH_WIDTH + CELL, CELL)
    bins = [x_edges, y_edges]
    counts, _, _ = np.histogram2d(width, length, bins=bins)
    diff_sum, _, _ = np.histogram2d(
        width, length, bins=bins, weights=data.llm - data.stat
    )
    enough = counts >= MIN_CELL_SHOTS
    mean = np.where(enough, diff_sum / np.where(counts > 0, counts, 1), np.nan)
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    cmap = cmap.with_extremes(bad=SURFACE)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), layout="constrained")
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(mean.T),
        cmap=cmap,
        vmin=-DIFF_MAX,
        vmax=DIFF_MAX,
        edgecolors=SURFACE,
        linewidth=0.5,
        zorder=1,
    )
    half_pitch(ax)
    ax.set_title(
        "Gemini minus statistical model (mean probability)", loc="left"
    )
    bar = fig.colorbar(image, ax=ax, shrink=0.8, extend="both", aspect=30)
    bar.outline.set_visible(False)  # pyright: ignore[reportAttributeAccessIssue]
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    fig.supxlabel(
        f"Red: Gemini higher. Blue: Gemini lower. {CELL:.0f} m cells with "
        f"{MIN_CELL_SHOTS}+ test shots; others left blank.",
        x=0.01,
        ha="left",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    """テストの 8,626本で比べ、表と図を出す（`mise run compare-llm`）。"""
    data = paired(
        test_shots(load_shots_csv()),
        load_xg(DEFAULT_XG_PATH),
        load_xg(XG_PATH),
    )
    report(data, bootstrap_differences(data))
    style()
    DEFAULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    calibration_figure(data, DEFAULT_FIGURES_DIR / "llm_calibration.png")
    difference_map(data, DEFAULT_FIGURES_DIR / "llm_difference_map.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
