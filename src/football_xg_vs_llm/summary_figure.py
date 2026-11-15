"""LLM と統計モデルの比べを 1枚にまとめる図（README 用）。

「LLM − 統計モデル」のログ損失の差と 95% の幅（試合単位のブートストラップ。
decisions/0006 と同じ）を、横に並べる。0 より右なら LLM が悪く、幅が 0 を
またげば見分けられない。

- 上: テストの 8,626本（Gemini だけ）
- 下: テストから選んだ 2,004本（Gemini と Claude。decisions/0009）
- 行: 3つの情報・広い情報（decisions/0007）・広い情報で高さを直したもの
  （decisions/0008 の高さだけの直し。同じシュートの中で試合単位の 5分割で学ぶ）

統計モデルは、それぞれ同じ情報のもの（3つの行は 3つの情報、残りは広い情報）。
`mise run summary-figure` で `notes/figures/llm_summary.png` を作る。
"""
# pyright: reportUnknownMemberType=false

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from football_xg_vs_llm.compare_context import aligned, pair_of
from football_xg_vs_llm.compare_llm import bootstrap_differences
from football_xg_vs_llm.cv import CONTEXT_XG_PATH, DEFAULT_XG_PATH, load_xg
from football_xg_vs_llm.figures import (
    DEFAULT_FIGURES_DIR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    plt,
    style,
)
from football_xg_vs_llm.llm_claude import XG_PATHS, sample_shots
from football_xg_vs_llm.llm_test import CONTEXT_XG_PATH as GEMINI_CONTEXT_PATH
from football_xg_vs_llm.llm_test import XG_PATH as GEMINI_PATH
from football_xg_vs_llm.llm_test import test_shots
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.recalibrate import recalibrate
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import Floats

# dataviz の既定のパレット: カテゴリの 2番目（オレンジ）と 3番目（アクア）。
# 1番目の青は、ほかの図で統計モデルに使っているので、ここでは使わない
GEMINI_COLOR = "#eb6834"
CLAUDE_COLOR = "#1baf7a"
COLORS = {"Gemini": GEMINI_COLOR, "Claude": CLAUDE_COLOR}
ROWS = (
    "Distance, angle, head",
    "+ previous action, counter",
    "+ context, intercept recalibrated",
)
FIGURE_PATH = DEFAULT_FIGURES_DIR / "llm_summary.png"


@dataclass(frozen=True)
class Interval:
    """1つの比べの、ログ損失の差（LLM − 統計モデル）と 95% の幅。"""

    panel: int
    row: int
    llm: str
    estimate: float
    low: float
    high: float


def _log_loss_difference(
    rows: Sequence[Shot], goals: Floats, stat: Floats, llm: Floats
) -> tuple[float, float, float]:
    (d,) = [
        d
        for d in bootstrap_differences(pair_of(rows, goals, stat, llm))
        if d.name == "ログ損失"
    ]
    return d.estimate, d.low, d.high


def intervals(shots: Sequence[Shot]) -> list[Interval]:
    """2つのシュートの組で、LLM ごと・行ごとの差と幅を出す。"""
    xgs = {
        "stat": load_xg(DEFAULT_XG_PATH),
        "stat_context": load_xg(CONTEXT_XG_PATH),
        "Gemini": load_xg(GEMINI_PATH),
        "Gemini_context": load_xg(GEMINI_CONTEXT_PATH),
        "Claude": load_xg(XG_PATHS[False]),
        "Claude_context": load_xg(XG_PATHS[True]),
    }
    test = test_shots(shots)
    panels = (
        (test, ("Gemini",)),
        (sample_shots(test), ("Gemini", "Claude")),
    )
    out: list[Interval] = []
    for panel, (subset, llms) in enumerate(panels):
        needed = {k: v for k, v in xgs.items() if k.split("_")[0] in llms}
        needed["stat"] = xgs["stat"]
        needed["stat_context"] = xgs["stat_context"]
        rows, goals, probs = aligned(subset, needed)
        match_ids = np.array([s.match_id for s in rows], dtype=np.int64)
        for llm in llms:
            wide = probs[f"{llm}_context"]
            fixed = recalibrate(wide, goals, match_ids).shift
            comparisons = (
                (probs["stat"], probs[llm]),
                (probs["stat_context"], wide),
                (probs["stat_context"], fixed),
            )
            for row, (stat, model) in enumerate(comparisons):
                est, low, high = _log_loss_difference(rows, goals, stat, model)
                out.append(Interval(panel, row, llm, est, low, high))
    return out


def figure(
    items: Sequence[Interval], sizes: Sequence[str], path: Path
) -> None:
    """差と幅を、2段の横向きの図にする。"""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        sharex=True,
        layout="constrained",
        height_ratios=(3, 3),
    )
    offsets = {"Gemini": 0.14, "Claude": -0.14}
    limit = max(max(abs(i.low), abs(i.high)) for i in items) * 1.15
    for panel, ax in enumerate(axes):
        mine = [i for i in items if i.panel == panel]
        llms = sorted({i.llm for i in mine}, key=list(COLORS).index)
        # 0 の線（統計モデルと同じ）は、格子より濃く太くする
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.2, zorder=1)
        for i in mine:
            y = len(ROWS) - 1 - i.row
            if len(llms) > 1:
                y += offsets[i.llm]
            color = COLORS[i.llm]
            ax.plot(
                [i.low, i.high], [y, y], color=color, linewidth=2, zorder=2
            )
            ax.scatter(
                [i.estimate],
                [y],
                s=64,
                color=color,
                edgecolors="#fcfcfb",
                linewidths=2,
                zorder=3,
                label=i.llm if i.row == 0 else None,
            )
            if i.row == 0:
                # 名前を点の右に直接書く（色だけに頼らない）
                ax.annotate(
                    i.llm,
                    (i.high, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=TEXT_PRIMARY,
                )
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels(list(reversed(ROWS)))
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(sizes[panel], loc="left", fontsize=10)
        ax.tick_params(axis="y", length=0)
    # 凡例は 2系列ある下の段だけに置く（上の段は Gemini だけ）
    axes[-1].legend(frameon=False, loc="upper left", fontsize=9)
    axes[-1].set_xlim(-limit, limit)
    axes[-1].set_xlabel(
        "Log-loss difference, LLM minus statistical model "
        "(right of 0: LLM worse)"
    )
    fig.suptitle(
        "LLMs vs a statistical xG model, with 95% intervals",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    fig.supxlabel(
        "Each LLM is compared with a statistical model given the same "
        "information.\nIntervals: match-level bootstrap (2,000 rounds). "
        "An interval crossing 0 means no clear difference.",
        x=0.01,
        ha="left",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    """図を作る（`mise run summary-figure`）。"""
    shots = load_shots_csv()
    items = intervals(shots)
    for i in items:
        print(
            f"{i.panel} {i.row} {i.llm}: {i.estimate:+.4f}"
            f"（{i.low:+.4f}〜{i.high:+.4f}）"
        )
    test = test_shots(shots)
    sample = sample_shots(test)
    sizes = (
        f"All test shots: {len(test):,} shots, "
        f"{len({s.match_id for s in test})} matches (Gemini only)",
        f"Claude subset: {len(sample):,} shots, "
        f"{len({s.match_id for s in sample})} matches",
    )
    style()
    DEFAULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure(items, sizes, FIGURE_PATH)
    print(f"書いた: {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
