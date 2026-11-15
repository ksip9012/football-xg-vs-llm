"""summary_figure のテスト（作り物の数で、図が書けることを確かめる）。"""

from pathlib import Path

from football_xg_vs_llm.figures import style
from football_xg_vs_llm.summary_figure import Interval, figure


def test_figure_writes_a_png(tmp_path: Path) -> None:
    items = [
        Interval(
            panel=0,
            row=r,
            llm="Gemini",
            estimate=0.001,
            low=-0.001,
            high=0.003,
        )
        for r in range(3)
    ] + [
        Interval(
            panel=1, row=r, llm=llm, estimate=0.002, low=-0.002, high=0.006
        )
        for r in range(3)
        for llm in ("Gemini", "Claude")
    ]
    path = tmp_path / "summary.png"
    style()

    figure(items, ("top", "bottom"), path)

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
