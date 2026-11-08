"""試合単位の交差検証で、シュートごとの xG を出す。

どのシュートの xG も、そのシュートの試合を学習に使っていないモデルで出す。
同じシュートで学習したモデルの xG を使うと、決定力の差が小さく見えるため。

xG は `shot_id, xg` の 2列の CSV に置く（`data/xg_stat.csv`）。集計と図は、
この形の CSV なら何でも受け取れるので、あとで LLM の予測を同じ形にして
並べられる。
決めごとは decisions/0005。
"""

import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import (
    brier_score,
    build_dataset,
    fit_numpy,
    log_loss,
    predict,
    roc_auc,
)

DEFAULT_XG_PATH = Path("data/xg_stat.csv")
FOLDS = 5
SEED = 2026


class XgRow(BaseModel):
    """シュート 1本の xG（`data/xg_*.csv` の 1行）。"""

    model_config = ConfigDict(frozen=True)

    shot_id: int
    xg: float = Field(ge=0, le=1)


def assign_folds(
    match_ids: NDArray[np.int64], *, folds: int = FOLDS, seed: int = SEED
) -> NDArray[np.int64]:
    """試合ごとに、0〜`folds - 1` の組を割り当てる（同じ試合は同じ組）。"""
    matches = np.unique(match_ids)
    shuffled = np.random.default_rng(seed).permutation(matches)
    fold_of = {int(m): i % folds for i, m in enumerate(shuffled)}
    return np.array([fold_of[int(m)] for m in match_ids], dtype=np.int64)


def out_of_fold_xg(
    shots: Sequence[Shot], *, folds: int = FOLDS, seed: int = SEED
) -> dict[int, float]:
    """シュートの ID → xG（その試合の組を外して学習したモデルの値）。

    特徴は距離・角度・頭か体か（decisions/0004 の 2つ目の組）。
    """
    data = build_dataset(shots, with_head=True)
    fold = assign_folds(data.match_ids, folds=folds, seed=seed)
    xg = np.empty(len(data.goals))
    for k in range(folds):
        test = fold == k
        coef = fit_numpy(data.features[~test], data.goals[~test])
        xg[test] = predict(coef, data.features[test])
    return {int(s): float(v) for s, v in zip(data.shot_ids, xg, strict=True)}


def write_xg(xg: Mapping[int, float], path: Path) -> None:
    """xG を `shot_id, xg` の CSV に書く（ID の順、小数 6桁）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["shot_id", "xg"])
        for shot_id in sorted(xg):
            writer.writerow([shot_id, f"{xg[shot_id]:.6f}"])


def load_xg(path: Path = DEFAULT_XG_PATH) -> dict[int, float]:
    """`shot_id, xg` の CSV を読み、1行ずつ確かめる。"""
    with path.open(encoding="utf-8", newline="") as f:
        rows = [XgRow.model_validate(r) for r in csv.DictReader(f)]
    return {r.shot_id: r.xg for r in rows}


def main() -> int:
    """`data/shots.csv` から `data/xg_stat.csv` を作る（`mise run xg-cv`）。"""
    shots = load_shots_csv()
    xg = out_of_fold_xg(shots)
    write_xg(xg, DEFAULT_XG_PATH)
    data = build_dataset(shots, with_head=True)
    probs = np.array([xg[int(s)] for s in data.shot_ids])
    print(f"書いた: {DEFAULT_XG_PATH}（{len(xg)}本、{FOLDS}分割）")
    print(
        f"全体: ログ損失 {log_loss(data.goals, probs):.5f}・"
        f"ブライアスコア {brier_score(data.goals, probs):.5f}・"
        f"AUC {roc_auc(data.goals, probs):.4f}・"
        f"xG の合計 {probs.sum():.1f}・ゴール {int(data.goals.sum())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
