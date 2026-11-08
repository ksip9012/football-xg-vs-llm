"""シュートの表から、ロジスティック回帰で最初の xG を作るための共通の部品。

- 位置をメートルに直し、ゴールまでの距離と、両ポストを見込む角度を出す
- 試合単位で学習とテストに分ける
- 当たり方の物差し（ログ損失・ブライアスコア・AUC・確率の当たり具合）
- numpy だけで書いたロジスティック回帰（ニュートン法）

scikit-learn と statsmodels で同じことをするのは `libraries`。決めごとは
decisions/0004。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from football_xg_vs_llm.models import Shot

Floats = NDArray[np.float64]

# ピッチの大きさとゴールの幅（メートル）。元データは縦横を 0〜100 にした割合
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32

# 自陣から打ったシュートは、向きが逆に記録されたものが混ざっているので外す
MIN_X = 50


def to_meters(x: Floats, y: Floats) -> tuple[Floats, Floats]:
    """割合（0〜100）の位置を、メートルにする。"""
    return x / 100 * PITCH_LENGTH, y / 100 * PITCH_WIDTH


def distance_and_angle(x: Floats, y: Floats) -> tuple[Floats, Floats]:
    """ゴールの中央までの距離（m）と、両ポストを見込む角度（ラジアン）。

    ゴールは `x=100, y=50`（攻める向きにそろっている）。角度はゴールの幅が
    どれだけ開いて見えるかで、ゴールラインの上のポストの間なら π。

    Args:
        x: 打った位置の縦（0〜100）。
        y: 打った位置の横（0〜100）。
    """
    xm, ym = to_meters(x, y)
    dx = PITCH_LENGTH - xm
    dy = ym - PITCH_WIDTH / 2
    distance = np.hypot(dx, dy)
    half = GOAL_WIDTH / 2
    # 両ポストへの向きの差。x ≤ 100（dx ≥ 0）なので 0〜π に収まる
    angle = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - half**2)
    return distance, angle


@dataclass(frozen=True)
class Dataset:
    """モデルに渡す形にしたシュート。

    Attributes:
        features: 特徴の行列（行 = シュート、列 = `names`）。切片は入れない。
        names: 特徴の名前。
        goals: ゴールになったか（1.0 / 0.0）。
        match_ids: 試合の ID（学習とテストを試合単位で分けるため）。
        shot_ids: シュートの ID（xG を表の行とつなぐため）。
    """

    features: Floats
    names: tuple[str, ...]
    goals: Floats
    match_ids: NDArray[np.int64]
    shot_ids: NDArray[np.int64]


def is_modeled(shot: Shot) -> bool:
    """xG のモデルの対象か（普通のシュートで、自陣から打ったものではない）。"""
    return shot.shot_type == "shot" and shot.x >= MIN_X


def build_dataset(shots: Sequence[Shot], *, with_head: bool) -> Dataset:
    """普通のシュート（自陣からのものは外す）を、特徴の行列にする。

    Args:
        shots: シュートの表の行。
        with_head: 特徴に「頭か体か」（1.0 / 0.0）を足すか。
    """
    rows = [s for s in shots if is_modeled(s)]
    x = np.array([s.x for s in rows], dtype=np.float64)
    y = np.array([s.y for s in rows], dtype=np.float64)
    distance, angle = distance_and_angle(x, y)
    columns = [distance, angle]
    names = ["distance", "angle"]
    if with_head:
        head = [s.body_part == "head_or_body" for s in rows]
        columns.append(np.array(head, dtype=np.float64))
        names.append("head")
    return Dataset(
        features=np.column_stack(columns),
        names=tuple(names),
        goals=np.array([s.is_goal for s in rows], dtype=np.float64),
        match_ids=np.array([s.match_id for s in rows], dtype=np.int64),
        shot_ids=np.array([s.shot_id for s in rows], dtype=np.int64),
    )


def split_by_match(
    match_ids: NDArray[np.int64], *, test_share: float = 0.2, seed: int = 2026
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """試合単位で学習とテストに分ける（同じ試合のシュートは同じ側）。

    Returns:
        学習の行の印と、テストの行の印。
    """
    matches = np.unique(match_ids)
    rng = np.random.default_rng(seed)
    test_count = round(len(matches) * test_share)
    test_matches = rng.choice(matches, size=test_count, replace=False)
    test = np.isin(match_ids, test_matches)
    return ~test, test


def with_intercept(features: Floats) -> Floats:
    """先頭に切片の列（1.0）を足す。"""
    return np.column_stack([np.ones(len(features)), features])


def sigmoid(z: Floats) -> Floats:
    """ロジスティック関数。"""
    return 1 / (1 + np.exp(-z))


def fit_numpy(
    features: Floats, goals: Floats, *, tol: float = 1e-10, max_iter: int = 50
) -> Floats:
    """numpy だけでロジスティック回帰を当てはめる（ニュートン法、正則化なし）。

    対数尤度の勾配 `Xᵀ(y - p)` と、ヘッセ行列 `-Xᵀ W X`（`W = p(1 - p)`）で
    係数を更新し、変化が `tol` より小さくなったら止める。

    Returns:
        係数（先頭が切片、残りが `features` の列の順）。

    Raises:
        RuntimeError: `max_iter` 回までに収まらなかった場合。
    """
    design = with_intercept(features)
    coef = np.zeros(design.shape[1])
    for _ in range(max_iter):
        p = sigmoid(design @ coef)
        gradient = design.T @ (goals - p)
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian, gradient)
        coef = coef + step
        if np.max(np.abs(step)) < tol:
            return coef
    raise RuntimeError(f"{max_iter} 回で収まらなかった")


def predict(coef: Floats, features: Floats) -> Floats:
    """係数から、ゴールになる確率を出す。"""
    return sigmoid(with_intercept(features) @ coef)


def log_loss(goals: Floats, probs: Floats) -> float:
    """ログ損失（小さいほどよい）。"""
    p = np.clip(probs, 1e-15, 1 - 1e-15)
    return float(-np.mean(goals * np.log(p) + (1 - goals) * np.log(1 - p)))


def brier_score(goals: Floats, probs: Floats) -> float:
    """ブライアスコア（確率と結果の差の 2乗の平均。小さいほどよい）。"""
    return float(np.mean((probs - goals) ** 2))


def roc_auc(goals: Floats, probs: Floats) -> float:
    """AUC（ゴールのシュートに、外れより高い確率を付けられた割合）。

    同じ確率は順位を平均して数える（マン・ホイットニーの U と同じ）。
    ブートストラップで何千回も呼ぶので、ループを使わずに numpy で数える。
    """
    _, inverse, counts = np.unique(
        probs, return_inverse=True, return_counts=True
    )
    # 同じ値のかたまりの、最初の順位（1 から）と最後の順位の平均
    first = np.cumsum(counts) - counts + 1
    average_rank = first + (counts - 1) / 2
    ranks = average_rank[inverse]
    positives = goals == 1
    n_pos = int(positives.sum())
    n_neg = len(goals) - n_pos
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass(frozen=True)
class CalibrationBin:
    """確率の当たり具合の 1段階。"""

    count: int
    mean_prob: float
    goal_rate: float


def calibration(
    goals: Floats, probs: Floats, *, bins: int = 10
) -> list[CalibrationBin]:
    """確率の当たり具合を、段階ごとに並べる。

    確率の低い順に同じ本数ずつ `bins` 段階に分け、出した確率の平均と、
    実際のゴールの率を出す。
    """
    order = np.argsort(probs, kind="mergesort")
    return [
        CalibrationBin(
            count=len(part),
            mean_prob=float(probs[part].mean()),
            goal_rate=float(goals[part].mean()),
        )
        for part in np.array_split(order, bins)
    ]
