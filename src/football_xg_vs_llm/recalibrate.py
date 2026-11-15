"""Gemini の確率の「低めに出すくせ」を直して、統計モデルと比べる。

- 本題: Gemini の確率の対数オッズに、切片と傾きを当てはめる（Platt の方法）。
  `直した確率 = σ(a + b × logit(Gemini の確率))`
- 参考: 全体の高さだけを直す（傾きは 1 のまま、切片 `a` だけ）
- 直し方は、テストの 8,626本の中で試合単位の 5分割で学び、学んだのと別の試合の
  シュートに当てる。同じシュートで学んで測ると、よく見えすぎるため
- どちらの直し方も、同じ組の中では確率の高い順を変えない（傾きが正なら）。
  ただし組ごとに当てはめた数が少し違うので、組をまたぐと順位がわずかに入れ
  替わり、全体の AUC は少しだけ変わる。主に変わるのはログ損失・ブライア
  スコア・確率の当たり具合

決めごとは decisions/0008。`mise run recalibrate`（先に `mise run xg-cv` と、
Gemini の 2つの回）。
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from football_xg_vs_llm.compare_context import aligned, pair_of
from football_xg_vs_llm.compare_llm import (
    BOOTSTRAP,
    METRICS,
    bootstrap_differences,
)
from football_xg_vs_llm.cv import (
    CONTEXT_XG_PATH,
    DEFAULT_XG_PATH,
    assign_folds,
    load_xg,
)
from football_xg_vs_llm.llm_test import CONTEXT_XG_PATH as GEMINI_CONTEXT_PATH
from football_xg_vs_llm.llm_test import XG_PATH as GEMINI_PATH
from football_xg_vs_llm.llm_test import test_shots
from football_xg_vs_llm.models import Shot
from football_xg_vs_llm.shots import load_shots_csv
from football_xg_vs_llm.xg import (
    GROUP_ORDER,
    Floats,
    fit_numpy,
    prev_group,
    sigmoid,
)

# 確率 0・1 は対数オッズが無限になるので、この幅に収めてから直す
EPS = 1e-6


def logit(p: Floats) -> Floats:
    """確率を対数オッズにする（0・1 の近くは `EPS` で止める）。"""
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def fit_platt(probs: Floats, goals: Floats) -> tuple[float, float]:
    """`σ(a + b × logit(probs))` の切片 `a` と傾き `b` を当てはめる。"""
    coef = fit_numpy(logit(probs)[:, None], goals)
    return float(coef[0]), float(coef[1])


def fit_shift(
    probs: Floats, goals: Floats, *, tol: float = 1e-12, max_iter: int = 50
) -> float:
    """`σ(a + logit(probs))` の切片 `a` だけを当てはめる（傾きは 1 のまま）。

    対数尤度の勾配 `Σ(y − p)` が 0 になる `a` を、ニュートン法で探す。

    Raises:
        RuntimeError: `max_iter` 回までに収まらなかった場合。
    """
    z = logit(probs)
    a = 0.0
    for _ in range(max_iter):
        p = sigmoid(z + a)
        step = float(np.sum(goals - p) / np.sum(p * (1 - p)))
        a += step
        if abs(step) < tol:
            return a
    raise RuntimeError(f"{max_iter} 回で収まらなかった")


@dataclass(frozen=True)
class Recalibrated:
    """交差検証で直した確率と、組ごとに当てはめた数。"""

    platt: Floats
    shift: Floats
    platt_coefs: list[tuple[float, float]]
    shifts: list[float]


def recalibrate(
    probs: Floats, goals: Floats, match_ids: NDArray[np.int64]
) -> Recalibrated:
    """試合単位の 5分割で、ほかの 4組で学んだ直し方を残りの 1組に当てる。"""
    fold = assign_folds(match_ids)
    platt = np.empty(len(probs))
    shift = np.empty(len(probs))
    platt_coefs: list[tuple[float, float]] = []
    shifts: list[float] = []
    z = logit(probs)
    for k in np.unique(fold):
        test = fold == k
        a, b = fit_platt(probs[~test], goals[~test])
        platt[test] = sigmoid(a + b * z[test])
        platt_coefs.append((a, b))
        s = fit_shift(probs[~test], goals[~test])
        shift[test] = sigmoid(s + z[test])
        shifts.append(s)
    return Recalibrated(
        platt=platt, shift=shift, platt_coefs=platt_coefs, shifts=shifts
    )


def report(
    rows: Sequence[Shot], goals: Floats, probs: dict[str, Floats]
) -> None:
    """直す前と後の物差し・統計モデルとの差・組ごとの平均を出す。"""
    match_ids = np.array([s.match_id for s in rows], dtype=np.int64)
    fixed = {
        name: recalibrate(probs[name], goals, match_ids)
        for name in ("gemini", "gemini_context")
    }
    table: dict[str, Floats] = {
        "統計モデル（3つ）": probs["stat"],
        "Gemini（3つ）": probs["gemini"],
        "Gemini（3つ）・高さだけ直す": fixed["gemini"].shift,
        "Gemini（3つ）・高さと傾きを直す": fixed["gemini"].platt,
        "統計モデル（広い）": probs["stat_context"],
        "Gemini（広い）": probs["gemini_context"],
        "Gemini（広い）・高さだけ直す": fixed["gemini_context"].shift,
        "Gemini（広い）・高さと傾きを直す": fixed["gemini_context"].platt,
    }

    print("# Gemini の低めに出すくせを直して、統計モデルと比べる\n")
    print(
        f"- テストの {len(goals)}本・{len(np.unique(match_ids))}試合・"
        f"ゴール {int(goals.sum())}（{goals.mean():.4f}）。直し方は、"
        "この中で試合単位の 5分割で学び、別の試合に当てた\n"
    )
    print("| モデル | ログ損失 | ブライアスコア | AUC | 確率の平均 |")
    print("| --- | --- | --- | --- | --- |")
    for label, p in table.items():
        values = " | ".join(
            f"{metric(goals, p):.4f}" for metric in METRICS.values()
        )
        print(f"| {label} | {values} | {p.mean():.4f} |")

    print("\n当てはめた数（5組の平均と、いちばん小さい〜大きい）\n")
    print("| Gemini | 高さだけ: 切片 | 高さと傾き: 切片 | 高さと傾き: 傾き |")
    print("| --- | --- | --- | --- |")
    for name, label in (("gemini", "3つ"), ("gemini_context", "広い")):
        r = fixed[name]
        a = np.array([c[0] for c in r.platt_coefs])
        b = np.array([c[1] for c in r.platt_coefs])
        s = np.array(r.shifts)
        print(f"| {label} | {_range(s)} | {_range(a)} | {_range(b)} |")

    print(
        f"\n差（直した Gemini − 統計モデル）と 95% の幅（試合単位の"
        f"ブートストラップ {BOOTSTRAP}回。マイナスなら Gemini がよい）\n"
    )
    print("| 比べるもの | 物差し | 差 | 95% の幅 |")
    print("| --- | --- | --- | --- |")
    pairs = (
        ("統計モデル（3つ）", "Gemini（3つ）・高さと傾きを直す"),
        ("統計モデル（3つ）", "Gemini（3つ）・高さだけ直す"),
        ("統計モデル（広い）", "Gemini（広い）・高さと傾きを直す"),
        ("統計モデル（広い）", "Gemini（広い）・高さだけ直す"),
    )
    for before, after in pairs:
        data = pair_of(rows, goals, table[before], table[after])
        for d in bootstrap_differences(data):
            if d.name == "AUC":
                continue
            print(
                f"| {after} − {before} | {d.name} | {d.estimate:+.4f} | "
                f"{d.low:+.4f}〜{d.high:+.4f} |"
            )

    groups = np.array([prev_group(s) for s in rows])
    print("\n組ごとの、実際の率と確率の平均（広い情報）\n")
    print(
        "| 組 | 本数 | 実際の率 | 統計モデル（広い） | Gemini（広い） | "
        "高さと傾きを直す |"
    )
    print("| --- | --- | --- | --- | --- | --- |")
    for g in GROUP_ORDER:
        mask = groups == g
        if not mask.any():
            continue
        print(
            f"| {g} | {int(mask.sum())} | {goals[mask].mean():.3f} | "
            f"{probs['stat_context'][mask].mean():.3f} | "
            f"{probs['gemini_context'][mask].mean():.3f} | "
            f"{fixed['gemini_context'].platt[mask].mean():.3f} |"
        )


def _range(values: Floats) -> str:
    return f"{values.mean():+.3f}（{values.min():+.3f}〜{values.max():+.3f}）"


def main() -> int:
    """直して比べ、表を出す（`mise run recalibrate`）。"""
    xgs = {
        "stat": load_xg(DEFAULT_XG_PATH),
        "stat_context": load_xg(CONTEXT_XG_PATH),
        "gemini": load_xg(GEMINI_PATH),
        "gemini_context": load_xg(GEMINI_CONTEXT_PATH),
    }
    rows, goals, probs = aligned(test_shots(load_shots_csv()), xgs)
    report(rows, goals, probs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
