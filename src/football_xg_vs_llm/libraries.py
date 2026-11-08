"""scikit-learn と statsmodels で、ロジスティック回帰を当てはめる。

どちらも型の情報（スタブ）がないので、pyright の strict の「型がわからない」
の指摘は、このモジュールだけで外す。外に出す値は numpy の配列にそろえる。
テストの確率は、それぞれのライブラリ自身の予測の関数で出す。
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from football_xg_vs_llm.xg import Floats, with_intercept


@dataclass(frozen=True)
class Fit:
    """当てはめた結果。

    Attributes:
        coef: 係数（先頭が切片、残りは特徴の列の順）。
        test_probs: テストのシュートがゴールになる確率。
        std_err: 係数の標準誤差（statsmodels だけ）。
        p_values: 係数の p 値（statsmodels だけ）。
        conf_low: 係数の 95% 信頼区間の下（statsmodels だけ）。
        conf_high: 係数の 95% 信頼区間の上（statsmodels だけ）。
    """

    coef: Floats
    test_probs: Floats
    std_err: Floats | None = None
    p_values: Floats | None = None
    conf_low: Floats | None = None
    conf_high: Floats | None = None


def fit_sklearn(train: Floats, goals: Floats, test: Floats) -> Fit:
    """scikit-learn で当てはめる。正則化は外して、最尤推定にそろえる。

    正則化を外すのは `C=np.inf`（`penalty=None` は 1.8 で非推奨になり、
    1.10 で消える）。
    """
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10_000)
    model.fit(train, goals)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    slopes = np.asarray(model.coef_, dtype=np.float64)[0]
    probs = np.asarray(model.predict_proba(test), dtype=np.float64)[:, 1]
    return Fit(coef=np.concatenate([intercept, slopes]), test_probs=probs)


def fit_statsmodels(train: Floats, goals: Floats, test: Floats) -> Fit:
    """statsmodels の `Logit` で当てはめる（ニュートン法）。"""
    result = sm.Logit(goals, with_intercept(train)).fit(disp=0)
    conf = np.asarray(result.conf_int(alpha=0.05), dtype=np.float64)
    probs = np.asarray(result.predict(with_intercept(test)), dtype=np.float64)
    return Fit(
        coef=np.asarray(result.params, dtype=np.float64),
        test_probs=probs,
        std_err=np.asarray(result.bse, dtype=np.float64),
        p_values=np.asarray(result.pvalues, dtype=np.float64),
        conf_low=conf[:, 0],
        conf_high=conf[:, 1],
    )
