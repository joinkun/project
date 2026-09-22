"""生成された終点(x0, x1、総資産で正規化済み=tilde空間)から行ごとの指標を計算する(ステージ13)。

入力はすでに総資産で正規化済みなので、Assetsによる再正規化はしない
(ステージ10.6の二重正規化バグの再発防止)。validityは分類器を必要とするので、
呼び出し側(lightgbm/sklearnのプロセス)が``classifiers``を渡す。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import CanonicalSpec, build_constraint_matrix
from fsgen.evaluation.classifiers import ValidityClassifier

_SPARSITY_THR = 0.01
_SIGN_TOL = 1e-4
_CASH = "jppfs_cor_CashAndDeposits"
_RE = "jppfs_cor_RetainedEarnings"


def compute_row_metrics(
    df: pd.DataFrame,
    names: tuple[str, ...],
    spec: CanonicalSpec,
    w_plus: np.ndarray,
    w_minus: np.ndarray,
    classifiers: Mapping[str, ValidityClassifier] | None = None,
) -> pd.DataFrame:
    """dfの各行(x0_*, x1_*)について指標を計算し、識別列と結合して返す。"""
    c = build_constraint_matrix(spec, include_subtotals=False)
    x0 = df[[f"x0_{n}" for n in names]].to_numpy()
    x1 = df[[f"x1_{n}" for n in names]].to_numpy()
    delta = x1 - x0
    id_candidates = ("tag", "design", "condition", "seed", "fold", "method", "docID", "filerName")
    id_cols = [col for col in id_candidates if col in df.columns]
    out = df[id_cols].copy()
    out["proximity_l2"] = np.linalg.norm(delta, axis=1)
    out["proximity_cost_weighted"] = np.sum(
        np.where(delta >= 0, w_plus, w_minus) * np.abs(delta), axis=1
    )
    out["sparsity"] = np.sum(np.abs(delta) > _SPARSITY_THR, axis=1)
    out["balance_residual"] = np.max(np.abs(x1 @ c), axis=1)
    viol = np.zeros(len(df), dtype=int)
    for j, b in enumerate(spec.basis):
        if b.sign == "nonneg":
            viol += (x1[:, j] < -_SIGN_TOL).astype(int)
        elif b.sign == "nonpos":
            viol += (x1[:, j] > _SIGN_TOL).astype(int)
    out["sign_violations"] = viol
    asset_idx = [i for i, b in enumerate(spec.basis) if b.side == "assets"]
    a0 = x0[:, asset_idx].sum(axis=1)
    out["asset_change"] = (x1[:, asset_idx].sum(axis=1) - a0) / a0
    j_cash, j_re = names.index(_CASH), names.index(_RE)
    out["cash_zero"] = ((x1[:, j_cash] < 1e-6) & (x0[:, j_cash] > 1e-3)).astype(int)
    out["cash_start"] = x0[:, j_cash]
    out["re_change"] = np.abs(x1[:, j_re] - x0[:, j_re])
    if classifiers is not None:
        for name, clf in classifiers.items():
            out[f"validity_{name}"] = 1.0 - clf.predict_proba_class2(x1)
    return out
