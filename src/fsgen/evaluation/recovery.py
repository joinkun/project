"""再建の度合いを測る指標(ステージ25)。

これまでの主指標(|総資産変化|・validity)は「改善したか」を測っていなかった。ここでは会計的な
再建の定義そのものを測る。いずれも x は総資産で正規化した状態ベクトル(tilde 空間)、
x0 が始点、x1 が終点。A0・A1 は始点・終点の総資産。

1. 純資産の改善: (NA(1) - NA(0)) / A0(大きいほど良い)
2. 債務超過の解消: 始点で NA < 0 の行のうち、終点で NA >= 0 になった行(率は呼び出し側で集計)
3. 有利子負債の圧縮: (D(0) - D(1)) / A0。D = 短期借入金(CP・1 年内返済予定の長期借入金を含む)
   + 長期借入金 + 社債(流動・固定)。転換社債型新株予約権付社債は残余科目にあるため含まない
4. 自己資本比率の改善: NA(1)/A1 - NA(0)/A0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

INTEREST_BEARING_DEBT: tuple[str, ...] = (
    "short_term_loans",
    "long_term_loans",
    "bonds_current",
    "bonds_noncurrent",
)
_ASSETS = "jppfs_cor_Assets"
_NET_ASSETS = "jppfs_cor_NetAssets"


@dataclass(frozen=True)
class RecoveryMetrics:
    net_asset_improvement: np.ndarray  # (n,)
    insolvent_start: np.ndarray  # (n,) bool: 始点で NA < 0
    insolvency_resolved: np.ndarray  # (n,) bool: 始点で NA < 0 かつ終点で NA >= 0
    debt_reduction: np.ndarray  # (n,)
    equity_ratio_change: np.ndarray  # (n,)


def recovery_metrics(
    x0: np.ndarray,
    x1: np.ndarray,
    members: Mapping[str, np.ndarray],
    names: tuple[str, ...],
) -> RecoveryMetrics:
    """行ごとの再建指標。``members`` は ``subtotal_members`` の戻り値(合計・小計の係数)。"""
    if x0.shape != x1.shape or x0.ndim != 2 or x0.shape[1] != len(names):
        raise ValueError(f"shape mismatch: x0 {x0.shape}, x1 {x1.shape}, names {len(names)}")
    a0 = x0 @ members[_ASSETS]
    a1 = x1 @ members[_ASSETS]
    if np.any(a0 <= 0) or np.any(a1 <= 0):
        raise ValueError("total assets must be positive at both ends")
    na0 = x0 @ members[_NET_ASSETS]
    na1 = x1 @ members[_NET_ASSETS]
    missing = [n for n in INTEREST_BEARING_DEBT if n not in names]
    if missing:
        raise ValueError(f"interest-bearing debt accounts missing from names: {missing}")
    idx = [names.index(n) for n in INTEREST_BEARING_DEBT]
    d0 = x0[:, idx].sum(axis=1)
    d1 = x1[:, idx].sum(axis=1)
    insolvent = na0 < 0
    return RecoveryMetrics(
        net_asset_improvement=(na1 - na0) / a0,
        insolvent_start=insolvent,
        insolvency_resolved=insolvent & (na1 >= 0),
        debt_reduction=(d0 - d1) / a0,
        equity_ratio_change=na1 / a1 - na0 / a0,
    )


def net_asset_decomposition(
    x0: np.ndarray,
    x1: np.ndarray,
    members: Mapping[str, np.ndarray],
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """純資産の変化を、純資産の部の基礎科目ごとの (x1 - x0) / A0 に分解する。

    対象の科目は ``members["jppfs_cor_NetAssets"]`` の係数が 0 でない列。各成分の和は
    ``recovery_metrics(...).net_asset_improvement`` に一致する。
    """
    a0 = x0 @ members[_ASSETS]
    coef = members[_NET_ASSETS]
    return {
        names[j]: coef[j] * (x1[:, j] - x0[:, j]) / a0 for j in np.flatnonzero(coef != 0)
    }
