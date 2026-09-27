"""目標分布(輸送の対応先の候補)を自己資本比率で絞る(ステージ26)。

ステージ25 で、旧重みの効果の正体が「資本の厚い健全企業を対応先に選ぶ」偏りだったと分かった。
この性質を、コスト重みではなく目標の定義として明示する。自己資本比率は**対応先の選択にだけ**使い、
生成対象・出力は勘定科目のまま。

案(x は総資産で正規化した状態ベクトル、比率 = 純資産 / 総資産):
- ``baseline``: class 0 全体(現行)
- ``relative``: 始点 i ごとに、比率が始点より高い対応先 j だけ(比率_j > 比率_i)
- ``median``: class 0 のうち比率が class 0 の中央値以上の企業だけ(始点によらない)
- ``absolute``: 比率が閾値を超える企業だけ(比率_j > 閾値)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

TargetFilter = Literal["baseline", "relative", "median", "absolute"]
TARGET_FILTERS: tuple[TargetFilter, ...] = ("baseline", "relative", "median", "absolute")
DISALLOWED_PENALTY = 1e10  # 業種のハード制約(cost.INDUSTRY_HARD_PENALTY)と同じ大きさ


def equity_ratio(x: np.ndarray, members: Mapping[str, np.ndarray]) -> np.ndarray:
    """純資産 / 総資産(x は tilde 空間でも円建てでもよい)。"""
    assets = x @ members["jppfs_cor_Assets"]
    if np.any(assets <= 0):
        raise ValueError("total assets must be positive")
    result: np.ndarray = (x @ members["jppfs_cor_NetAssets"]) / assets
    return result


@dataclass(frozen=True)
class FilterResult:
    allowed: np.ndarray  # (n_src, n_tgt) bool: 絞り込み後の候補(フォールバック込み)
    n_candidates: np.ndarray  # (n_src,) 絞り込みで残った候補数(フォールバック前)
    fallback: np.ndarray  # (n_src,) bool: 候補が 0 のため全体に戻した始点
    threshold: float | None  # median / absolute で使った閾値


def target_mask(
    kind: TargetFilter,
    src_ratio: np.ndarray,
    tgt_ratio: np.ndarray,
    absolute_threshold: float = 0.3,
) -> FilterResult:
    """始点 × 対応先の候補マスク。候補が 0 の始点は全候補に戻し、``fallback`` に記録する。"""
    n_src, n_tgt = len(src_ratio), len(tgt_ratio)
    threshold: float | None = None
    if kind == "baseline":
        allowed = np.ones((n_src, n_tgt), dtype=bool)
    elif kind == "relative":
        allowed = tgt_ratio[None, :] > src_ratio[:, None]
    elif kind == "median":
        threshold = float(np.median(tgt_ratio))
        allowed = np.broadcast_to(tgt_ratio >= threshold, (n_src, n_tgt)).copy()
    elif kind == "absolute":
        threshold = absolute_threshold
        allowed = np.broadcast_to(tgt_ratio > threshold, (n_src, n_tgt)).copy()
    else:
        raise ValueError(f"unknown target filter: {kind!r}")
    n_candidates = allowed.sum(axis=1)
    fallback = n_candidates == 0
    allowed[fallback] = True
    return FilterResult(allowed, n_candidates, fallback, threshold)


def apply_mask(normalized_cost: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """候補外のコストを ``DISALLOWED_PENALTY`` にする(正規化は呼び出し側で先に済ませる)。"""
    if normalized_cost.shape != allowed.shape:
        raise ValueError(f"shape mismatch: {normalized_cost.shape} vs {allowed.shape}")
    result: np.ndarray = np.where(allowed, normalized_cost, DISALLOWED_PENALTY)
    return result
