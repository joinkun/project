"""部分調整モデルによる科目ごとの調整速度 λ_a の推定(ステージ21)。

総資産で正規化した科目 a について

    x_{a,t}/A_t - x_{a,t-1}/A_{t-1} = λ_a (x*_{a,t}/A_t - x_{a,t-1}/A_{t-1}) + ε

を推定し、``adjustment_cost`` で c_a = (1 - λ_a) / λ_a に変換して ``costs.yaml`` の
重みの根拠にする。単年断面では調整コストとショックの変動性を分離できないため
(Asker, Collard-Wexler & De Loecker 2014)、複数期のパネルが必要になる。

**現時点では骨格のみ。** 推定方法(固定効果か差分 GMM か)は、バランスパネルの
社数が確定してから決める(ステージ21-A の結果次第。3,000 社を切るなら固定効果に
落とす判断もありうる)。

パネルの形式(``build_panel_stage21.py`` の ``panel_long.csv`` に総資産列を足したもの):
1 行 = 1 社 × 1 決算年。列は ``edinetCode``、``year``、``assets`` と各科目(円)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Method = Literal["fe", "diff_gmm"]
Direction = Literal["both", "positive", "negative"]

PANEL_KEY_COLUMNS: tuple[str, ...] = ("edinetCode", "year", "assets")

_METHODS: tuple[Method, ...] = ("fe", "diff_gmm")
_DIRECTIONS: tuple[Direction, ...] = ("both", "positive", "negative")


@dataclass(frozen=True)
class AdjustmentResult:
    """1 科目・1 方向の推定結果。"""

    account: str
    method: Method
    direction: Direction
    lam: float  # 調整速度 λ_a
    se: float  # λ_a の標準誤差
    ci_low: float
    ci_high: float
    n_firms: int
    n_obs: int
    # 差分 GMM の診断(AR(2) 検定・Hansen 検定の p 値)。固定効果では None。
    ar2_p: float | None = None
    hansen_p: float | None = None

    @property
    def cost(self) -> float:
        return adjustment_cost(self.lam)


def adjustment_cost(lam: float) -> float:
    """調整速度 λ から調整コストの重み c = (1 - λ) / λ に変換する。

    λ ∈ (0, 1] を前提とする(λ=1 は即時調整でコスト 0)。範囲外は推定の失敗か
    モデルの誤特定を意味するので、黙って丸めずに例外にする。
    """
    if not math.isfinite(lam) or not 0.0 < lam <= 1.0:
        raise ValueError(f"adjustment speed must be in (0, 1], got {lam!r}")
    return (1.0 - lam) / lam


def validate_panel(panel: pd.DataFrame, account: str) -> None:
    """推定に必要な列があり、(edinetCode, year) が一意であることを確かめる。"""
    missing = [c for c in (*PANEL_KEY_COLUMNS, account) if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")
    if panel.duplicated(subset=["edinetCode", "year"]).any():
        raise ValueError("panel has duplicated (edinetCode, year) rows")


def estimate_adjustment_speed(
    panel: pd.DataFrame,
    account: str,
    method: Method,
    direction: Direction = "both",
) -> AdjustmentResult:
    """科目 ``account`` の調整速度 λ を推定する(未実装)。

    ``direction`` は増加(positive)・減少(negative)を分けて推定するためのもの。
    ``costs.yaml`` の非対称な重みに対応させる。
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method: {method!r}")
    if direction not in _DIRECTIONS:
        raise ValueError(f"unknown direction: {direction!r}")
    validate_panel(panel, account)
    raise NotImplementedError(
        "推定方法はパネルの社数の確定後に実装する(ステージ21-A の結果待ち)"
    )
