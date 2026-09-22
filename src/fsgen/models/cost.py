"""科目ごとの調整コスト(ステージ9)。

現在の輸送コスト(低次元PCA空間の二乗ユークリッド距離)は全科目が等コストで、
「利益剰余金を280億円増やせ」のような直接操作不可能な指示や、社債ゼロの
企業に負の社債が生じるといった問題を生んでいた。ここでは科目ごとの
調整速度・調整コストの文献値(``conf/accounts/costs.yaml``)を使い、
元空間(基礎科目そのもの)での非対称重み付きコストに置き換える。

    c(x_i, y_j) = Σ_a ( w_a^+ max(y_ja - x_ia, 0)^2 + w_a^- max(x_ia - y_ja, 0)^2 )

x, y は総資産で正規化した値。w^+ は増加のコスト(cost_increase)、
w^- は減少のコスト(cost_decrease)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

_NON_ACTIONABLE_COST = 1e6  # 方式A(§1.3): infは数値計算で扱いにくいため巨大値で代用
_DEFAULT_PROJECTION_WEIGHT_FLOOR = 1e-3  # ステージ8のdelta(assets比)と同じ値
INDUSTRY_HARD_PENALTY = 1e10  # ステージ12 §2.1(A): infは数値計算で扱いにくいため大きな有限値


@dataclass(frozen=True)
class AccountCostSpec:
    name: str
    cost_increase: float
    cost_decrease: float
    source: str


@dataclass(frozen=True)
class CostConfig:
    accounts: tuple[AccountCostSpec, ...]
    non_actionable: tuple[str, ...]


def load_cost_config(path: Path) -> CostConfig:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    accounts = tuple(
        AccountCostSpec(
            name=a["name"],
            cost_increase=float(a["cost_increase"]),
            cost_decrease=float(a["cost_decrease"]),
            source=a["source"],
        )
        for a in doc["basis_costs"]
    )
    non_actionable = tuple(doc.get("non_actionable", ()))
    return CostConfig(accounts=accounts, non_actionable=non_actionable)


NonActionableMode = Literal["keep", "freeze", "ignore"]


def cost_vectors(
    config: CostConfig, names: tuple[str, ...], non_actionable_mode: NonActionableMode = "freeze"
) -> tuple[np.ndarray, np.ndarray]:
    """names順に (w_plus, w_minus) を返す。

    non_actionable科目の扱いは用途によって意味が逆になるため、呼び出し側が
    明示的に選ぶ:

    - ``"freeze"``(射影の重み §2.2 用、既定): コストを``_NON_ACTIONABLE_COST``
      (1e6)に上書きする。射影の目的関数では「この科目を始点から動かす代償」
      を測るため、コストが大きいほど始点付近に留まる(実質固定)。
    - ``"ignore"``(輸送コスト §2.1 用): コストを0にする。輸送コストでは
      「この科目の差」を対応先選びの基準にしてしまう意味になるため、1e6を
      使うと逆効果になる——企業iのnon_actionable値(例: 大きく負の利益剰余金)
      と大きく異なる(=健全な)対応先が「コストが高い」として避けられ、
      むしろ同じくらい不健全な対応先を選んでしまう(実測で確認、ステージ9
      報告参照)。non_actionableな科目は対応先選びの基準にしないという
      解釈のもと、コストを0にして選定から除外する。
    - ``"keep"``: costs.yamlの値をそのまま使う(比較・デバッグ用)。
    """
    by_name = {a.name: a for a in config.accounts}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"costs.yaml に定義のない科目: {missing}")

    w_plus = np.array([by_name[n].cost_increase for n in names], dtype=float)
    w_minus = np.array([by_name[n].cost_decrease for n in names], dtype=float)
    if non_actionable_mode != "keep":
        override = _NON_ACTIONABLE_COST if non_actionable_mode == "freeze" else 0.0
        non_actionable_set = set(config.non_actionable)
        for i, n in enumerate(names):
            if n in non_actionable_set:
                w_plus[i] = override
                w_minus[i] = override
    return w_plus, w_minus


def weighted_asymmetric_cost_matrix(
    x: np.ndarray, y: np.ndarray, w_plus: np.ndarray, w_minus: np.ndarray
) -> np.ndarray:
    """元空間での非対称重み付きコスト行列 (n0, n1)(§2.1)。x, y は正規化済み。

    科目ごとに(n0, n1)の差分行列を作って足し込む(D=30回のループ)。
    (n0, n1, D)の3次元配列を一度に確保しない実装にしてメモリを抑える。
    """
    n0 = x.shape[0]
    n1 = y.shape[0]
    dim = x.shape[1]
    cost = np.zeros((n0, n1))
    for a in range(dim):
        diff = y[:, a][None, :] - x[:, a][:, None]  # (n0, n1) = y_ja - x_ia
        cost += w_plus[a] * np.maximum(diff, 0.0) ** 2
        cost += w_minus[a] * np.maximum(-diff, 0.0) ** 2
    return cost


def low_dim_symmetric_cost_matrix(
    z0: np.ndarray, z1: np.ndarray, b: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """低次元近似のコスト行列(§2.1後半)。

    非対称(max(...,0))の扱いは低次元へは素直に持ち込めないため、対称重み
    w(呼び出し側で通常 max(w_plus, w_minus) を渡す)による二次形式
    (z_j-z_i)^T (B^T diag(w) B) (z_j-z_i) で近似する。方向非対称性が
    失われる点が、元空間版との既知の違い。
    """
    m_z = b.T @ (w[:, None] * b)  # (k, k)
    a0 = z0 @ m_z
    diag0 = np.sum(a0 * z0, axis=1)
    a1 = z1 @ m_z
    diag1 = np.sum(a1 * z1, axis=1)
    cross = z0 @ m_z @ z1.T
    cost = diag0[:, None] + diag1[None, :] - 2.0 * cross
    result: np.ndarray = np.maximum(cost, 0.0)  # 数値誤差での微小負値をクリップ
    return result


def combined_projection_weights(
    x_start: np.ndarray,
    assets: float,
    w_plus: np.ndarray,
    w_minus: np.ndarray,
    delta: float = _DEFAULT_PROJECTION_WEIGHT_FLOOR,
) -> np.ndarray:
    """射影の重み W = diag(max(w+,w-) / (|x_start|+delta*Assets))(§2.2)。

    方向(増加/減少)は射影の時点では確定しないため w=max(w+,w-) を使う。
    """
    w = np.maximum(w_plus, w_minus)
    floor = assets * delta
    result: np.ndarray = w / (np.abs(x_start) + floor)
    return result


def industry_mismatch(src_labels: np.ndarray, tgt_labels: np.ndarray) -> np.ndarray:
    """(n0, n1) の bool 行列。始点iと目標jの業種(またはブロック)が異なればTrue。"""
    result: np.ndarray = src_labels[:, None] != tgt_labels[None, :]
    return result


def soft_industry_cost(
    base_cost: np.ndarray, mismatch: np.ndarray, lam_multiple: float
) -> np.ndarray:
    """(B) ソフトペナルティ: c = C/median(C) + λ·1[業種不一致]、λ = lam_multiple × median。

    基本コストを中央値で正規化してから加えるので、正規化後の中央値は1で
    ``lam_multiple`` がそのまま「中央値の何倍か」になる。返り値は**正規化済み**
    (``transport.solve_ot_normalized`` に直接渡す)。lam_multiple=0 のとき
    ``C/median(C)`` そのもので、現行の ``solve_ot_from_cost`` が内部で作る
    正規化済みコストとビット単位で一致する(§2.1: λ=0が現行と一致)。
    """
    normalized: np.ndarray = base_cost / np.median(base_cost)
    if lam_multiple == 0.0:
        return normalized
    result: np.ndarray = normalized + lam_multiple * mismatch
    return result


def hard_industry_cost(
    base_cost: np.ndarray, mismatch: np.ndarray, penalty: float = INDUSTRY_HARD_PENALTY
) -> np.ndarray:
    """(A) ハード制約: 業種不一致のコストを penalty(既定1e10)にする。正規化済みで返す。

    中央値による正規化は**ペナルティを足す前の**基本コストで行う(不一致が
    大多数だと、足した後の中央値が1e10になってしまうため)。
    """
    normalized: np.ndarray = base_cost / np.median(base_cost)
    result: np.ndarray = np.where(mismatch, penalty, normalized)
    return result


__all__ = [
    "INDUSTRY_HARD_PENALTY",
    "AccountCostSpec",
    "CostConfig",
    "combined_projection_weights",
    "cost_vectors",
    "load_cost_config",
    "low_dim_symmetric_cost_matrix",
    "hard_industry_cost",
    "industry_mismatch",
    "soft_industry_cost",
    "weighted_asymmetric_cost_matrix",
]
