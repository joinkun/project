"""変化率の分布に由来する輸送コスト(ステージ50)。

科目ごとに、始点 x(総資産比)から目標 y(総資産比)への移動の負の対数尤度
−log P(y_a | x_a) をコストとし、科目間で独立とみなして足す(会計恒等式により科目は従属するので近似)。

    x ≠ 0、y が同符号の非 0: −log P(継続 | x ≠ 0) − log p_a(log(|y| / |x|))
    x ≠ 0、y が逆符号:       −log P(符号反転 | x ≠ 0)
    x ≠ 0、y = 0:            −log P(消滅 | x ≠ 0)
    x = 0、y ≠ 0:            −log P(新規計上 | x = 0)
    x = 0、y = 0:            −log P(0 のまま | x = 0)

p_a は g = log R の密度(ステージ49 の当てはめ)。y の密度ではない(y の密度にすると
−log|y| の項が加わり、保有の小さい目標ほど尤度が高くなる)。確率は条件付き
(始点の状態ごとの件数で割る)で、``pi_min`` で下からクリップする(観測 0 件の区分を有限にするため)。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.special import gammaln

Family = Literal["normal", "subbotin", "t"]


@dataclass(frozen=True)
class AccountModel:
    """1 科目の確率モデル。shape は Subbotin の b、t の自由度(正規では使わない)。"""

    family: Family
    loc: float
    scale: float
    shape: float
    p_stay0: float  # P(y = 0 | x = 0)
    p_birth: float  # P(y ≠ 0 | x = 0)
    p_cont: float  # P(y が同符号の非 0 | x ≠ 0)
    p_death: float  # P(y = 0 | x ≠ 0)
    p_flip: float  # P(y が逆符号 | x ≠ 0)


def conditional_probs(
    n_both0: int, n_new: int, n_gone: int, n_both: int, n_flip: int, pi_min: float
) -> dict[str, float]:
    """区分の件数 → 始点の状態ごとの条件付き確率(``pi_min`` 未満は ``pi_min`` に上げる)。

    始点の状態の件数が 0 のときは、その状態の区分をすべて ``pi_min`` にする
    (目標による差をつけない)。
    """
    if min(n_both0, n_new, n_gone, n_both, n_flip) < 0 or not 0 < pi_min < 1:
        raise ValueError("件数は非負、pi_min は (0, 1)")
    z = n_both0 + n_new
    nz = n_gone + n_both + n_flip

    def p(k: int, total: int) -> float:
        return max(k / total, pi_min) if total > 0 else pi_min

    return {"p_stay0": p(n_both0, z), "p_birth": p(n_new, z), "p_cont": p(n_both, nz),
            "p_death": p(n_gone, nz), "p_flip": p(n_flip, nz)}


def neg_log_density(m: AccountModel, g: np.ndarray) -> np.ndarray:
    """g = log R の負の対数密度(正規化定数を含む)。"""
    if m.scale <= 0:
        raise ValueError(f"scale は正: {m.scale}")
    z = (g - m.loc) / m.scale
    if m.family == "normal":
        out: np.ndarray = 0.5 * np.log(2 * np.pi) + np.log(m.scale) + 0.5 * z * z
    elif m.family == "subbotin":
        b = m.shape
        out = -(np.log(b) - np.log(2 * m.scale) - gammaln(1 / b)) + np.abs(z) ** b
    elif m.family == "t":
        nu = m.shape
        log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(m.scale)
        out = -log_c + (nu + 1) / 2 * np.log1p(z * z / nu)
    else:
        raise ValueError(m.family)
    return out


def account_cost(m: AccountModel, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 科目のコスト行列 (len(x), len(y))。x・y は総資産比。"""
    xs, ys = x[:, None], y[None, :]
    x0, y0 = xs == 0, ys == 0
    same = ~x0 & ~y0 & (np.sign(xs) == np.sign(ys))
    flip = ~x0 & ~y0 & ~same
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(same, np.log(np.abs(ys) / np.where(x0, 1.0, np.abs(xs))), 0.0)
    cont = -np.log(m.p_cont) + neg_log_density(m, g)
    out: np.ndarray = np.select(
        [x0 & y0, x0 & ~y0, ~x0 & y0, flip],
        [-np.log(m.p_stay0), -np.log(m.p_birth), -np.log(m.p_death), -np.log(m.p_flip)],
        default=cont,
    )
    return out


def likelihood_cost_matrix(
    x: np.ndarray, y: np.ndarray, models: list[AccountModel | None]
) -> np.ndarray:
    """c(x_i, y_j) = Σ_a c_a(x_ia, y_ja)(科目間で独立とみなす)。モデルが None の科目はコスト 0。"""
    if x.shape[1] != len(models) or y.shape[1] != len(models):
        raise ValueError("科目数がモデルの数と合わない")
    cost = np.zeros((x.shape[0], y.shape[0]))
    for a, m in enumerate(models):
        if m is not None:
            cost += account_cost(m, x[:, a], y[:, a])
    if not np.isfinite(cost).all():
        raise ValueError("コストに非有限値がある")
    return cost


def scale_period(m: AccountModel, years: float) -> AccountModel:
    """独立増分とみなして期間を伸ばす: 位置 × years、尺度 × √years(離散部分は変えない)。"""
    return replace(m, loc=m.loc * years, scale=m.scale * float(np.sqrt(years)))
