"""基準単価 θ と、企業ごとの非対称 L1 コスト ψ_i(ステージ82)。

再建企業のプールの変化から、勘定科目・向きごとの基準単価 θ_a^± を 5 方式で推定し、企業 i ごとの
単価 θ_ia^± に直す。第 1 段階(輸送のコスト)と第 2 段階(QP の移動コスト)で同じ ψ_i を使う。

    ψ_i(y − x_i) = θ_i⁺ · (y − x_i)₊ + θ_i⁻ · (x_i − y)₊

numpy だけに依存する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THETA_MAX = 1e3
INACTIVE_ZERO = 1e-6
MODES = ("R1", "KDE0", "IPW", "scale", "A1")
KDE_MIN_N = 5
# 始点ゼロのときに新規の発生を抑える科目(非負の資産・負債と非支配株主持分。現金・株主資本・
# 残余科目・貸倒引当金は含めない)
INACTIVE_CANDIDATES = (
    "inventories", "receivables_trade", "jppfs_cor_OtherCA",
    "jppfs_cor_PropertyPlantAndEquipment", "jppfs_cor_IntangibleAssets",
    "jppfs_cor_InvestmentsAndOtherAssets", "short_term_loans", "jppfs_cor_IncomeTaxesPayable",
    "jppfs_cor_OtherCL", "payables_trade", "accounts_payable_other", "bonds_current",
    "long_term_loans", "bonds_noncurrent", "non_controlling_interests",
)


@dataclass(frozen=True)
class ThetaResult:
    theta: np.ndarray  # (2, D)、行 0 = +(増やす)、行 1 = −(減らす)
    n_movers: np.ndarray  # (2, D)
    n_fallback: int  # KDE0 で R1 に戻した方向の数


def kde_density_at_zero(z: np.ndarray) -> float:
    """正の値 z の密度の t = 0 での推定(鏡像法、Silverman の頑健な帯域)。帯域 0 なら 0 を返す。"""
    n = len(z)
    both = np.concatenate([z, -z])
    s = float(both.std(ddof=1))
    q75, q25 = np.percentile(both, [75, 25])
    spread = min(s, float(q75 - q25) / 1.349) if q75 > q25 else s
    h = 0.9 * spread * (2 * n) ** (-0.2)
    if not h > 0:
        return 0.0
    return float((2.0 * np.exp(-0.5 * (z / h) ** 2) / (h * np.sqrt(2 * np.pi))).sum() / n)


def estimate_theta(delta: np.ndarray, x0: np.ndarray, sigma: np.ndarray, mode: str,
                   theta_max: float = THETA_MAX) -> ThetaResult:
    """プール(K 社)の変化 delta と始点 x0(どちらも (K, D)、総資産比)から θ_a^± を推定する。"""
    if mode not in MODES:
        raise ValueError(f"mode は {MODES} のどれか: {mode!r}")
    n_pool, dim = delta.shape
    z = np.abs(delta) / (np.abs(x0) + sigma[None, :])
    theta = np.full((2, dim), theta_max)
    n_mv = np.zeros((2, dim), dtype=int)
    n_fb = 0
    for r, sg in enumerate((1.0, -1.0)):
        for j in range(dim):
            mv = sg * delta[:, j] > 0
            n_mv[r, j] = int(mv.sum())
            if not mv.any():
                continue
            zz = z[mv, j]
            pi = mv.sum() / n_pool
            m = float(np.median(zz))
            if mode == "R1" or mode == "KDE0":
                val = (np.log(2.0) / m) / pi if m > 0 else theta_max
                if mode == "KDE0":
                    f0 = kde_density_at_zero(zz) if len(zz) >= KDE_MIN_N else 0.0
                    if f0 > 0:
                        val = f0 / pi
                    else:
                        n_fb += 1
            elif mode == "IPW":
                val = 1.0 / pi
            elif mode == "scale":
                val = 1.0 / m if m > 0 else theta_max
            else:
                om = float(np.minimum(zz, m).mean())
                val = 1.0 / (pi * om) if om > 0 else theta_max
            theta[r, j] = min(val, theta_max)
    return ThetaResult(theta, n_mv, n_fb)


def individual_theta(theta: np.ndarray, x0: np.ndarray, sigma: np.ndarray, *,
                     indiv_scale: bool, inactive: np.ndarray | None = None,
                     theta_max: float = THETA_MAX) -> np.ndarray:
    """企業 i の単価 θ_i(2, D)。inactive の科目は θ_i⁺ = θ_max(始点ゼロの新規発生を抑える)。"""
    out = theta.copy()
    if indiv_scale:
        out = np.minimum(out / (np.abs(x0) + sigma)[None, :], theta_max)
    if inactive is not None:
        out[0, inactive] = theta_max
    return out


def inactive_mask(x0: np.ndarray, names: tuple[str, ...], *,
                  candidates: tuple[str, ...] = INACTIVE_CANDIDATES,
                  tol: float = INACTIVE_ZERO) -> np.ndarray:
    """A_inact(i): 候補の科目のうち、始点で |x| < tol のもの((D,) の真偽値)。"""
    cand = np.array([n in candidates for n in names])
    out: np.ndarray = cand & (np.abs(x0) < tol)
    return out


def psi(theta_i: np.ndarray, d: np.ndarray) -> np.ndarray:
    """ψ_i(d) を最後の軸で計算する。theta_i は (2, D) か、行ごとの (…, 2, D)。"""
    tp, tm = theta_i[..., 0, :], theta_i[..., 1, :]
    out: np.ndarray = (tp * np.maximum(d, 0.0) + tm * np.maximum(-d, 0.0)).sum(axis=-1)
    return out


def pairwise_cost(xs: np.ndarray, xt: np.ndarray, theta_rows: np.ndarray,
                  chunk: int = 16) -> np.ndarray:
    """C_ij = ψ_i(x_j − x_i)。theta_rows は (n0, 2, D)(始点ごとの単価)。"""
    out = np.empty((len(xs), len(xt)))
    for s in range(0, len(xs), chunk):
        d = xt[None, :, :] - xs[s:s + chunk, None, :]
        th = theta_rows[s:s + chunk]
        out[s:s + chunk] = (th[:, None, 0, :] * np.maximum(d, 0.0)
                            + th[:, None, 1, :] * np.maximum(-d, 0.0)).sum(axis=2)
    return out


def emergence_rate(x0: np.ndarray, y: np.ndarray, names: tuple[str, ...],
                   threshold: float) -> float:
    """始点ゼロ科目(A_inact)のうち、終点で |y| ≥ threshold になった組の割合。"""
    mask = np.array([inactive_mask(row, names) for row in x0])
    if not mask.any():
        return float("nan")
    return float((np.abs(y)[mask] >= threshold).mean())


__all__ = ["INACTIVE_CANDIDATES", "MODES", "THETA_MAX", "ThetaResult", "emergence_rate",
           "estimate_theta", "inactive_mask", "individual_theta", "kde_density_at_zero",
           "pairwise_cost", "psi"]
