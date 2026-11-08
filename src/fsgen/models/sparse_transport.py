"""2 次正則化の片側不均衡最適輸送(スパース OT、ステージ79)。

Blondel et al. (2018) の 2 次正則化の考え方で、解そのものが厳密に 0 の成分をもつ輸送計画を求める。
行で正規化した p_ij = n0 P_ij(各行の和 1)について、次を解く。

    min_p  Σ_ij p_ij C_ij + (γ/2) Σ_ij p_ij² + (τ/2) Σ_j (r_j − 1)²
    s.t.   p_ij ≥ 0、Σ_j p_ij = 1、許されない組は p_ij = 0

r_j = (n1/n0) Σ_i p_ij は列の相対的な負荷(1 = 周辺分布 b = 1/n1 と釣り合う)。行の制約は厳密に、
列は 2 次の罰則で緩める(片側不均衡)。γ > 0 で強凸なので、解は一意である。

解法は、行ごとの単体への厳密な射影をもつ加速射影勾配法(FISTA)。再始動は勾配による判定
(O'Donoghue & Candès 2015): 外挿の向きが降下の向きと逆になったら運動量を捨てる。関数値による判定は、
最適解の近くで関数値の差が丸め誤差の大きさになると誤って再始動を繰り返すので使わない。射影の
結果は厳密に 0 の成分をもつので、事後のカットオフなしで候補の集合 {j : p_ij > 0} が決まる。
収束しなければ ``SolverNotConvergedError`` を出す。

POT・torch・sklearn に依存しない(numpy だけ)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsgen.errors import SolverNotConvergedError

_TOP_K = 64  # 射影で最初に調べる上位の列数(足りなければ全列で計算し直す)
_BLOCKED = -1e12  # 許されない組の値(射影で必ず 0 になる)


@dataclass(frozen=True)
class SparseOTResult:
    p: np.ndarray  # (n0, n1) 行で正規化した計画(各行の和 1)
    n_iter: int
    objective: float
    residual: float  # 不動点の残差 max |p − Π(p − ∇F/L)|


def project_rows_to_simplex(v: np.ndarray) -> np.ndarray:
    """各行を確率単体 {p ≥ 0、Σ p = 1} へユークリッド射影する(Duchi et al. 2008 の方法)。"""
    n, m = v.shape
    k = min(_TOP_K, m)
    top = -np.partition(-v, k - 1, axis=1)[:, :k] if k < m else v.copy()
    u = -np.sort(-top, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, k + 1)
    cond = u - css / ind > 0
    rho = k - np.argmax(cond[:, ::-1], axis=1)  # 条件を満たす最後の位置(1 始まり)
    theta = css[np.arange(n), rho - 1] / rho
    # 上位 k 列の外に正の成分が残る行は、全列で計算し直す
    redo = (rho == k) & (k < m)
    if redo.any():
        ur = -np.sort(-v[redo], axis=1)
        cr = np.cumsum(ur, axis=1) - 1.0
        indr = np.arange(1, m + 1)
        condr = ur - cr / indr > 0
        rr = m - np.argmax(condr[:, ::-1], axis=1)
        theta[redo] = cr[np.arange(int(redo.sum())), rr - 1] / rr
    out: np.ndarray = np.maximum(v - theta[:, None], 0.0)
    return out


def solve_sparse_semi_unbalanced(
    cost: np.ndarray,
    gamma: float,
    tau: float,
    *,
    allowed: np.ndarray | None = None,
    tol: float = 1e-11,
    rel_obj_tol: float = 1e-13,
    max_iter: int = 200_000,
    check_every: int = 25,
) -> SparseOTResult:
    """2 次正則化の片側不均衡 OT を解く。

    Args:
        cost: (n0, n1) コスト行列(中央値などで正規化したもの)。
        gamma: 2 次正則化の強さ γ(> 0)。大きいほど非ゼロの候補が増える。
        tau: 列の 2 次の罰則の強さ τ(≥ 0)。
        allowed: (n0, n1) の真偽値。False の組は p = 0 に固定する。各行に 1 つ以上の True が必要。
        tol: 不動点の残差の許容。
        rel_obj_tol: 目的関数の相対変化の許容(``check_every`` 回ごとに比べる)。
        max_iter: 反復の上限。

    Raises:
        ValueError: 引数が不正。
        SolverNotConvergedError: 上限までに収束しなかった。
    """
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2 or not np.isfinite(c).all():
        raise ValueError("cost は有限の 2 次元配列")
    if gamma <= 0 or tau < 0:
        raise ValueError("gamma > 0、tau ≥ 0")
    n0, n1 = c.shape
    if allowed is None:
        allowed = np.ones_like(c, dtype=bool)
    if allowed.shape != c.shape or not allowed.any(axis=1).all():
        raise ValueError("allowed の形が違うか、許される列のない行がある")
    kappa = n1 / n0
    lip = gamma + tau * kappa ** 2 * n0

    def grad(p: np.ndarray) -> np.ndarray:
        r = kappa * p.sum(axis=0)
        out: np.ndarray = c + gamma * p + tau * kappa * (r - 1.0)[None, :]
        return out

    def obj(p: np.ndarray) -> float:
        r = kappa * p.sum(axis=0)
        return float((p * c).sum() + 0.5 * gamma * (p ** 2).sum()
                     + 0.5 * tau * ((r - 1.0) ** 2).sum())

    def prox(v: np.ndarray) -> np.ndarray:
        return project_rows_to_simplex(np.where(allowed, v, _BLOCKED))

    p = prox(-c / gamma)  # 列の罰則を除いた各行の解(sparsemax)から始める
    y, t = p.copy(), 1.0
    f_check = obj(p)
    for it in range(1, max_iter + 1):
        p_new = prox(y - grad(y) / lip)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        if float(np.vdot(y - p_new, p_new - p)) > 0:  # 勾配による再始動
            y, t = p_new.copy(), 1.0
        else:
            y, t = p_new + ((t - 1.0) / t_new) * (p_new - p), t_new
        p = p_new
        if it % check_every == 0:
            f_p = obj(p)
            res = float(np.abs(p - prox(p - grad(p) / lip)).max())
            rel = abs(f_check - f_p) / max(abs(f_p), 1.0)
            f_check = f_p
            if res <= tol and rel <= rel_obj_tol:
                return SparseOTResult(p, it, f_p, res)
    res = float(np.abs(p - prox(p - grad(p) / lip)).max())
    raise SolverNotConvergedError(
        f"スパース OT が {max_iter} 回で収束しない(残差 {res:.3e}、許容 {tol:.1e})")


__all__ = ["SparseOTResult", "project_rows_to_simplex", "solve_sparse_semi_unbalanced"]
