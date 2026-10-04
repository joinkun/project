"""ゼロを明示的に扱う健全企業の密度(ステージ52)。

版 I(``IndepZeroGaussian``): 科目ごとに独立なゼロ確率 π_a と、null(Cᵀ) の座標 z = Uᵀx の正規
N(μ, Σ) を、x_S = 0(U_S z = 0)で条件付けた連続部分の積。条件付けは零空間の正規直交基底 N の
座標 w(z = N w)で行い、精度 P = NᵀΣ⁻¹N、平均 m = P⁻¹NᵀΣ⁻¹μ の正規になる(次元 |S̄| − 1)。

版 II(``PatternMixture``): 一部の科目(ゼロ率の低い科目)でのゼロパターン S ごとに、出現確率 π_S と
非 0 の科目の正規 q_S を持つ混合。上位に入らないパターンはハミング距離が最小のパターンで評価する。

共分散の推定(Ledoit-Wolf など)は呼び出し側で行い、平均と共分散を渡す(このモジュールは numpy のみ)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def null_basis(a: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """a (k, d) の零空間の正規直交基底 (d, d − rank)。k = 0 なら単位行列。"""
    d = a.shape[1]
    if a.shape[0] == 0:
        return np.eye(d)
    _u, s, vt = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(s > tol * max(s[0], 1.0)))
    out: np.ndarray = vt[rank:].T
    return out


def gaussian_logpdf(w: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """行ごとの log N(w; mean, cov)。"""
    ch = np.linalg.cholesky(cov)
    sol = np.linalg.solve(ch, (np.atleast_2d(w) - mean).T)
    out: np.ndarray = (-0.5 * np.sum(sol**2, axis=0) - np.sum(np.log(np.diag(ch)))
                       - 0.5 * len(mean) * np.log(2 * np.pi))
    return out


def _restricted(mu: np.ndarray, prec: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N(μ, Σ) を z = N w の部分空間に制限した w の平均と共分散。"""
    p = n.T @ prec @ n
    cov = np.linalg.inv(p)
    cov = 0.5 * (cov + cov.T)
    return cov @ (n.T @ prec @ mu), cov


@dataclass
class IndepZeroGaussian:
    pi0: np.ndarray  # (D,) P(x_a = 0)
    u: np.ndarray  # (D, D−1) null(Cᵀ) の正規直交基底
    mu: np.ndarray  # (D−1,)
    cov: np.ndarray  # (D−1, D−1)

    def __post_init__(self) -> None:
        self.prec = np.linalg.inv(self.cov)

    def discrete_logp(self, zero: np.ndarray) -> np.ndarray:
        out: np.ndarray = (zero * np.log(self.pi0) + (~zero) * np.log1p(-self.pi0)).sum(axis=1)
        return out

    def cont_logpdf(self, x: np.ndarray) -> np.ndarray:
        zero = x == 0
        z = x @ self.u
        out = np.empty(len(x))
        keys = [r.tobytes() for r in zero]
        for k in set(keys):
            rows = np.array([i for i, kk in enumerate(keys) if kk == k])
            s = zero[rows[0]]
            n = null_basis(self.u[s])
            mean, cov = _restricted(self.mu, self.prec, n)
            out[rows] = gaussian_logpdf(z[rows] @ n, mean, cov)
        return out

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.discrete_logp(x == 0) + self.cont_logpdf(x)
        return out

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """パターンを独立に引き、条件付き正規から連続部分を引く(ゼロの科目はちょうど 0 にする)。"""
        zero = rng.random((n_samples, len(self.pi0))) < self.pi0
        out = np.zeros((n_samples, len(self.pi0)))
        keys = [r.tobytes() for r in zero]
        for k in set(keys):
            rows = np.array([i for i, kk in enumerate(keys) if kk == k])
            s = zero[rows[0]]
            n = null_basis(self.u[s])
            mean, cov = _restricted(self.mu, self.prec, n)
            w = rng.multivariate_normal(mean, cov, size=len(rows))
            x = (w @ n.T) @ self.u.T
            x[:, s] = 0.0
            out[rows] = x
        return out


@dataclass
class PatternMixture:
    cols: np.ndarray  # 対象の科目(状態ベクトルでの位置)
    patterns: np.ndarray  # (M, len(cols)) bool、True = 0
    pi: np.ndarray  # (M,)
    means: list[np.ndarray]  # パターンごとの非 0 科目の平均
    covs: list[np.ndarray]

    def assign(self, zero: np.ndarray) -> np.ndarray:
        """各行のパターン(ちょうど一致、なければハミング距離最小、同距離は π の大きい方)。"""
        dist = (zero[:, None, :] != self.patterns[None, :, :]).sum(axis=2)
        key = dist - 1e-9 * self.pi[None, :]
        out: np.ndarray = np.argmin(key, axis=1)
        return out

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        xs = x[:, self.cols]
        k = self.assign(xs == 0)
        out = np.empty(len(x))
        for m in range(len(self.pi)):
            rows = np.where(k == m)[0]
            if len(rows) == 0:
                continue
            nz = ~self.patterns[m]
            out[rows] = np.log(self.pi[m]) + gaussian_logpdf(xs[rows][:, nz], self.means[m],
                                                             self.covs[m])
        return out

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """(n, len(cols)) の値。パターンを π で引き、非 0 の科目を q_S から引く。"""
        k = rng.choice(len(self.pi), size=n_samples, p=self.pi / self.pi.sum())
        out = np.zeros((n_samples, len(self.cols)))
        for m in range(len(self.pi)):
            rows = np.where(k == m)[0]
            nz = ~self.patterns[m]
            out[np.ix_(rows, np.where(nz)[0])] = rng.multivariate_normal(
                self.means[m], self.covs[m], size=len(rows))
        return out
