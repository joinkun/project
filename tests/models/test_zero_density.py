"""ゼロを明示的に扱う密度(ステージ52)のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fsgen.models.zero_density import (
    IndepZeroGaussian,
    PatternMixture,
    gaussian_logpdf,
    null_basis,
)


def _model(rng: np.random.Generator, d: int = 5) -> IndepZeroGaussian:
    c = np.ones((d, 1))
    c[3:] = -1.0  # 資産 3 = 負債・純資産 2 のような恒等式
    u = null_basis(c.T)
    a = rng.normal(size=(d - 1, d - 1))
    return IndepZeroGaussian(np.full(d, 0.2), u, rng.normal(size=d - 1), a @ a.T + np.eye(d - 1))


def test_null_basis_orthonormal() -> None:
    a = np.array([[1.0, 1.0, -1.0, 0.0]])
    n = null_basis(a)
    assert n.shape == (4, 3)
    np.testing.assert_allclose(a @ n, 0.0, atol=1e-12)
    np.testing.assert_allclose(n.T @ n, np.eye(3), atol=1e-12)


def test_gaussian_logpdf_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    cov, mean = a @ a.T + np.eye(3), rng.normal(size=3)
    w = rng.normal(size=(4, 3))
    np.testing.assert_allclose(gaussian_logpdf(w, mean, cov),
                               stats.multivariate_normal(mean, cov).logpdf(w))


def test_no_zero_pattern_is_plain_gaussian() -> None:
    rng = np.random.default_rng(1)
    m = _model(rng)
    x = rng.normal(size=(3, 4)) @ m.u.T  # 恒等式を満たし、0 を含まない
    want = stats.multivariate_normal(m.mu, m.cov).logpdf(x @ m.u)
    np.testing.assert_allclose(m.cont_logpdf(x), want)


def test_conditional_matches_covariance_formula() -> None:
    """U_S z = 0 での条件付けを、共分散による条件付きの公式と比べる。"""
    rng = np.random.default_rng(2)
    m = _model(rng)
    s = np.array([True, False, False, False, False])
    a = m.u[s]
    n = null_basis(a)
    w = rng.normal(size=(2, n.shape[1]))
    x = (w @ n.T) @ m.u.T
    x[:, s] = 0.0
    k = m.cov @ a.T @ np.linalg.inv(a @ m.cov @ a.T)
    mu_c = m.mu - k @ (a @ m.mu)
    cov_c = m.cov - k @ a @ m.cov
    want = stats.multivariate_normal(n.T @ mu_c, n.T @ cov_c @ n).logpdf(w)
    np.testing.assert_allclose(m.cont_logpdf(x), want, rtol=1e-8)


def test_sample_respects_zeros_and_identity() -> None:
    rng = np.random.default_rng(3)
    m = _model(rng)
    x = m.sample(200, rng)
    c = np.array([1, 1, 1, -1, -1.0])
    np.testing.assert_allclose(x @ c, 0.0, atol=1e-9)
    assert 0.1 < np.mean(x == 0) < 0.3


def test_pattern_mixture_assign_and_logpdf() -> None:
    pats = np.array([[False, False, False], [True, False, False]])
    mix = PatternMixture(np.array([0, 2, 3]), pats, np.array([0.7, 0.3]),
                         [np.zeros(3), np.zeros(2)], [np.eye(3), np.eye(2)])
    zero = np.array([[False, False, False], [True, False, False], [True, True, False]])
    np.testing.assert_array_equal(mix.assign(zero), [0, 1, 1])
    x = np.array([[1.0, 9.0, 0.5, 0.2], [0.0, 9.0, 0.5, 0.2]])
    lp = mix.logpdf(x)
    assert lp[0] == pytest.approx(np.log(0.7) + stats.multivariate_normal(np.zeros(3)).logpdf(
        [1.0, 0.5, 0.2]))
    assert lp[1] == pytest.approx(np.log(0.3) + stats.multivariate_normal(np.zeros(2)).logpdf(
        [0.5, 0.2]))
