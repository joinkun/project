"""subspace.py のテスト。合成データで検証する(実データ非依存)。"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.subspace import build_reduced_basis, reconstruction_error

# 単純な制約: x[0] + x[1] - x[2] = 0 (D=4, K=1)。
_C = np.array([[1.0], [1.0], [-1.0], [0.0]])


def _make_constrained_data(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    d = rng.normal(size=n)
    c = a + b  # 制約 a + b - c = 0 を満たす
    return np.stack([a, b, c, d], axis=1)


def test_build_reduced_basis_columns_orthonormal() -> None:
    x = _make_constrained_data(50)
    basis = build_reduced_basis(_C, x, k=2)
    assert basis.shape == (4, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-8)


def test_build_reduced_basis_satisfies_constraint() -> None:
    x = _make_constrained_data(50)
    basis = build_reduced_basis(_C, x, k=2)
    assert np.allclose(_C.T @ basis, 0.0, atol=1e-8)


def test_build_reduced_basis_full_rank_reconstructs_exactly() -> None:
    """k = D - rank(C) (=3) なら null(C^T) を完全に張るので再構成誤差は0。"""
    x = _make_constrained_data(50)
    basis = build_reduced_basis(_C, x, k=3)
    error = reconstruction_error(basis, x)
    assert np.allclose(error, 0.0, atol=1e-8)


def test_reconstruction_error_decreases_with_k() -> None:
    x = _make_constrained_data(100)
    err_k1 = reconstruction_error(build_reduced_basis(_C, x, k=1), x)
    err_k2 = reconstruction_error(build_reduced_basis(_C, x, k=2), x)
    err_k3 = reconstruction_error(build_reduced_basis(_C, x, k=3), x)
    assert err_k1.mean() >= err_k2.mean() >= err_k3.mean() - 1e-9


def test_reconstruction_error_zero_row_is_zero() -> None:
    x = np.zeros((1, 4))
    basis = build_reduced_basis(_C, _make_constrained_data(50), k=2)
    error = reconstruction_error(basis, x)
    assert error[0] == 0.0
