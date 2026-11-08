"""2 次正則化の片側不均衡 OT(ステージ79)のテスト。"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from fsgen.errors import SolverNotConvergedError
from fsgen.models.sparse_transport import project_rows_to_simplex, solve_sparse_semi_unbalanced


def _reference(c: np.ndarray, gamma: float, tau: float, allowed: np.ndarray) -> np.ndarray:
    n0, n1 = c.shape
    p = cp.Variable((n0, n1), nonneg=True)
    r = (n1 / n0) * cp.sum(p, axis=0)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(p, c)) + 0.5 * gamma * cp.sum_squares(p)
                                  + 0.5 * tau * cp.sum_squares(r - 1.0)),
                      [cp.sum(p, axis=1) == 1.0, p[~allowed] == 0.0])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal"
    return np.asarray(p.value)


def test_projection_matches_sorting_definition() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, 200))
    p = project_rows_to_simplex(v)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()
    # 単体への射影の KKT: 正の成分では v − p が行ごとに一定、0 の成分ではそれ以上
    for i in range(5):
        pos = p[i] > 0
        th = (v[i] - p[i])[pos]
        assert np.allclose(th, th[0])
        assert (v[i][~pos] <= th[0] + 1e-12).all()


def test_projection_falls_back_when_support_exceeds_top_k() -> None:
    v = np.zeros((2, 300))
    v[:, :150] = 1.0  # 150 列が同点で、上位 64 列では足りない
    p = project_rows_to_simplex(v)
    assert np.allclose(p[:, :150], 1.0 / 150)
    assert (p[:, 150:] == 0).all()


@pytest.mark.parametrize(("gamma", "tau"), [(0.05, 0.0), (0.05, 0.02), (0.5, 0.1)])
def test_matches_reference_qp_and_is_sparse(gamma: float, tau: float) -> None:
    rng = np.random.default_rng(1)
    c = rng.uniform(0.0, 1.0, size=(6, 40))
    allowed = np.ones_like(c, dtype=bool)
    allowed[0, :3] = False
    res = solve_sparse_semi_unbalanced(c, gamma, tau, allowed=allowed)
    ref = _reference(c, gamma, tau, allowed)
    assert np.allclose(res.p, ref, atol=1e-6)
    assert np.allclose(res.p.sum(axis=1), 1.0)
    assert (res.p[~allowed] == 0).all()
    assert (res.p == 0).sum() > 0  # 厳密に 0 の成分がある
    assert res.residual <= 1e-11


def test_raises_when_not_converged() -> None:
    rng = np.random.default_rng(2)
    c = rng.uniform(0.0, 1.0, size=(20, 300))
    with pytest.raises(SolverNotConvergedError):
        solve_sparse_semi_unbalanced(c, 1e-4, 1.0, max_iter=50)


def test_rejects_rows_without_allowed_columns() -> None:
    c = np.ones((2, 3))
    allowed = np.ones((2, 3), dtype=bool)
    allowed[1] = False
    with pytest.raises(ValueError, match="allowed"):
        solve_sparse_semi_unbalanced(c, 0.1, 0.0, allowed=allowed)
