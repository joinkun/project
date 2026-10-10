"""transport.py のテスト。合成データで検証する(実データ非依存)。"""

from __future__ import annotations

import numpy as np

from fsgen.models.transport import (
    barycenter_paths,
    cost_matrix,
    effective_support_sizes,
    paths_to_state_vectors,
    plan_entropy,
    single_target_paths,
    solve_ot,
    solve_ot_from_cost,
    solve_ot_from_cost_verbose,
    solve_ot_verbose,
    solve_unbalanced_ot,
)


def _make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=(5, 3))
    z1 = rng.normal(loc=10.0, size=(8, 3))
    return z0, z1


def test_cost_matrix_shape_and_values() -> None:
    z0 = np.array([[0.0, 0.0]])
    z1 = np.array([[3.0, 4.0], [0.0, 0.0]])
    cost = cost_matrix(z0, z1)
    assert cost.shape == (1, 2)
    assert np.isclose(cost[0, 0], 25.0)
    assert np.isclose(cost[0, 1], 0.0)


def test_solve_ot_marginals_approximately_uniform() -> None:
    z0, z1 = _make_data()
    plan = solve_ot(z0, z1, epsilon=0.5)
    assert plan.shape == (5, 8)
    assert np.isclose(plan.sum(), 1.0, atol=1e-6)
    assert np.allclose(plan.sum(axis=1), 1.0 / 5, atol=1e-4)
    assert np.allclose(plan.sum(axis=0), 1.0 / 8, atol=1e-4)


def test_solve_ot_verbose_converges_for_reasonable_epsilon() -> None:
    z0, z1 = _make_data()
    result = solve_ot_verbose(z0, z1, epsilon=0.5)
    assert result.converged
    assert result.final_error < 1e-9


def test_solve_unbalanced_ot_shape_and_nonnegative() -> None:
    z0, z1 = _make_data()
    plan = solve_unbalanced_ot(z0, z1, epsilon=0.5, tau=1.0)
    assert plan.shape == (5, 8)
    assert np.all(plan >= 0.0)


def test_solve_unbalanced_ot_small_tau_relaxes_marginals() -> None:
    """tauが小さいほど周辺分布の制約が緩む(総質量が一様分布から離れやすい)。"""
    z0, z1 = _make_data()
    plan_strict = solve_unbalanced_ot(z0, z1, epsilon=0.05, tau=10.0)
    plan_loose = solve_unbalanced_ot(z0, z1, epsilon=0.05, tau=0.1)
    a = np.full(5, 1.0 / 5)
    err_strict = np.max(np.abs(plan_strict.sum(axis=1) - a))
    err_loose = np.max(np.abs(plan_loose.sum(axis=1) - a))
    assert err_loose >= err_strict - 1e-9


def test_small_epsilon_gives_lower_entropy_than_large_epsilon() -> None:
    """εが小さいほど輸送計画は決定的(低エントロピー)になるはず。"""
    z0, z1 = _make_data()
    plan_small = solve_ot(z0, z1, epsilon=0.01)
    plan_large = solve_ot(z0, z1, epsilon=5.0)
    assert plan_entropy(plan_small) < plan_entropy(plan_large)


def test_large_epsilon_gives_larger_effective_support() -> None:
    z0, z1 = _make_data()
    plan_small = solve_ot(z0, z1, epsilon=0.01)
    plan_large = solve_ot(z0, z1, epsilon=5.0)
    ess_small = effective_support_sizes(plan_small)
    ess_large = effective_support_sizes(plan_large)
    assert ess_large.mean() > ess_small.mean()


def test_single_target_paths_endpoints() -> None:
    z0, z1 = _make_data()
    plan = solve_ot(z0, z1, epsilon=0.01)
    ts = (0.0, 0.5, 1.0)
    paths, targets = single_target_paths(z0, z1, plan, ts)
    assert paths.shape == (5, 3, 3)
    assert targets.shape == (5,)
    assert np.allclose(paths[:, 0, :], z0)
    for i in range(5):
        assert np.allclose(paths[i, -1, :], z1[targets[i]])
        assert np.allclose(paths[i, 1, :], 0.5 * z0[i] + 0.5 * z1[targets[i]])


def test_single_target_uses_argmax_row() -> None:
    z0 = np.zeros((1, 2))
    z1 = np.array([[1.0, 1.0], [2.0, 2.0]])
    plan = np.array([[0.1, 0.9]])
    _paths, targets = single_target_paths(z0, z1, plan, (0.0, 1.0))
    assert targets[0] == 1


def test_barycenter_paths_endpoints() -> None:
    z0, z1 = _make_data()
    plan = solve_ot(z0, z1, epsilon=0.5)
    ts = (0.0, 1.0)
    paths = barycenter_paths(z0, z1, plan, ts)
    assert paths.shape == (5, 2, 3)
    assert np.allclose(paths[:, 0, :], z0)
    weights = plan / plan.sum(axis=1, keepdims=True)
    expected_bary = weights @ z1
    assert np.allclose(paths[:, 1, :], expected_bary)


def test_barycenter_has_smaller_spread_than_single_target() -> None:
    """バリセンターは複数終点の加重平均なので、単一対応より到達点の分散が
    小さくなるはず(§6の「バリセンターは平均に潰れる」)。"""
    z0, z1 = _make_data(seed=1)
    plan = solve_ot(z0, z1, epsilon=1.0)
    ts = (0.0, 1.0)
    single_paths, _targets = single_target_paths(z0, z1, plan, ts)
    bary_paths = barycenter_paths(z0, z1, plan, ts)
    single_spread = np.var(single_paths[:, -1, :], axis=0).sum()
    bary_spread = np.var(bary_paths[:, -1, :], axis=0).sum()
    assert bary_spread <= single_spread


def test_solve_ot_from_cost_matches_solve_ot_with_euclidean_cost() -> None:
    """事前計算したコスト行列を渡すsolve_ot_from_costは、solve_otが内部で
    計算するのと同じコスト行列(cost_matrix)を渡せば同じ結果になるはず。"""
    z0, z1 = _make_data()
    plan_direct = solve_ot(z0, z1, epsilon=0.5)
    cost = cost_matrix(z0, z1)
    plan_from_cost = solve_ot_from_cost(cost, epsilon=0.5)
    assert np.allclose(plan_direct, plan_from_cost)


def test_solve_ot_from_cost_verbose_reports_shape_from_cost_matrix() -> None:
    cost = np.abs(np.random.default_rng(0).normal(size=(4, 6)))
    result = solve_ot_from_cost_verbose(cost, n0=4, n1=6, epsilon=0.5)
    assert result.plan.shape == (4, 6)


_C_SIMPLE = np.array([[1.0], [1.0], [-1.0], [0.0]])  # x[0] + x[1] - x[2] = 0
_B_SIMPLE = np.array(
    [
        [1 / np.sqrt(2), 0.0],
        [-1 / np.sqrt(2), 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ]
)


def test_paths_to_state_vectors_satisfies_constraint() -> None:
    """x_anchor が恒等式を満たし、B の列が C^T B = 0 を満たすなら、
    x_anchor + Assets*B(z(t)-z_anchor) は常に恒等式を満たす。"""
    assert np.allclose(_C_SIMPLE.T @ _B_SIMPLE, 0.0, atol=1e-10)

    # 恒等式 x[0]+x[1]-x[2]=0 を満たす3社分の実測値
    x_anchor = np.array([[1.0, 1.0, 2.0, 0.0], [2.0, 3.0, 5.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    z_anchor = np.array([[0.1, 0.2], [0.3, -0.1], [0.0, 0.0]])

    rng = np.random.default_rng(0)
    z_paths = rng.normal(size=(3, 5, 2))  # (n, T, k)
    assets = np.array([100.0, 200.0, 300.0])
    x_paths = paths_to_state_vectors(z_paths, z_anchor, x_anchor, _B_SIMPLE, assets)
    assert x_paths.shape == (3, 5, 4)
    residual = np.einsum("ntd,dk->ntk", x_paths, _C_SIMPLE)
    assert np.allclose(residual, 0.0, atol=1e-8)


def test_paths_to_state_vectors_exact_at_anchor() -> None:
    """z(t)=z_anchor となる t では x(t)=x_anchor が厳密成立する(§B)。"""
    x_anchor = np.array([[1.0, 1.0, 2.0, 0.0], [2.0, 3.0, 5.0, 1.0]])
    z_anchor = np.array([[0.1, 0.2], [0.3, -0.1]])
    rng = np.random.default_rng(1)
    z_target = rng.normal(size=(2, 2))
    ts = np.array([0.0, 0.5, 1.0])
    z_paths = (1 - ts)[None, :, None] * z_anchor[:, None, :] + ts[None, :, None] * z_target[
        :, None, :
    ]
    assets = np.array([100.0, 200.0])
    x_paths = paths_to_state_vectors(z_paths, z_anchor, x_anchor, _B_SIMPLE, assets)
    assert np.allclose(x_paths[:, 0, :], x_anchor)


def test_paths_to_state_vectors_scales_displacement_by_assets() -> None:
    b = np.eye(4)[:, :2]
    x_anchor = np.zeros((2, 4))
    z_anchor = np.zeros((2, 2))
    z_paths = np.ones((2, 1, 2))  # displacement = 1 in each of 2 dims
    assets = np.array([1.0, 10.0])
    x_paths = paths_to_state_vectors(z_paths, z_anchor, x_anchor, b, assets)
    assert np.allclose(x_paths[0], [1.0, 1.0, 0.0, 0.0])
    assert np.allclose(x_paths[1], [10.0, 10.0, 0.0, 0.0])


def test_semi_unbalanced_preserves_rows_and_relaxes_columns() -> None:
    from fsgen.models.transport import solve_semi_unbalanced_normalized

    rng = np.random.default_rng(0)
    cost = rng.uniform(0.0, 2.0, size=(5, 40))
    a, b = np.full(5, 0.2), np.full(40, 1.0 / 40)
    res = solve_semi_unbalanced_normalized(cost, a, b, epsilon=0.05, tau1=0.01)
    assert np.allclose(res.plan.sum(axis=1), a, atol=1e-8)  # 始点側は厳密
    assert np.abs(res.plan.sum(axis=0) - b).max() > 1e-3  # 目標側は緩む
    # 目標側を緩めると、各行は最安の列に寄る(バランスの下限 n1/n0 = 8 を割る)
    from fsgen.models.transport import effective_support_sizes

    assert np.median(effective_support_sizes(res.plan)) < 8.0


def test_semi_unbalanced_large_tau_approaches_balanced() -> None:
    from fsgen.models.transport import solve_ot_normalized, solve_semi_unbalanced_normalized

    rng = np.random.default_rng(1)
    cost = rng.uniform(0.0, 1.0, size=(4, 12))
    a, b = np.full(4, 0.25), np.full(12, 1.0 / 12)
    semi = solve_semi_unbalanced_normalized(cost, a, b, epsilon=0.1, tau1=1e4).plan
    bal = solve_ot_normalized(cost, a, b, 0.1).plan
    assert np.abs(semi - bal).max() < 1e-3


def test_semi_unbalanced_rejects_nonpositive_parameters() -> None:
    import pytest

    from fsgen.models.transport import solve_semi_unbalanced_normalized

    with pytest.raises(ValueError):
        solve_semi_unbalanced_normalized(np.ones((2, 2)), np.full(2, 0.5), np.full(2, 0.5),
                                         0.1, 0.0)
