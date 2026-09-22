"""収束チェック(ステージ13 §A)のテスト: 未収束は警告ではなく例外。"""

from __future__ import annotations

import numpy as np
import pytest

from fsgen.errors import SolverNotConvergedError
from fsgen.models.transport import (
    check_plan_health,
    solve_ot_from_cost,
    solve_ot_normalized,
    solve_unbalanced_normalized,
)


def _underflow_cost() -> np.ndarray:
    cost = np.full((4, 6), 1.0)
    cost[0, :] = 900.0  # 線形領域Sinkhornでは全要素アンダーフロー
    return cost


def test_standard_sinkhorn_raises_when_it_does_not_converge() -> None:
    a, b = np.full(4, 1 / 4), np.full(6, 1 / 6)
    with pytest.raises(SolverNotConvergedError):
        solve_ot_normalized(_underflow_cost(), a, b, 0.05, log_domain=False, max_iter=1000)


def test_default_solver_is_log_domain_and_survives_underflow() -> None:
    plan = solve_ot_from_cost(_underflow_cost(), 0.05)
    assert plan.sum(axis=1).min() > 0


def test_plan_health_rejects_zero_mass_row_and_nan() -> None:
    plan = np.full((3, 4), 1 / 12)
    plan[1] = 0.0
    with pytest.raises(SolverNotConvergedError, match="質量0"):
        check_plan_health(plan)
    bad = np.full((3, 4), 1 / 12)
    bad[0, 0] = np.nan
    with pytest.raises(SolverNotConvergedError, match="NaN"):
        check_plan_health(bad)


def test_plan_health_marginal_error_and_support() -> None:
    plan = np.full((3, 4), 1 / 12)
    a, b = np.full(3, 1 / 3), np.full(4, 1 / 4)
    health = check_plan_health(plan, a, b)
    assert health.max_marginal_error < 1e-12
    np.testing.assert_allclose(health.effective_support, 4.0)
    with pytest.raises(SolverNotConvergedError, match="周辺分布"):
        check_plan_health(plan * 1.1, a, b)


def test_unbalanced_log_domain_matches_pot_on_regular_problem() -> None:
    import ot

    rng = np.random.default_rng(0)
    cost = rng.uniform(0.5, 2.0, size=(5, 8))
    a, b = np.full(5, 1 / 5), np.full(8, 1 / 8)
    mine = solve_unbalanced_normalized(cost, a, b, 0.1, 1.0)
    ref = ot.unbalanced.sinkhorn_unbalanced(a, b, cost, reg=0.1, reg_m=1.0, numItermax=10000)
    np.testing.assert_allclose(mine, ref, atol=1e-6)


def test_unbalanced_drops_extreme_source_and_health_check_flags_it() -> None:
    cost = np.full((4, 6), 1.0)
    cost[0, :] = 900.0  # 不均衡OTは外れ値の始点を切り捨てる → 質量0の行は例外(§A.2)
    a, b = np.full(4, 1 / 4), np.full(6, 1 / 6)
    with pytest.raises(SolverNotConvergedError, match="質量0"):
        solve_unbalanced_normalized(cost, a, b, 1.0, 1.0)


def test_unbalanced_solver_converges_on_regular_problem() -> None:
    rng = np.random.default_rng(0)
    cost = rng.uniform(0.5, 2.0, size=(5, 8))
    a, b = np.full(5, 1 / 5), np.full(8, 1 / 8)
    plan = solve_unbalanced_normalized(cost, a, b, 0.1, 1.0)
    assert np.isfinite(plan).all() and plan.sum(axis=1).min() > 0
