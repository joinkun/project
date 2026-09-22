"""metrics.py のテスト。合成 canonical スペックで検証する(実データ非依存)。"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.constraints import BasisSpec, CanonicalSpec, SubtotalSpec
from fsgen.evaluation import metrics as m

_BASIS = (
    BasisSpec("Cash", "jppfs_cor_CurrentAssets", "assets", sign="nonneg"),
    BasisSpec("STL", "jppfs_cor_CurrentLiabilities", "liabilities", sign="nonneg"),
    BasisSpec("Treasury", "jppfs_cor_ShareholdersEquity", "net_assets", sign="nonpos"),
    BasisSpec("Other", "jppfs_cor_ShareholdersEquity", "net_assets", sign="free"),
)
_SUBTOTALS = (
    SubtotalSpec("jppfs_cor_Assets", "assets"),
    SubtotalSpec("jppfs_cor_Liabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NetAssets", "net_assets"),
)
_SPEC = CanonicalSpec(name="synthetic", basis=_BASIS, subtotals=_SUBTOTALS)
_NAMES = ("Cash", "STL", "Treasury", "Other")
_C = np.array([[1.0], [-1.0], [-1.0], [-1.0]])  # Cash - STL - Treasury - Other = 0


def _feasible_point(cash: float, stl: float, treasury: float) -> np.ndarray:
    other = cash - stl - treasury
    return np.array([cash, stl, treasury, other])


def test_proximity_l2_is_zero_for_static_path() -> None:
    x = _feasible_point(100.0, 30.0, -10.0)
    path = np.tile(x, (5, 1))
    assert m.proximity_l2(path, assets_start=100.0) == 0.0


def test_proximity_l2_scales_with_displacement() -> None:
    x0 = _feasible_point(100.0, 30.0, -10.0)
    x1 = _feasible_point(150.0, 30.0, -10.0)
    path = np.stack([x0, x1])
    expected = float(np.linalg.norm(x1 - x0)) / 100.0
    assert abs(m.proximity_l2(path, assets_start=100.0) - expected) < 1e-9


def test_proximity_cost_weighted_picks_direction_specific_weight() -> None:
    x0 = _feasible_point(100.0, 30.0, -10.0)
    x1 = x0.copy()
    x1[0] += 10.0  # Cash増加
    x1[1] -= 5.0  # STL減少
    w_plus = np.array([2.0, 3.0, 1.0, 1.0])
    w_minus = np.array([5.0, 7.0, 1.0, 1.0])
    path = np.stack([x0, x1])
    result = m.proximity_cost_weighted(path, assets_start=100.0, w_plus=w_plus, w_minus=w_minus)
    # Cash: 増加なのでw_plus[0]=2.0 * 10 = 20、STL: 減少なのでw_minus[1]=7.0 * 5 = 35
    expected = (2.0 * 10.0 + 7.0 * 5.0) / 100.0
    assert abs(result - expected) < 1e-9


def test_sparsity_counts_only_materially_changed_accounts() -> None:
    x0 = _feasible_point(100.0, 30.0, -10.0)
    x1 = x0.copy()
    x1[0] += 5.0  # 5%の変化(閾値1%を超える)
    x1[1] += 0.05  # 0.05%の変化(閾値未満)
    path = np.stack([x0, x1])
    assert m.sparsity(path, assets_start=100.0) == 1


def test_balance_residual_is_zero_for_feasible_path() -> None:
    path = np.stack([_feasible_point(100.0, 30.0, -10.0), _feasible_point(80.0, 20.0, -5.0)])
    assert m.balance_residual(path, _C, assets_start=100.0) < 1e-9


def test_balance_residual_detects_violation() -> None:
    x_bad = np.array([100.0, 30.0, -10.0, 0.0])  # 100 != 30-10+0
    path = np.stack([x_bad, x_bad])
    assert m.balance_residual(path, _C, assets_start=100.0) > 0.1


def test_sign_violation_count_detects_and_ignores_within_tolerance() -> None:
    x_ok = _feasible_point(100.0, 30.0, -10.0)
    x_violate = _feasible_point(-5.0, 30.0, -10.0)  # Cash negative
    path = np.stack([x_ok, x_violate, x_ok])
    count = m.sign_violation_count(path, _SPEC, _NAMES, assets_start=100.0, tol_ratio=1e-4)
    assert count == 1


def test_direction_violation_count_respects_trigger_flag() -> None:
    x0 = _feasible_point(10.0, 100.0, -5.0)
    x1 = _feasible_point(10.0, 100.0, -50.0)  # Treasuryの絶対値が増えた
    path = np.stack([x0, x1])
    assert m.direction_violation_count(path, _NAMES, "Treasury", True, 100.0, 1e-4) == 1
    assert m.direction_violation_count(path, _NAMES, "Treasury", False, 100.0, 1e-4) == 0


def test_monotonicity_reversals_counts_direction_changes() -> None:
    path = np.array([[0.0], [1.0], [0.5], [2.0]])  # +,-,+ で2回反転
    assert m.monotonicity_reversals(path) == 2

    path_monotone = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert m.monotonicity_reversals(path_monotone) == 0


def test_nearest_neighbor_distance_finds_closest_point() -> None:
    pool = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    target = np.array([0.9, 0.0])
    assert abs(m.nearest_neighbor_distance(target, pool) - 0.1) < 1e-9


def test_percentile_rank_basic() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m.percentile_rank(3.0, reference) == 60.0  # 5個中3個が<=3.0


def test_correlation_frobenius_distance_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 3))
    assert m.correlation_frobenius_distance(data, data) < 1e-9
