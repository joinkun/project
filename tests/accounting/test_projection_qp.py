"""projection_qp.py のテスト。合成 canonical スペックで検証する(実データ非依存)。"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.constraints import BasisSpec, CanonicalSpec, SubtotalSpec
from fsgen.accounting.projection_qp import (
    DirectionConstraintSpec,
    apply_projection_to_paths,
    evaluate_condition,
    is_feasible,
    project_onto_feasible,
)

# 最小限の合成スペック: Cash(nonneg), STL(nonneg, 負債), Treasury(nonpos, 純資産)
# 恒等式: Cash - STL - Treasury - Other = 0 (他の科目は省略、Otherはfree)
_BASIS = (
    BasisSpec("Cash", "jppfs_cor_CurrentAssets", "assets", sign="nonneg"),
    BasisSpec("STL", "jppfs_cor_CurrentLiabilities", "liabilities", sign="nonneg"),
    BasisSpec("Treasury", "jppfs_cor_ShareholdersEquity", "net_assets", sign="nonpos"),
    BasisSpec("Other", "jppfs_cor_ShareholdersEquity", "net_assets", sign="free"),
)
_SUBTOTALS = (
    SubtotalSpec("jppfs_cor_Assets", "assets"),
    SubtotalSpec("jppfs_cor_CurrentAssets", "assets"),
    SubtotalSpec("jppfs_cor_NoncurrentAssets", "assets"),
    SubtotalSpec("jppfs_cor_Liabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_CurrentLiabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NoncurrentLiabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NetAssets", "net_assets"),
    SubtotalSpec("jppfs_cor_ShareholdersEquity", "net_assets"),
    SubtotalSpec("jppfs_cor_ValuationAndTranslationAdjustments", "net_assets"),
)
_SPEC = CanonicalSpec(name="synthetic", basis=_BASIS, subtotals=_SUBTOTALS)
# 恒等式(include_subtotals=False, 単一制約): assets側 - (liab+net_assets側) = 0
# つまり Cash - (STL + Treasury + Other) = 0


def _feasible_point(cash: float, stl: float, treasury: float) -> np.ndarray:
    other = cash - stl - treasury
    return np.array([cash, stl, treasury, other])


def test_feasible_point_stays_unchanged() -> None:
    x = _feasible_point(cash=100.0, stl=30.0, treasury=-10.0)
    x_proj, info = project_onto_feasible(x, _SPEC, x_start=x)
    assert info.status in ("optimal", "optimal_inaccurate")
    assert np.allclose(x_proj, x, atol=1e-5)
    assert info.distance_before_after < 1e-4


def test_infeasible_point_gets_projected_onto_sign_constraint() -> None:
    # Cash が負(nonneg違反)、恒等式もついでに破る点
    y = np.array([-50.0, 30.0, -10.0, -90.0])
    x_proj, info = project_onto_feasible(y, _SPEC, x_start=y)
    assert info.status in ("optimal", "optimal_inaccurate")
    assert x_proj[0] >= -1e-6  # Cash >= 0
    assert x_proj[2] <= 1e-6  # Treasury <= 0
    feasible, violations = is_feasible(x_proj, _SPEC, tol=1e-4)
    assert feasible, violations


def test_projection_is_idempotent() -> None:
    y = np.array([-50.0, 30.0, -10.0, -90.0])
    x1, _info1 = project_onto_feasible(y, _SPEC, x_start=y)
    x2, _info2 = project_onto_feasible(x1, _SPEC, x_start=y)
    assert np.allclose(x1, x2, atol=1e-4)


def test_is_feasible_detects_balance_violation() -> None:
    x = np.array([100.0, 30.0, -10.0, 0.0])  # 100 != 30-10+0=20
    feasible, violations = is_feasible(x, _SPEC, tol=1e-6)
    assert not feasible
    assert any(v[1] == "balance" for v in violations)


def test_is_feasible_detects_sign_violation() -> None:
    x = _feasible_point(cash=-10.0, stl=0.0, treasury=0.0)  # Cash負
    feasible, violations = is_feasible(x, _SPEC, tol=1e-6)
    assert not feasible
    assert any(v[0] == "Cash" and v[1] == "nonneg" for v in violations)


def test_evaluate_condition_net_assets_negative() -> None:
    # Treasury+Other = 10-100 = -90 < 0
    x_negative_na = _feasible_point(cash=10.0, stl=100.0, treasury=0.0)
    assert evaluate_condition("net_assets_negative", x_negative_na, _SPEC) is True
    x_positive_na = _feasible_point(cash=100.0, stl=10.0, treasury=0.0)
    assert evaluate_condition("net_assets_negative", x_positive_na, _SPEC) is False


def test_direction_constraint_prevents_treasury_buyback_when_debt_overhang() -> None:
    x_start = _feasible_point(cash=10.0, stl=100.0, treasury=-5.0)  # net_assets < 0
    dc = (DirectionConstraintSpec("Treasury", "net_assets_negative", "monotone_from_start"),)
    # 射影先 y は Treasury をさらにマイナスへ動かそうとする(自社株買い増加)
    y = _feasible_point(cash=10.0, stl=100.0, treasury=-50.0)
    x_proj, info = project_onto_feasible(y, _SPEC, x_start=x_start, direction_constraints=dc)
    assert info.status in ("optimal", "optimal_inaccurate")
    assert x_proj[2] >= x_start[2] - 1e-4  # Treasury(t) >= Treasury(0)


def test_direction_constraint_not_triggered_without_debt_overhang() -> None:
    x_start = _feasible_point(cash=100.0, stl=10.0, treasury=-5.0)  # net_assets > 0
    dc = (DirectionConstraintSpec("Treasury", "net_assets_negative", "monotone_from_start"),)
    y = _feasible_point(cash=100.0, stl=10.0, treasury=-20.0)
    x_proj, info = project_onto_feasible(y, _SPEC, x_start=x_start, direction_constraints=dc)
    assert info.status in ("optimal", "optimal_inaccurate")
    # 制約が発動しないので y にそのまま一致してよい(nonposは満たしている)
    assert np.allclose(x_proj, y, atol=1e-4)


def test_enforce_inactive_keeps_zero_components_zero() -> None:
    x_start = _feasible_point(cash=100.0, stl=0.0, treasury=0.0)
    y = _feasible_point(cash=80.0, stl=20.0, treasury=-5.0)
    x_proj, _info = project_onto_feasible(y, _SPEC, x_start=x_start, enforce_inactive=True)
    assert abs(x_proj[1]) < 1e-6  # STL は始点で0だったので0のまま
    assert abs(x_proj[2]) < 1e-6  # Treasury も始点で0だったので0のまま


def test_weighted_and_unweighted_projections_differ() -> None:
    y = np.array([-50.0, 30.0, -10.0, -90.0])
    x_unweighted, _ = project_onto_feasible(y, _SPEC, x_start=y, weights=None)
    weights = 1.0 / (np.abs(y) + 1e-3)
    x_weighted, _ = project_onto_feasible(y, _SPEC, x_start=y, weights=weights)
    assert not np.allclose(x_unweighted, x_weighted, atol=1e-3)


def _make_path(n: int, t_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    x_anchor = np.stack([_feasible_point(100.0 + 10 * i, 20.0, -5.0) for i in range(n)])
    x_paths = np.zeros((n, t_len, 4))
    for i in range(n):
        for t_i in range(t_len):
            x_paths[i, t_i] = x_anchor[i] + rng.normal(scale=5.0, size=4)
    assets = np.full(n, 500.0)
    return x_paths, x_anchor, assets


def test_apply_projection_every_step_projects_all_t() -> None:
    x_paths, x_anchor, assets = _make_path(n=2, t_len=3)
    x_proj, infos = apply_projection_to_paths(
        x_paths, x_anchor, assets, _SPEC, mode="every_step"
    )
    assert len(infos) == 2 * 3
    for i in range(2):
        for t_i in range(3):
            feasible, violations = is_feasible(x_proj[i, t_i], _SPEC, tol=1e-4)
            assert feasible, violations


def test_apply_projection_endpoint_only_leaves_early_t_unprojected() -> None:
    x_paths, x_anchor, assets = _make_path(n=2, t_len=3)
    x_proj, infos = apply_projection_to_paths(
        x_paths, x_anchor, assets, _SPEC, mode="endpoint_only"
    )
    assert len(infos) == 2  # 終端のみ射影
    # t=0, t=1(中間)は無補正のまま
    assert np.array_equal(x_proj[:, 0, :], x_paths[:, 0, :])
    assert np.array_equal(x_proj[:, 1, :], x_paths[:, 1, :])
    # 終端は実行可能になっている
    for i in range(2):
        feasible, violations = is_feasible(x_proj[i, -1], _SPEC, tol=1e-4)
        assert feasible, violations


def test_l1_move_zero_lambda_matches_plain_projection() -> None:
    y = np.array([-50.0, 30.0, -10.0, -90.0])
    x0 = _feasible_point(cash=100.0, stl=30.0, treasury=-10.0)
    plain, _ = project_onto_feasible(y, _SPEC, x_start=x0)
    moved, info = project_onto_feasible(y, _SPEC, x_start=x0,
                                        l1_move=(np.ones(4), np.ones(4), 0.0))
    assert np.allclose(plain, moved, atol=1e-4)
    assert info.move_up is not None and info.move_down is not None


def test_l1_move_parts_are_complementary_and_large_lambda_stays_at_start() -> None:
    x0 = _feasible_point(cash=100.0, stl=30.0, treasury=-10.0)
    y = _feasible_point(cash=60.0, stl=10.0, treasury=-5.0)
    c_plus, c_minus = np.array([1.0, 2.0, 1.0, 0.5]), np.array([3.0, 1.0, 1.0, 0.5])
    small, info = project_onto_feasible(y, _SPEC, x_start=x0, l1_move=(c_plus, c_minus, 0.01))
    assert info.move_up is not None and info.move_down is not None
    assert np.minimum(info.move_up, info.move_down).max() < 1e-6  # 同時に正にならない
    assert np.allclose(info.move_up - info.move_down, small - x0, atol=1e-4)
    big, _ = project_onto_feasible(y, _SPEC, x_start=x0, l1_move=(c_plus, c_minus, 1e4))
    assert np.allclose(big, x0, atol=1e-3)  # 動かす代償が大きければ始点に留まる
    assert np.linalg.norm(small - y) < np.linalg.norm(big - y)
