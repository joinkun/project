"""plausibility.py と、射影QPの追加制約(ステージ14)のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    RESIDUAL_FLOOR,
    RESIDUAL_LOWER_BOUNDS,
    PlausibilityConfig,
    build_linear_constraints,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.accounting.projection_qp import project_onto_feasible
from fsgen.errors import InfeasibleProjectionError

_SPEC = load_canonical_spec(Path("conf/accounts/core.yaml"))
_NAMES = state_vector_names(_SPEC, include_subtotals=False)
_IDX = {n: i for i, n in enumerate(_NAMES)}


def _start() -> np.ndarray:
    """貸借が一致する簡単な始点(総資産=1)。"""
    x = np.zeros(len(_NAMES))
    x[_IDX["jppfs_cor_CashAndDeposits"]] = 0.6
    x[_IDX["inventories"]] = 0.4
    x[_IDX["long_term_loans"]] = 0.5
    x[_IDX["jppfs_cor_CapitalStock"]] = 0.8
    x[_IDX["jppfs_cor_RetainedEarnings"]] = -0.3
    return x


def test_start_is_balanced_and_members_are_consistent() -> None:
    x = _start()
    m = subtotal_members(_SPEC, _NAMES)
    assert x @ m["jppfs_cor_Assets"] == pytest.approx(1.0)
    assert x @ m["jppfs_cor_Liabilities"] == pytest.approx(0.5)
    assert x @ m["jppfs_cor_NetAssets"] == pytest.approx(0.5)
    assert x @ m["jppfs_cor_CurrentAssets"] == pytest.approx(1.0)
    assert x @ m["jppfs_cor_NoncurrentLiabilities"] == pytest.approx(0.5)


def test_default_config_adds_no_constraints() -> None:
    assert build_linear_constraints(_SPEC, _NAMES, _start(), PlausibilityConfig()) == ()


def test_subtotal_nonneg_blocks_offsetting_negative_subtotal() -> None:
    """固定負債の残余を負にして長期借入金を相殺する解を、小計の非負が禁じる。"""
    x0 = _start()
    y = x0.copy()
    y[_IDX["__OTHER__jppfs_cor_NoncurrentLiabilities"]] = -0.5  # 固定負債合計=0
    y[_IDX["jppfs_cor_CashAndDeposits"]] -= 0.5  # 貸借を保つ
    cfg = PlausibilityConfig(subtotal_nonneg=True)
    extra = build_linear_constraints(_SPEC, _NAMES, x0, cfg)
    x, info = project_onto_feasible(y, _SPEC, x_start=x0, extra_constraints=extra)
    m = subtotal_members(_SPEC, _NAMES)
    assert x @ m["jppfs_cor_NoncurrentLiabilities"] >= -1e-7
    assert "jppfs_cor_NoncurrentLiabilities>=0" in info.active_constraints


def test_residual_floor_forbids_negative_liability_residual() -> None:
    x0 = _start()
    y = x0.copy()
    y[_IDX["__OTHER__jppfs_cor_CurrentLiabilities"]] = -0.3
    y[_IDX["jppfs_cor_CashAndDeposits"]] -= 0.3
    extra = build_linear_constraints(_SPEC, _NAMES, x0, PlausibilityConfig(residual_nonneg=True))
    x, _ = project_onto_feasible(y, _SPEC, x_start=x0, extra_constraints=extra)
    assert x[_IDX["__OTHER__jppfs_cor_CurrentLiabilities"]] >= RESIDUAL_FLOOR - 1e-7
    assert evaluate_extra_checks(y[None, :], x0[None, :], _SPEC, _NAMES)[0, 1]
    assert not evaluate_extra_checks(x[None, :], x0[None, :], _SPEC, _NAMES)[0, 1]


def test_net_assets_non_deterioration_and_capital_floor() -> None:
    x0 = _start()
    y = x0.copy()
    y[_IDX["jppfs_cor_CapitalStock"]] = 0.1  # 減資
    y[_IDX["jppfs_cor_CashAndDeposits"]] -= 0.7
    cfg = PlausibilityConfig(True, True, 0.5)
    extra = build_linear_constraints(_SPEC, _NAMES, x0, cfg)
    x, _ = project_onto_feasible(y, _SPEC, x_start=x0, extra_constraints=extra)
    m = subtotal_members(_SPEC, _NAMES)
    assert x @ m["jppfs_cor_NetAssets"] >= x0 @ m["jppfs_cor_NetAssets"] - 1e-7
    assert x[_IDX["jppfs_cor_CapitalStock"]] >= 0.5 * x0[_IDX["jppfs_cor_CapitalStock"]] - 1e-7


def test_contradictory_constraints_raise_infeasible() -> None:
    x0 = _start()
    m = subtotal_members(_SPEC, _NAMES)
    a = m["jppfs_cor_Assets"]
    from fsgen.accounting.plausibility import LinearConstraint

    bad = (LinearConstraint("Assets>=3", a, 3.0), LinearConstraint("Assets<=1", -a, -1.0))
    with pytest.raises(InfeasibleProjectionError):
        project_onto_feasible(x0, _SPEC, x_start=x0, extra_constraints=bad)


def test_evaluate_checks_flags_each_condition() -> None:
    x0 = _start()
    x1 = x0.copy()
    x1[_IDX["jppfs_cor_CashAndDeposits"]] = 0.0
    x1[_IDX["inventories"]] = 0.0
    x1[_IDX["jppfs_cor_CapitalStock"]] = 0.3
    x1[_IDX["jppfs_cor_OtherCA"]] = 0.6  # 貸借は保つ(資産合計=0.6+...)
    flags = evaluate_checks(x0[None, :], x1[None, :], _SPEC, _NAMES)[0]
    assert flags[2] and flags[3] and flags[7]  # 現金≈0・主要資産全滅・資本金半減以下
    assert not flags[0]


def test_residual_bounds_cover_only_named_accounts() -> None:
    assert all(n in _IDX for n in RESIDUAL_LOWER_BOUNDS)


def test_residual_floor_is_rounding_tolerance_without_start_relaxation() -> None:
    idx = _IDX["__OTHER__jppfs_cor_CurrentAssets"]
    cfg = PlausibilityConfig(residual_nonneg=True)
    x0 = _start()
    assets0 = float(subtotal_members(_SPEC, _NAMES)["jppfs_cor_Assets"] @ x0)
    lo = next(c for c in build_linear_constraints(_SPEC, _NAMES, x0, cfg)
              if c.name == "residual:jppfs_cor_CurrentAssets")
    assert lo.rhs == RESIDUAL_FLOOR * assets0  # 始点への緩和なし(全始点に同じ下限)
    x0n = x0.copy()
    x0n[idx] = 2 * RESIDUAL_FLOOR  # 始点が下限を破る -> 黙って緩和せず例外
    with pytest.raises(ValueError):
        build_linear_constraints(_SPEC, _NAMES, x0n, cfg)


def test_cash_over_assets_flag() -> None:
    x0 = _start()
    y = x0.copy()
    y[_IDX["jppfs_cor_CashAndDeposits"]] += 0.5
    y[_IDX["__OTHER__jppfs_cor_CurrentAssets"]] -= 0.5
    assert evaluate_extra_checks(y[None, :], x0[None, :], _SPEC, _NAMES)[0, 2]
    assert not evaluate_extra_checks(x0[None, :], x0[None, :], _SPEC, _NAMES)[0, 2]


def test_basis_sign_violation_flag() -> None:
    x0 = _start()
    y = x0.copy()
    y[_IDX["jppfs_cor_OtherCL"]] = -0.05  # nonneg科目が負
    z = x0.copy()
    z[_IDX["allowance_doubtful_current"]] = 0.05  # nonpos科目が正
    ok = evaluate_extra_checks(x0[None, :], x0[None, :], _SPEC, _NAMES)[0, 3]
    assert not ok
    assert evaluate_extra_checks(y[None, :], x0[None, :], _SPEC, _NAMES)[0, 3]
    assert evaluate_extra_checks(z[None, :], x0[None, :], _SPEC, _NAMES)[0, 3]


def test_contra_net_blocks_cash_over_assets_via_allowance() -> None:
    x0 = _start()
    y = x0.copy()
    y[_IDX["jppfs_cor_CashAndDeposits"]] = 1.1  # 現金を増やし、貸倒引当金(負)で相殺して総資産を保つ
    y[_IDX["allowance_doubtful_current"]] = -0.5
    cfg = PlausibilityConfig(subtotal_nonneg=True, residual_nonneg=True, contra_net_nonneg=True)
    extra = build_linear_constraints(_SPEC, _NAMES, x0, cfg)
    x, info = project_onto_feasible(y, _SPEC, x_start=x0, extra_constraints=extra)
    parts = ("receivables_trade", "jppfs_cor_OtherCA", "__OTHER__jppfs_cor_CurrentAssets",
             "allowance_doubtful_current")
    net = sum(x[_IDX[n]] for n in parts)
    assert net >= -1e-7
    assert not evaluate_extra_checks(x[None, :], x0[None, :], _SPEC, _NAMES)[0, 4]
    assert evaluate_extra_checks(y[None, :], x0[None, :], _SPEC, _NAMES)[0, 4]


def test_retention_beta_keeps_operating_assets() -> None:
    x0 = _start()
    y = x0.copy()
    y[_IDX["inventories"]] = 0.0  # 棚卸資産を全売却
    y[_IDX["jppfs_cor_CashAndDeposits"]] += 0.4
    cfg = PlausibilityConfig(retention_beta=0.5)
    extra = build_linear_constraints(_SPEC, _NAMES, x0, cfg)
    x, info = project_onto_feasible(y, _SPEC, x_start=x0, extra_constraints=extra)
    assert x[_IDX["inventories"]] >= 0.5 * x0[_IDX["inventories"]] - 1e-7
    assert any(a.startswith("retention:inventories") for a in info.active_constraints)


def test_apply_layers_forces_layer2_off_to_s0() -> None:
    from fsgen.accounting.plausibility import ConstraintLayers, apply_layers

    cfg = PlausibilityConfig(True, True, 0.5, None, True, retention_beta=0.3,
                             contra_net_nonneg=True, label="full")
    off = apply_layers(cfg, ConstraintLayers(layer2=False))
    x0 = _start()
    assert build_linear_constraints(_SPEC, _NAMES, x0, off) == ()
    on = apply_layers(cfg, ConstraintLayers(layer2=True))
    assert on == cfg


def test_load_constraint_layers_rejects_layer1_disabled(tmp_path: Path) -> None:
    from fsgen.accounting.plausibility import load_constraint_layers

    p = tmp_path / "core.yaml"
    p.write_text("constraint_layers:\n  layer1: false\n  layer2: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="layer1"):
        load_constraint_layers(p)


def test_load_constraint_layers_from_core_yaml() -> None:
    from fsgen.accounting.plausibility import load_constraint_layers

    layers = load_constraint_layers(Path("conf/accounts/core.yaml"))
    assert layers.layer1 is True
    assert layers.layer2 is True


def test_liability_doubling_ignores_negligible_liabilities() -> None:
    # 始点の負債が総資産の0.2%: 3倍になっても終点は0.6%(<1%)なので「負債が2倍超」にしない
    x0 = _start()
    x0[_IDX["long_term_loans"]] = 0.002
    x0[_IDX["jppfs_cor_CapitalStock"]] = 1.298
    y = x0.copy()
    y[_IDX["long_term_loans"]] = 0.006
    y[_IDX["jppfs_cor_CashAndDeposits"]] += 0.004
    assert not evaluate_checks(x0[None, :], y[None, :], _SPEC, _NAMES)[0, 4]
    z = x0.copy()
    z[_IDX["long_term_loans"]] = 0.05  # 25倍かつ総資産の5%: 対象
    z[_IDX["jppfs_cor_CashAndDeposits"]] += 0.048
    assert evaluate_checks(x0[None, :], z[None, :], _SPEC, _NAMES)[0, 4]
