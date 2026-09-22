"""differentiable_projection.py のテスト。非微分版(project_onto_feasible)と
数値一致することを確認する。"""

from __future__ import annotations

import numpy as np
import torch

from fsgen.accounting.constraints import BasisSpec, CanonicalSpec, SubtotalSpec
from fsgen.accounting.projection_qp import DirectionConstraintSpec, project_onto_feasible
from fsgen.models.differentiable_projection import build_projection_layer

_BASIS = (
    BasisSpec("Cash", "jppfs_cor_CurrentAssets", "assets", sign="nonneg"),
    BasisSpec("STL", "jppfs_cor_CurrentLiabilities", "liabilities", sign="nonneg"),
    BasisSpec("Treasury", "jppfs_cor_ShareholdersEquity", "net_assets", sign="nonpos"),
    BasisSpec("RE", "jppfs_cor_ShareholdersEquity", "net_assets", sign="free"),
)
_SUBTOTALS = (
    SubtotalSpec("jppfs_cor_Assets", "assets"),
    SubtotalSpec("jppfs_cor_Liabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NetAssets", "net_assets"),
)
_SPEC = CanonicalSpec(name="synthetic", basis=_BASIS, subtotals=_SUBTOTALS)


def _feasible_point(cash: float, stl: float, treasury: float) -> np.ndarray:
    re = cash - stl - treasury
    return np.array([cash, stl, treasury, re])


def test_differentiable_projection_matches_non_differentiable() -> None:
    x_start = _feasible_point(100.0, 30.0, -10.0)
    weights = 1.0 / (np.abs(x_start) + 0.1)
    y = np.array([-50.0, 30.0, -10.0, -90.0])

    x_ref, _info = project_onto_feasible(y, _SPEC, x_start=x_start, weights=weights)

    layer = build_projection_layer(_SPEC, x_start, weights)
    y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    x_diff = layer.project(y_t)

    # cvxpylayers(diffcp/SCS)はOSQPより緩い既定許容誤差で解くため、比較は粗めにする。
    assert torch.allclose(x_diff, torch.tensor(x_ref, dtype=torch.float64), atol=1e-3)


def test_differentiable_projection_gradient_is_finite() -> None:
    x_start = _feasible_point(100.0, 30.0, -10.0)
    weights = 1.0 / (np.abs(x_start) + 0.1)
    y = np.array([-50.0, 30.0, -10.0, -90.0])
    layer = build_projection_layer(_SPEC, x_start, weights)
    y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    x_diff = layer.project(y_t)
    loss = x_diff.sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert y_t.grad is not None
    assert torch.all(torch.isfinite(y_t.grad))


def test_frozen_accounts_are_held_at_x_start() -> None:
    x_start = _feasible_point(100.0, 30.0, -10.0)
    weights = 1.0 / (np.abs(x_start) + 0.1)
    y = np.array([-50.0, 30.0, -10.0, -9000.0])  # REを大きく動かそうとする目標
    layer = build_projection_layer(_SPEC, x_start, weights, frozen_accounts=("RE",))
    y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    x_diff = layer.project(y_t)
    re_idx = layer.names.index("RE")
    assert abs(float(x_diff[re_idx].detach()) - x_start[re_idx]) < 1e-2


def test_direction_constraint_respected_in_differentiable_layer() -> None:
    x_start = _feasible_point(10.0, 100.0, -5.0)  # net_assets < 0 (debt overhang)
    weights = 1.0 / (np.abs(x_start) + 0.1)
    dc = (DirectionConstraintSpec("Treasury", "net_assets_negative", "monotone_from_start"),)
    layer = build_projection_layer(_SPEC, x_start, weights, direction_constraints=dc)
    y = _feasible_point(10.0, 100.0, -50.0)  # Treasuryをさらにマイナスへ
    y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    x_diff = layer.project(y_t)
    treasury_idx = layer.names.index("Treasury")
    assert float(x_diff[treasury_idx].detach()) >= x_start[treasury_idx] - 1e-4
