"""recourse_qp.py のテスト。合成 canonical スペック(projection_qp のテストと同じ)で検証する。"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.constraints import BasisSpec, CanonicalSpec, SubtotalSpec
from fsgen.accounting.recourse_qp import (
    RecourseQP,
    RecourseWeights,
    lower_bound,
    recourse_objective,
)

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


def _point(cash: float, stl: float, treasury: float) -> np.ndarray:
    return np.array([cash, stl, treasury, cash - stl - treasury])


def _weights(eta: float) -> RecourseWeights:
    # A_BS = Cash・STL(純資産の 2 科目は NA の項だけ)
    return RecourseWeights(w_bs=np.array([4.0, 2.0, 0.0, 0.0]),
                           na_coef=np.array([0.0, 0.0, 1.0, 1.0]), w_na=3.0,
                           c_plus=np.array([1.0, 2.0, 0.5, 1.5]),
                           c_minus=np.array([3.0, 1.0, 0.5, 1.5]), eta=eta)


def test_solution_feasible_and_parts_consistent() -> None:
    x0 = _point(1.0, 0.6, -0.1)
    tau = _point(1.4, 0.2, 0.0)
    for eta in (0.0, 1.0):
        rw = _weights(eta)
        r = RecourseQP(_SPEC, x0, rw, lam=0.3).solve(tau)
        assert r.status == "optimal"
        assert abs(r.y[0] - r.y[1] - r.y[2] - r.y[3]) < 1e-7  # 貸借一致
        assert r.y[0] >= -1e-9 and r.y[2] <= 1e-9
        np.testing.assert_allclose(r.move_up - r.move_down, r.y - x0, atol=1e-6)
        assert np.minimum(r.move_up, r.move_down).max() < 1e-6
        assert abs(r.objective - recourse_objective(r.y, x0, tau, rw, 0.3)) < 1e-9


def test_lower_bound_never_exceeds_qp_optimum() -> None:
    rng = np.random.default_rng(0)
    x0 = _point(1.0, 0.6, -0.1)
    rw = _weights(1.0)
    qp = RecourseQP(_SPEC, x0, rw, lam=0.5)
    targets = np.array([_point(*rng.uniform([0.2, 0.0, -0.5], [2.0, 1.5, 0.0]))
                        for _ in range(20)])
    lb = lower_bound(x0, targets, rw, 0.5)
    for t, b in zip(targets, lb, strict=True):
        assert b <= qp.solve(t).objective + 1e-8


def test_large_lambda_stays_at_start_and_zero_lambda_reaches_feasible_target() -> None:
    x0 = _point(1.0, 0.6, -0.1)
    tau = _point(1.3, 0.4, -0.1)  # 実行可能な目標
    rw = _weights(1.0)
    still = RecourseQP(_SPEC, x0, rw, lam=1e4).solve(tau)
    np.testing.assert_allclose(still.y, x0, atol=1e-4)
    near = RecourseQP(_SPEC, x0, rw, lam=0.0).solve(tau)
    np.testing.assert_allclose(near.y[:2], tau[:2], atol=1e-5)
    assert abs(rw.na_coef @ (near.y - tau)) < 1e-5
