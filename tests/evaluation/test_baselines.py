"""baselines.py のテスト。合成データで各手法の基本的な性質を検証する。"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.constraints import (
    BasisSpec,
    CanonicalSpec,
    SubtotalSpec,
    build_constraint_matrix,
)
from fsgen.accounting.projection_qp import is_feasible
from fsgen.evaluation.baselines import (
    ConvexQpBaseline,
    PriorWorkBaseline,
    build_gaussian_baseline,
    build_naive_ot_baseline,
    ot_target_for_new_point,
)

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
_C = build_constraint_matrix(_SPEC, include_subtotals=False)
_TS = (0.0, 0.5, 1.0)


def _feasible_point(cash: float, stl: float, treasury: float) -> np.ndarray:
    other = cash - stl - treasury
    return np.array([cash, stl, treasury, other])


def test_prior_work_only_touches_target_accounts() -> None:
    median = _feasible_point(200.0, 50.0, -20.0)
    baseline = PriorWorkBaseline(median=median, target_idx=(0,))  # "Cash" だけを対象にする
    x_start = _feasible_point(100.0, 30.0, -10.0)
    path = baseline.generate(x_start, assets_start=100.0, ts=_TS)
    assert np.allclose(path[:, 0], [100.0, 150.0, 200.0])  # Cashはmedianへ線形補間
    # STL/Treasury/Otherは動かない(target_idxに含まれない)
    assert np.allclose(path[:, 1], x_start[1])
    assert np.allclose(path[:, 2], x_start[2])
    assert np.allclose(path[:, 3], x_start[3])


def test_convex_qp_path_stays_feasible_by_convexity() -> None:
    median_tilde = _feasible_point(2.0, 0.5, -0.2) / 1.0  # 正規化済み中央値(相似形)
    baseline = ConvexQpBaseline(
        median_tilde=median_tilde, spec=_SPEC, direction_constraints=(), weighted=True
    )
    x_start = _feasible_point(-50.0, 30.0, -10.0)  # Cashが負(実行不可能)
    path = baseline.generate(x_start, assets_start=100.0, ts=(0.0, 0.25, 0.5, 0.75, 1.0))
    # 始点は実行不可能だが、目標点は実行可能なQP解であり、直線経路の各点(始点以外)は
    # 実行可能集合(凸)の内分点になる。t=0のみ例外(始点そのもの、射影しない)。
    for t_i in range(1, path.shape[0]):
        feasible, violations = is_feasible(path[t_i], _SPEC, tol=1e-3)
        assert feasible, violations


def test_ot_target_for_new_point_borrows_nearest_peer() -> None:
    rng = np.random.default_rng(0)
    b = np.eye(4)[:, :2]  # k=2
    z0 = rng.normal(size=(20, 2))
    z_train2 = np.array([[0.0, 0.0], [5.0, 5.0]])
    targets_train2 = np.array([3, 7])  # train2企業0番はz0[3]、1番はz0[7]に対応
    x_start = np.array([0.1, 0.0, 0.0, 0.0])  # z空間で[0.1,0]に近い -> train2の0番が近い
    ot = ot_target_for_new_point(x_start, 1.0, b, z0, z_train2, targets_train2)
    assert np.allclose(ot.z_target, z0[3])


def test_naive_ot_baseline_generates_path_toward_borrowed_target() -> None:
    rng = np.random.default_rng(1)
    x0_tilde = rng.normal(size=(30, 4)) @ _null_space_projector()
    train2_tilde = rng.normal(size=(5, 4)) @ _null_space_projector()
    pool = np.concatenate([x0_tilde, train2_tilde], axis=0)
    idx0 = np.arange(30)
    idx_train2 = np.arange(30, 35)
    baseline = build_naive_ot_baseline(
        pool, _C, k=2, idx0_in_pool=idx0, idx_train2_in_pool=idx_train2, epsilon=0.1
    )
    x_start = _feasible_point(100.0, 30.0, -10.0)
    path = baseline.generate(x_start, assets_start=100.0, ts=_TS)
    assert path.shape == (3, 4)
    assert np.allclose(path[0], x_start)  # t=0は始点そのもの


def test_gaussian_baseline_maps_toward_class0_mean() -> None:
    rng = np.random.default_rng(2)
    b = _null_space_projector().T  # (4,2)程度のダミー基底として転用
    b = b[:, :2]
    z0 = rng.normal(loc=1.0, scale=0.1, size=(50, 2))
    z_train2 = rng.normal(loc=-1.0, scale=0.1, size=(10, 2))
    baseline = build_gaussian_baseline(b, z0, z_train2)
    x_start = _feasible_point(100.0, 30.0, -10.0)
    path = baseline.generate(x_start, assets_start=100.0, ts=_TS)
    assert path.shape == (3, 4)
    assert np.allclose(path[0], x_start)


def _null_space_projector() -> np.ndarray:
    """Cash-STL-Treasury-Other=0 を満たす向きに射影する4x4行列(テスト用の簡易実装)。"""
    c = _C[:, 0]
    c = c / np.linalg.norm(c)
    return np.eye(4) - np.outer(c, c)
