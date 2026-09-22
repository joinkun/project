"""flow_matching.py のテスト。合成データで検証する(実データ非依存)。"""

from __future__ import annotations

import numpy as np
import torch

from fsgen.accounting.constraints import BasisSpec, CanonicalSpec, SubtotalSpec
from fsgen.models.differentiable_projection import build_projection_layer
from fsgen.models.flow_matching import (
    VelocityField,
    euler_sample,
    euler_sample_batch,
    sample_ot_targets,
    train_cafm,
    train_fm,
)

_TS = (0.0, 0.5, 1.0)


def test_velocity_field_output_shape() -> None:
    net = VelocityField(dim=4, hidden=8, n_layers=2)
    z = torch.zeros(5, 4, dtype=torch.float32)
    t = torch.zeros(5, dtype=torch.float32)
    net = net.double()
    out = net(z.double(), t.double())
    assert out.shape == (5, 4)


def test_sample_ot_targets_respects_distribution() -> None:
    rng = np.random.default_rng(0)
    plan_row = np.array([0.0, 0.0, 1.0, 0.0])
    for _ in range(10):
        assert sample_ot_targets(plan_row, rng) == 2


def test_sample_ot_targets_falls_back_to_uniform_for_zero_mass_row() -> None:
    """孤立企業(輸送計画の行がほぼ0)ではNaNにならず、一様分布から選ばれる。"""
    rng = np.random.default_rng(0)
    plan_row = np.zeros(5)
    result = sample_ot_targets(plan_row, rng)
    assert 0 <= result < 5


def test_train_fm_reduces_loss_on_trivial_problem() -> None:
    """始点が全て同じ、終点も全て同じ(定数場)なら損失は単調に近く減るはず。"""
    z_train = np.zeros((6, 3))
    z_pool = np.ones((6, 3))
    plan = np.eye(6)  # 各社が対応する1点だけに全質量
    net = VelocityField(dim=3, hidden=16, n_layers=2).double()
    result = train_fm(z_train, z_pool, plan, net, epochs=200, lr=1e-2, seed=0)
    assert result.train_losses[-1] < result.train_losses[0]
    assert result.train_losses[-1] < 0.05


def test_train_fm_records_validation_loss_when_provided() -> None:
    z_train = np.zeros((4, 2))
    z_pool = np.ones((4, 2))
    plan = np.eye(4)
    z_val = np.zeros((2, 2))
    plan_val = np.eye(2, 4)
    net = VelocityField(dim=2, hidden=8, n_layers=1).double()
    result = train_fm(
        z_train, z_pool, plan, net, epochs=10, lr=1e-2, seed=1, z_val=z_val, plan_val=plan_val
    )
    assert len(result.val_losses) == 10


def test_euler_sample_starts_at_z_start() -> None:
    net = VelocityField(dim=3, hidden=8, n_layers=1).double()
    z_start = np.array([1.0, 2.0, 3.0])
    path = euler_sample(net, z_start, _TS)
    assert path.shape == (3, 3)
    assert np.allclose(path[0], z_start)


def test_euler_sample_batch_matches_single_sample() -> None:
    net = VelocityField(dim=3, hidden=8, n_layers=1).double()
    z_start = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 0.5]])
    batch_path = euler_sample_batch(net, z_start, _TS)
    for i in range(2):
        single_path = euler_sample(net, z_start[i], _TS)
        assert np.allclose(batch_path[i], single_path)


_CAFM_BASIS = (
    BasisSpec("Cash", "jppfs_cor_CurrentAssets", "assets", sign="nonneg"),
    BasisSpec("STL", "jppfs_cor_CurrentLiabilities", "liabilities", sign="nonneg"),
    BasisSpec("Treasury", "jppfs_cor_ShareholdersEquity", "net_assets", sign="nonpos"),
    BasisSpec("RE", "jppfs_cor_ShareholdersEquity", "net_assets", sign="free"),
)
_CAFM_SUBTOTALS = (
    SubtotalSpec("jppfs_cor_Assets", "assets"),
    SubtotalSpec("jppfs_cor_Liabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NetAssets", "net_assets"),
)
_CAFM_SPEC = CanonicalSpec(name="synthetic", basis=_CAFM_BASIS, subtotals=_CAFM_SUBTOTALS)


def _feasible_point(cash: float, stl: float, treasury: float) -> np.ndarray:
    re = cash - stl - treasury
    return np.array([cash, stl, treasury, re])


def test_train_cafm_runs_and_reduces_cafm_loss() -> None:
    """train_cafmがz->x->射影->zの往復を正しく行い、損失が発散しないことを確認する。"""
    rng = np.random.default_rng(0)
    n_train, n_pool, dim, k = 4, 6, 4, 2
    x_start_train = np.stack(
        [_feasible_point(1.0 + 0.1 * i, 0.3, -0.1) for i in range(n_train)]
    )
    # bの列は恒等式(Cash-STL-Treasury-RE=0)を満たす向きに射影しておく
    c = np.array([1.0, -1.0, -1.0, -1.0])
    c = c / np.linalg.norm(c)
    projector = np.eye(dim) - np.outer(c, c)
    b = (rng.normal(size=(dim, k)) @ np.linalg.qr(rng.normal(size=(k, k)))[0])
    b = projector @ b
    b, _ = np.linalg.qr(b)  # 正規直交化

    z_train = x_start_train @ b
    z_pool = (rng.normal(size=(n_pool, dim)) * 0.1 + x_start_train.mean(axis=0)) @ b
    plan = np.full((n_train, n_pool), 1.0 / n_pool)

    layers = [
        build_projection_layer(
            _CAFM_SPEC, x_start_train[i], 1.0 / (np.abs(x_start_train[i]) + 0.1)
        )
        for i in range(n_train)
    ]
    net = VelocityField(dim=k, hidden=16, n_layers=2).double()
    result = train_cafm(
        z_train, z_pool, plan, net, layers, x_start_train, b, epochs=15, lr=1e-2, seed=0
    )
    assert len(result.cafm_losses) == 15
    assert all(np.isfinite(v) for v in result.cafm_losses)
    assert all(np.isfinite(v) for v in result.train_losses)
