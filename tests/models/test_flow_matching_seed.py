"""seeded_velocity_field のテスト(ステージ26 で見つかった乱数種の設定順序の誤りの再発防止)。"""

from __future__ import annotations

import torch

from fsgen.models.flow_matching import seeded_velocity_field


def _params(seed: int) -> list[torch.Tensor]:
    torch.manual_seed(12345)  # 呼び出し前の乱数状態に依存しないことも確かめる
    torch.rand(7)
    net = seeded_velocity_field(seed, dim=4, hidden=8, n_layers=2)
    return [p.detach().clone() for p in net.parameters()]


def test_same_seed_gives_identical_initialization() -> None:
    a, b = _params(3), _params(3)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert a[0].dtype == torch.float64


def test_different_seed_gives_different_initialization() -> None:
    a, b = _params(3), _params(4)
    assert not torch.equal(a[0], b[0])


def test_loss_metric_identity_matches_default() -> None:
    """ステージ38: loss_metric = I は従来の損失と同じ学習になる。"""
    import numpy as np

    from fsgen.models.flow_matching import seeded_velocity_field, train_fm

    rng = np.random.default_rng(0)
    z_train, z_pool = rng.normal(size=(6, 3)), rng.normal(size=(8, 3))
    plan = np.full((6, 8), 1 / 48)
    a = seeded_velocity_field(1, dim=3, hidden=8, n_layers=2)
    b = seeded_velocity_field(1, dim=3, hidden=8, n_layers=2)
    ra = train_fm(z_train, z_pool, plan, a, epochs=3, lr=1e-2, seed=0)
    rb = train_fm(z_train, z_pool, plan, b, epochs=3, lr=1e-2, seed=0, loss_metric=np.eye(3))
    assert np.allclose(ra.train_losses, rb.train_losses)


def test_loss_metric_equals_original_space_weighting() -> None:
    """e^T (B^T W B) e = ‖W^{1/2} B e‖²(方法 1 と方法 2 は同じ)。"""
    import numpy as np

    rng = np.random.default_rng(1)
    b = np.linalg.qr(rng.normal(size=(7, 3)))[0]
    w = rng.uniform(0.1, 5.0, size=7)
    e = rng.normal(size=3)
    lhs = e @ (b.T @ np.diag(w) @ b) @ e
    rhs = np.sum((np.sqrt(w) * (b @ e)) ** 2)
    assert np.isclose(lhs, rhs)
