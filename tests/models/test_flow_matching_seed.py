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
