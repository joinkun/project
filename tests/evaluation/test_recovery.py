"""recovery.py のテスト。小さな canonical で指標の定義どおりに計算されるかを確かめる。"""

from __future__ import annotations

import numpy as np
import pytest

from fsgen.evaluation.recovery import net_asset_decomposition, recovery_metrics

# 資産 2 科目、負債 5 科目(うち有利子 4)、純資産 2 科目
NAMES = (
    "cash",
    "ppe",
    "short_term_loans",
    "long_term_loans",
    "payables_trade",
    "capital",
    "retained",
    "bonds_current",
    "bonds_noncurrent",
)
MEMBERS = {
    "jppfs_cor_Assets": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    "jppfs_cor_NetAssets": np.array([0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=float),
}


def _x(*rows: list[float]) -> np.ndarray:
    """社債 2 列(0)を末尾に足す。"""
    return np.array([[*r, 0.0, 0.0] for r in rows], dtype=float)


def test_recovery_metrics() -> None:
    # 始点: 資産 100、負債 120(借入 100)、純資産 -20 → 終点: 増資 +40、借入返済 -20
    x0 = _x([40, 60, 50, 50, 20, 10, -30])
    x1 = _x([60, 60, 30, 50, 20, 50, -30])
    r = recovery_metrics(x0, x1, MEMBERS, NAMES)
    assert r.net_asset_improvement[0] == pytest.approx(40 / 100)
    assert bool(r.insolvent_start[0]) and bool(r.insolvency_resolved[0])
    assert r.debt_reduction[0] == pytest.approx(20 / 100)
    assert r.equity_ratio_change[0] == pytest.approx(20 / 120 - (-20) / 100)


def test_not_resolved_and_solvent_start() -> None:
    x0 = _x([40, 60, 50, 50, 20, 10, -30], [50, 50, 10, 10, 10, 50, 20])
    x1 = _x([40, 60, 50, 50, 20, 15, -30], [50, 50, 10, 10, 10, 50, 20])
    r = recovery_metrics(x0, x1, MEMBERS, NAMES)
    assert list(r.insolvent_start) == [True, False]
    assert list(r.insolvency_resolved) == [False, False]


def test_decomposition_sums_to_net_asset_improvement() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.uniform(1, 10, size=(5, 9))
    x1 = x0 + rng.normal(0, 1, size=(5, 9))
    parts = net_asset_decomposition(x0, x1, MEMBERS, NAMES)
    assert set(parts) == {"capital", "retained"}
    total = recovery_metrics(x0, x1, MEMBERS, NAMES).net_asset_improvement
    np.testing.assert_allclose(sum(parts.values()), total)


def test_rejects_nonpositive_assets() -> None:
    x = _x([0, 0, 1, 1, 1, 1, -4])
    with pytest.raises(ValueError, match="positive"):
        recovery_metrics(x, x, MEMBERS, NAMES)


def test_rejects_missing_debt_accounts() -> None:
    x = np.ones((1, 7))
    with pytest.raises(ValueError, match="interest-bearing"):
        recovery_metrics(x, x, {k: v[:7] for k, v in MEMBERS.items()}, NAMES[:7])
