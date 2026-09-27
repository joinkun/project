"""target_filter.py のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from fsgen.models.target_filter import (
    DISALLOWED_PENALTY,
    apply_mask,
    equity_ratio,
    target_mask,
)

MEMBERS = {
    "jppfs_cor_Assets": np.array([1.0, 1.0, 0.0, 0.0]),
    "jppfs_cor_NetAssets": np.array([0.0, 0.0, 0.0, 1.0]),
}


def test_equity_ratio() -> None:
    x = np.array([[60.0, 40.0, 70.0, 30.0], [10.0, 10.0, 25.0, -5.0]])
    np.testing.assert_allclose(equity_ratio(x, MEMBERS), [0.3, -0.25])


def test_baseline_allows_all() -> None:
    r = target_mask("baseline", np.array([0.1, -0.5]), np.array([0.2, 0.4, 0.6]))
    assert r.allowed.all() and not r.fallback.any() and r.threshold is None


def test_relative_depends_on_source() -> None:
    r = target_mask("relative", np.array([0.3, -0.5]), np.array([0.2, 0.4, 0.6]))
    assert r.allowed.tolist() == [[False, True, True], [True, True, True]]
    assert r.n_candidates.tolist() == [2, 3]


def test_median_and_absolute() -> None:
    tgt = np.array([0.1, 0.2, 0.5, 0.7])
    med = target_mask("median", np.array([0.0]), tgt)
    assert med.threshold == pytest.approx(0.35)
    assert med.allowed.tolist() == [[False, False, True, True]]
    ab = target_mask("absolute", np.array([0.0]), tgt, absolute_threshold=0.5)
    assert ab.allowed.tolist() == [[False, False, False, True]]


def test_fallback_when_no_candidate() -> None:
    r = target_mask("relative", np.array([0.9]), np.array([0.2, 0.4]))
    assert r.n_candidates.tolist() == [0]
    assert r.fallback.tolist() == [True]
    assert r.allowed.all()


def test_apply_mask() -> None:
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = apply_mask(cost, np.array([[True, False], [False, True]]))
    assert out.tolist() == [[1.0, DISALLOWED_PENALTY], [DISALLOWED_PENALTY, 4.0]]
    assert int(np.argmin(out[0])) == 0
