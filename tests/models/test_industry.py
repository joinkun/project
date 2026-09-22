"""industry.py と業種コスト(cost.py)・正規化済みSinkhorn(transport.py)のテスト。"""

from __future__ import annotations

import numpy as np

from fsgen.models.cost import (
    hard_industry_cost,
    industry_mismatch,
    soft_industry_cost,
)
from fsgen.models.industry import assign_blocks, block_marginals
from fsgen.models.transport import solve_ot_from_cost_verbose, solve_ot_normalized

_COARSE = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "z": "Z"}


def test_soft_lambda_zero_equals_legacy_plan() -> None:
    """§2.1: λ=0 は現行(solve_ot_from_cost)と厳密に同じ輸送計画になる。"""
    rng = np.random.default_rng(0)
    cost = rng.uniform(0.1, 3.0, size=(6, 9))
    mismatch = industry_mismatch(np.array(list("aabbcc")), np.array(list("abcabcabc")))
    legacy = solve_ot_from_cost_verbose(cost, 6, 9, 0.5, log_domain=False).plan
    normalized = soft_industry_cost(cost, mismatch, 0.0)
    a, b = np.full(6, 1 / 6), np.full(9, 1 / 9)
    new = solve_ot_normalized(normalized, a, b, 0.5, log_domain=False, max_iter=1000).plan
    np.testing.assert_array_equal(legacy, new)


def test_soft_penalty_is_multiple_of_median() -> None:
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    mismatch = np.array([[False, True], [True, False]])
    out = soft_industry_cost(cost, mismatch, 2.0)
    base = cost / np.median(cost)
    np.testing.assert_allclose(out[mismatch], base[mismatch] + 2.0)
    np.testing.assert_allclose(out[~mismatch], base[~mismatch])


def test_hard_cost_normalizes_before_penalty() -> None:
    cost = np.arange(1.0, 7.0).reshape(2, 3)
    mismatch = np.array([[False, True, True], [True, True, False]])
    out = hard_industry_cost(cost, mismatch)
    base = cost / np.median(cost)
    np.testing.assert_allclose(out[~mismatch], base[~mismatch])
    assert np.all(out[mismatch] == 1e10)


def test_log_domain_matches_standard_when_standard_converges() -> None:
    rng = np.random.default_rng(1)
    cost = rng.uniform(0.5, 2.0, size=(5, 7))
    a, b = np.full(5, 1 / 5), np.full(7, 1 / 7)
    std = solve_ot_normalized(cost, a, b, 0.5, log_domain=False, max_iter=5000)
    log = solve_ot_normalized(cost, a, b, 0.5, log_domain=True)
    assert std.converged and log.converged
    np.testing.assert_allclose(std.plan, log.plan, atol=1e-6)


def test_log_domain_survives_underflow_where_standard_fails() -> None:
    cost = np.full((4, 6), 1.0)
    cost[0, :] = 900.0  # exp(-900/0.05) は線形領域で全要素アンダーフロー
    a, b = np.full(4, 1 / 4), np.full(6, 1 / 6)
    std = solve_ot_normalized(
        cost, a, b, 0.05, log_domain=False, max_iter=1000, require_converged=False
    )
    log = solve_ot_normalized(cost, a, b, 0.05, log_domain=True)
    assert not std.converged
    assert log.converged and log.n_zero_mass_rows == 0


def test_assign_blocks_33_no_fallback() -> None:
    src = ["a1"] * 3 + ["b1"] * 2
    tgt = ["a1"] * 12 + ["b1"] * 15 + ["z"] * 20
    ba = assign_blocks(src, tgt, _COARSE, "33", min_targets=10)
    assert ba.src_level == ["33"] * 5
    assert ba.src_block[0] != ba.src_block[3]
    a, b = block_marginals(ba.src_block, ba.tgt_block)
    assert np.isclose(a.sum(), 1.0) and np.isclose(b.sum(), 1.0)
    assert np.all(b[np.array(tgt) == "z"] == 0.0)  # 始点のいない業種の目標は質量0


def test_assign_blocks_falls_back_to_coarse_then_all() -> None:
    # a1の目標が3社(<10)→粗い分類A(a1+a2)へ。a2は12社なので併合後15社で足りる。
    src = ["a1", "b1"]
    tgt = ["a1"] * 3 + ["a2"] * 12 + ["b1"] * 10
    ba = assign_blocks(src, tgt, _COARSE, "33", min_targets=10)
    assert ba.src_level == ["coarse", "33"]
    assert ba.src_block[0] != ba.src_block[1]
    # zは目標が0社で粗い分類Zにも無い→業種制約なし。他の始点の制約は消えない。
    ba2 = assign_blocks(
        ["z", "a1", "b1"], ["a1"] * 12 + ["b1"] * 12, _COARSE, "33", min_targets=10
    )
    assert ba2.src_level == ["all", "33", "33"]
    assert ba2.src_block[0] == -1
    assert ba2.src_block[1] != ba2.src_block[2]
    a, b = block_marginals(ba2.src_block, ba2.tgt_block)
    assert np.isclose(b.sum(), 2 / 3)


def test_coarse_level_starts_from_coarse_groups() -> None:
    src = ["a1", "a2", "b1"]
    tgt = ["a1"] * 6 + ["a2"] * 6 + ["b1"] * 12
    ba = assign_blocks(src, tgt, _COARSE, "coarse", min_targets=10)
    assert ba.src_block[0] == ba.src_block[1] != ba.src_block[2]
    assert ba.src_level == ["coarse"] * 3
