from __future__ import annotations

import numpy as np

from fsgen.evaluation.paired import cliff_magnitude, cliffs_delta, paired_dominance, paired_test


def test_cliffs_delta_extremes() -> None:
    assert cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == -1.0
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_cliffs_delta_counts_pairs() -> None:
    # a=[1,3], b=[2]: 1<2, 3>2 → (1-1)/2 = 0
    assert cliffs_delta(np.array([1.0, 3.0]), np.array([2.0])) == 0.0
    # a=[3,3,1], b=[2]: 2 勝 1 敗 → 1/3
    assert np.isclose(cliffs_delta(np.array([3.0, 3.0, 1.0]), np.array([2.0])), 1 / 3)


def test_paired_dominance() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 2.0, 4.0, 3.0])
    assert paired_dominance(a, b) == (2 - 1) / 4


def test_cliff_magnitude() -> None:
    assert cliff_magnitude(0.1) == "無視できる"
    assert cliff_magnitude(-0.2) == "小"
    assert cliff_magnitude(0.4) == "中"
    assert cliff_magnitude(-0.9) == "大"


def test_paired_test_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=30), rng.normal(size=30)
    r1, r2 = paired_test(a, b), paired_test(a, b)
    assert r1 == r2
