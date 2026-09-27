"""partial_adjustment.py のテスト。真の λ が既知の合成パネルで推定を確かめる。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fsgen.estimation.partial_adjustment import (
    AdjustmentResult,
    adjustment_cost,
    estimate_adjustment_speed,
    validate_frame,
)

_LAM = 0.3


def _synthetic(n: int = 20000, lam: float = _LAM, seed: int = 0) -> pd.DataFrame:
    """y_t = (1-λ) y_{t-1} + λ (m_t + a_i) + ε_t を 2022〜2024 年で生成する。

    業種 3 つで中央値相当の m_t が年ごとに動く。差分と前期は同じ「書類」から取るので、
    t=2024 の y_prev は t=2023 の y_cur と等しい(書類間の不一致なし)。
    """
    rng = np.random.default_rng(seed)
    industry = rng.integers(0, 3, size=n)
    m = {2022: np.array([0.2, 0.3, 0.4]), 2023: np.array([0.22, 0.29, 0.43])}
    m[2024] = np.array([0.25, 0.27, 0.41])
    a = rng.normal(0, 0.05, size=n)
    y = {2021: m[2022][industry] + a + rng.normal(0, 0.1, size=n)}
    for t in (2022, 2023, 2024):
        y[t] = (1 - lam) * y[t - 1] + lam * (m[t][industry] + a) + rng.normal(0, 0.03, size=n)
    size = rng.normal(10, 1, size=n)
    rows = []
    for t in (2023, 2024):
        rows.append(
            pd.DataFrame(
                {
                    "edinetCode": [f"E{i:05d}" for i in range(n)],
                    "t": t,
                    "y_prev": y[t - 1],
                    "y_cur": y[t],
                    "target": m[t][industry],
                    "size": size + rng.normal(0, 0.1, size=n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_adjustment_cost() -> None:
    assert adjustment_cost(1.0) == 0.0
    assert adjustment_cost(0.5) == pytest.approx(1.0)
    assert adjustment_cost(0.2) == pytest.approx(4.0)


@pytest.mark.parametrize("lam", [0.0, -0.1, 1.2, math.nan, math.inf])
def test_adjustment_cost_rejects_out_of_range(lam: float) -> None:
    with pytest.raises(ValueError, match="adjustment speed"):
        adjustment_cost(lam)


def test_result_cost_is_none_outside_unit_interval() -> None:
    ok = AdjustmentResult("a", "industry", "fe", "both", 0.25, 0.05, 10, 20)
    bad = AdjustmentResult("a", "industry", "fe", "both", 1.3, 0.05, 10, 20)
    assert ok.cost == pytest.approx(3.0)
    assert bad.cost is None and not bad.in_unit_interval


def test_validate_frame() -> None:
    frame = _synthetic(100)
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(frame.drop(columns=["target"]))
    with pytest.raises(ValueError, match="duplicated"):
        validate_frame(pd.concat([frame, frame.iloc[[0]]]))
    with pytest.raises(ValueError, match="consecutive"):
        validate_frame(frame.assign(t=frame["t"].replace({2024: 2025})))


def test_diff_gmm_recovers_lambda_industry_target() -> None:
    res = estimate_adjustment_speed(_synthetic(), "x", "diff_gmm", "industry")
    assert res.status == "ok"
    assert res.lam == pytest.approx(_LAM, abs=0.05)
    assert res.j_df == 1 and res.j_pval is not None and res.j_pval > 0.01
    assert res.first_stage_f is not None and res.first_stage_f > 10


def test_fe_is_biased_upward_with_two_periods() -> None:
    """T=2 の固定効果は Nickell バイアスで λ を過大に推定する(既知の性質)。"""
    res = estimate_adjustment_speed(_synthetic(), "x", "fe", "industry")
    assert res.status == "ok"
    assert res.lam > _LAM + 0.1


def test_firm_target_runs_and_recovers_lambda_with_gmm() -> None:
    """生成過程と一致する説明変数(業種の目標 m_t)を与えれば、GMM は λ を回復する。"""
    frame = _synthetic().assign(m=lambda d: d["target"])
    res = estimate_adjustment_speed(frame, "x", "diff_gmm", "firm", controls=("m",))
    assert res.status == "ok"
    assert res.lam == pytest.approx(_LAM, abs=0.08)


def test_direction_split_counts() -> None:
    frame = _synthetic(4000)
    both = estimate_adjustment_speed(frame, "x", "fe", "industry")
    pos = estimate_adjustment_speed(frame, "x", "fe", "industry", "positive")
    neg = estimate_adjustment_speed(frame, "x", "fe", "industry", "negative")
    assert pos.n_obs + neg.n_obs <= both.n_obs
    assert pos.n_obs > 0 and neg.n_obs > 0


def test_insufficient_obs_is_reported_not_raised() -> None:
    res = estimate_adjustment_speed(_synthetic(10), "x", "diff_gmm", "industry")
    assert res.status == "insufficient_obs" and math.isnan(res.lam)


def test_rejects_unknown_arguments() -> None:
    frame = _synthetic(100)
    with pytest.raises(ValueError, match="method"):
        estimate_adjustment_speed(frame, "x", "ols")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="controls"):
        estimate_adjustment_speed(frame, "x", "fe", "firm")


def test_rank_deficiency_is_reported() -> None:
    frame = _synthetic(500).assign(y_prev=0.0, y_cur=0.0, target=0.0)
    res = estimate_adjustment_speed(frame, "x", "fe", "industry")
    assert res.status == "rank_deficient" and math.isnan(res.lam)
