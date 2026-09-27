"""partial_adjustment.py の骨格のテスト(入力検証と λ → c の変換)。"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fsgen.estimation.partial_adjustment import (
    AdjustmentResult,
    adjustment_cost,
    estimate_adjustment_speed,
    validate_panel,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edinetCode": ["E1", "E1", "E2", "E2"],
            "year": [2023, 2024, 2023, 2024],
            "assets": [100, 110, 50, 45],
            "inventories": [10, 12, 5, 4],
        }
    )


def test_adjustment_cost() -> None:
    assert adjustment_cost(1.0) == 0.0
    assert adjustment_cost(0.5) == pytest.approx(1.0)
    assert adjustment_cost(0.2) == pytest.approx(4.0)


@pytest.mark.parametrize("lam", [0.0, -0.1, 1.2, math.nan, math.inf])
def test_adjustment_cost_rejects_out_of_range(lam: float) -> None:
    with pytest.raises(ValueError, match="adjustment speed"):
        adjustment_cost(lam)


def test_result_cost_property() -> None:
    result = AdjustmentResult("inventories", "fe", "both", 0.25, 0.05, 0.15, 0.35, 100, 200)
    assert result.cost == pytest.approx(3.0)


def test_validate_panel_missing_column() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_panel(_panel().drop(columns=["assets"]), "inventories")
    with pytest.raises(ValueError, match="missing columns"):
        validate_panel(_panel(), "receivables_trade")


def test_validate_panel_duplicates() -> None:
    panel = pd.concat([_panel(), _panel().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicated"):
        validate_panel(panel, "inventories")


def test_estimate_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        estimate_adjustment_speed(_panel(), "inventories", "fe")


def test_estimate_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        estimate_adjustment_speed(_panel(), "inventories", "ols")  # type: ignore[arg-type]
