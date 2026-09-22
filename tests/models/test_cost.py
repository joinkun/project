"""cost.py のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fsgen.models.cost import (
    combined_projection_weights,
    cost_vectors,
    load_cost_config,
    low_dim_symmetric_cost_matrix,
    weighted_asymmetric_cost_matrix,
)

_NAMES = ("A", "B", "C")


def _write_costs_yaml(path: Path) -> None:
    doc = {
        "name": "costs",
        "basis_costs": [
            {"name": "A", "cost_increase": 1.0, "cost_decrease": 1.0, "source": "s"},
            {"name": "B", "cost_increase": 2.0, "cost_decrease": 5.0, "source": "s"},
            {"name": "C", "cost_increase": 1.0, "cost_decrease": 1.0, "source": "s"},
        ],
        "non_actionable": ["C"],
    }
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")


def test_load_cost_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    _write_costs_yaml(path)
    config = load_cost_config(path)
    assert len(config.accounts) == 3
    assert config.non_actionable == ("C",)
    b = next(a for a in config.accounts if a.name == "B")
    assert b.cost_increase == 2.0
    assert b.cost_decrease == 5.0


def test_cost_vectors_freeze_mode_sets_huge_cost(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    _write_costs_yaml(path)
    config = load_cost_config(path)

    w_plus, w_minus = cost_vectors(config, _NAMES, non_actionable_mode="freeze")
    assert w_plus[0] == 1.0 and w_minus[0] == 1.0  # A
    assert w_plus[1] == 2.0 and w_minus[1] == 5.0  # B
    assert w_plus[2] == pytest.approx(1e6)  # C (non_actionable)
    assert w_minus[2] == pytest.approx(1e6)


def test_cost_vectors_ignore_mode_sets_zero_cost(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    _write_costs_yaml(path)
    config = load_cost_config(path)
    w_plus, w_minus = cost_vectors(config, _NAMES, non_actionable_mode="ignore")
    assert w_plus[2] == 0.0
    assert w_minus[2] == 0.0
    assert w_plus[0] == 1.0  # 通常科目は変わらない


def test_cost_vectors_keep_mode_uses_raw_values(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    _write_costs_yaml(path)
    config = load_cost_config(path)
    w_plus, _w_minus = cost_vectors(config, _NAMES, non_actionable_mode="keep")
    assert w_plus[2] == 1.0  # 上書きされない


def test_cost_vectors_raises_on_missing_account(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    _write_costs_yaml(path)
    config = load_cost_config(path)
    with pytest.raises(KeyError):
        cost_vectors(config, ("A", "B", "UNKNOWN"))


def test_weighted_asymmetric_cost_matrix_matches_hand_computation() -> None:
    x = np.array([[0.0, 0.0]])  # 1社、2科目
    y = np.array([[1.0, -1.0], [0.0, 0.0]])  # 2社
    w_plus = np.array([2.0, 3.0])
    w_minus = np.array([5.0, 7.0])
    cost = weighted_asymmetric_cost_matrix(x, y, w_plus, w_minus)
    # y0: 科目0が+1(増加、w+=2) -> 2*1^2=2、科目1が-1(減少、w-=7) -> 7*1^2=7、合計9
    assert cost[0, 0] == pytest.approx(9.0)
    # y1: 差分0なので合計0
    assert cost[0, 1] == pytest.approx(0.0)


def test_weighted_asymmetric_cost_is_symmetric_when_weights_equal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3))
    y = rng.normal(size=(5, 3))
    w = np.array([2.0, 3.0, 1.0])
    cost_sym = weighted_asymmetric_cost_matrix(x, y, w, w)
    expected = np.sum(w[None, None, :] * (y[None, :, :] - x[:, None, :]) ** 2, axis=2)
    assert np.allclose(cost_sym, expected)


def test_low_dim_symmetric_cost_matrix_zero_for_identical_points() -> None:
    b = np.eye(4)[:, :2]
    z = np.array([[1.0, 2.0], [3.0, -1.0]])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    cost = low_dim_symmetric_cost_matrix(z, z, b, w)
    assert np.allclose(np.diag(cost), 0.0, atol=1e-9)


def test_low_dim_symmetric_cost_matrix_matches_direct_quadratic_form() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(size=(5, 2))
    z0 = rng.normal(size=(3, 2))
    z1 = rng.normal(size=(4, 2))
    w = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    cost = low_dim_symmetric_cost_matrix(z0, z1, b, w)

    x0 = z0 @ b.T
    x1 = z1 @ b.T
    expected = np.sum(w[None, None, :] * (x1[None, :, :] - x0[:, None, :]) ** 2, axis=2)
    assert np.allclose(cost, expected, atol=1e-8)


def test_combined_projection_weights_scales_with_max_cost() -> None:
    x_start = np.array([100.0, 0.0])
    w_plus = np.array([2.0, 1.0])
    w_minus = np.array([5.0, 1.0])
    weights = combined_projection_weights(x_start, assets=1000.0, w_plus=w_plus, w_minus=w_minus)
    # 科目0: max(2,5)=5 / (100 + 0.001*1000=1) = 5/101
    assert weights[0] == pytest.approx(5.0 / 101.0)
    # 科目1: max(1,1)=1 / (0 + 1) = 1
    assert weights[1] == pytest.approx(1.0)
