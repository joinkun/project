"""constraints.py のテスト。合成 canonical スペックで検証する(実データ非依存)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fsgen.accounting.constraints import (
    BasisSpec,
    CanonicalSpec,
    SubtotalSpec,
    build_constraint_matrix,
    build_projection,
    compute_rank,
    load_canonical_spec,
    orthonormal_basis,
    state_vector_names,
)

# 実際の core.yaml の形を最小限に縮小した合成スペック。
_BASIS = (
    BasisSpec("Cash", "jppfs_cor_CurrentAssets", "assets"),
    BasisSpec("__OTHER__jppfs_cor_CurrentAssets", "jppfs_cor_CurrentAssets", "assets"),
    BasisSpec("__OTHER__jppfs_cor_NoncurrentAssets", "jppfs_cor_NoncurrentAssets", "assets"),
    BasisSpec("__OTHER__jppfs_cor_Assets", "jppfs_cor_Assets", "assets"),
    BasisSpec("STL", "jppfs_cor_CurrentLiabilities", "liabilities"),
    BasisSpec(
        "__OTHER__jppfs_cor_CurrentLiabilities", "jppfs_cor_CurrentLiabilities", "liabilities"
    ),
    BasisSpec(
        "__OTHER__jppfs_cor_NoncurrentLiabilities",
        "jppfs_cor_NoncurrentLiabilities",
        "liabilities",
    ),
    BasisSpec("__OTHER__jppfs_cor_Liabilities", "jppfs_cor_Liabilities", "liabilities"),
    BasisSpec("RetainedEarnings", "jppfs_cor_ShareholdersEquity", "net_assets"),
    BasisSpec(
        "__OTHER__jppfs_cor_ShareholdersEquity", "jppfs_cor_ShareholdersEquity", "net_assets"
    ),
    BasisSpec(
        "__OTHER__jppfs_cor_ValuationAndTranslationAdjustments",
        "jppfs_cor_ValuationAndTranslationAdjustments",
        "net_assets",
    ),
    BasisSpec("__OTHER__jppfs_cor_NetAssets", "jppfs_cor_NetAssets", "net_assets"),
)

_SUBTOTALS = (
    SubtotalSpec("jppfs_cor_Assets", "assets"),
    SubtotalSpec("jppfs_cor_CurrentAssets", "assets"),
    SubtotalSpec("jppfs_cor_NoncurrentAssets", "assets"),
    SubtotalSpec("jppfs_cor_Liabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_CurrentLiabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NoncurrentLiabilities", "liabilities"),
    SubtotalSpec("jppfs_cor_NetAssets", "net_assets"),
    SubtotalSpec("jppfs_cor_ShareholdersEquity", "net_assets"),
    SubtotalSpec("jppfs_cor_ValuationAndTranslationAdjustments", "net_assets"),
)

_SPEC = CanonicalSpec(name="synthetic", basis=_BASIS, subtotals=_SUBTOTALS)

# 恒等式を厳密に満たす合成データ(会計的に整合した1社分):
# CurrentAssets=120(Cash100+OTHER20), NoncurrentAssets=80, Assets=200
# CurrentLiabilities=40, NoncurrentLiabilities=15, Liabilities=55
# ShareholdersEquity=55, VTA=10, NetAssets=145
# 200 = 55 + 145 (貸借一致)
_BASIS_VALUES = {
    "Cash": 100.0,
    "__OTHER__jppfs_cor_CurrentAssets": 20.0,
    "__OTHER__jppfs_cor_NoncurrentAssets": 80.0,
    "__OTHER__jppfs_cor_Assets": 0.0,
    "STL": 30.0,
    "__OTHER__jppfs_cor_CurrentLiabilities": 10.0,
    "__OTHER__jppfs_cor_NoncurrentLiabilities": 15.0,
    "__OTHER__jppfs_cor_Liabilities": 0.0,
    "RetainedEarnings": 50.0,
    "__OTHER__jppfs_cor_ShareholdersEquity": 5.0,
    "__OTHER__jppfs_cor_ValuationAndTranslationAdjustments": 10.0,
    "__OTHER__jppfs_cor_NetAssets": 80.0,
}
_SUBTOTAL_VALUES = {
    "jppfs_cor_Assets": 200.0,
    "jppfs_cor_CurrentAssets": 120.0,
    "jppfs_cor_NoncurrentAssets": 80.0,
    "jppfs_cor_Liabilities": 55.0,
    "jppfs_cor_CurrentLiabilities": 40.0,
    "jppfs_cor_NoncurrentLiabilities": 15.0,
    "jppfs_cor_NetAssets": 145.0,
    "jppfs_cor_ShareholdersEquity": 55.0,
    "jppfs_cor_ValuationAndTranslationAdjustments": 10.0,
}


def _vector(names: tuple[str, ...]) -> np.ndarray:
    merged = {**_BASIS_VALUES, **_SUBTOTAL_VALUES}
    return np.array([merged[n] for n in names])


def test_state_vector_names_preserves_yaml_order_basis_only() -> None:
    names = state_vector_names(_SPEC, include_subtotals=False)
    assert names == tuple(b.name for b in _BASIS)


def test_state_vector_names_appends_subtotals_in_order() -> None:
    names = state_vector_names(_SPEC, include_subtotals=True)
    assert names[: len(_BASIS)] == tuple(b.name for b in _BASIS)
    assert names[len(_BASIS) :] == tuple(s.element_id for s in _SUBTOTALS)


def test_constraint_matrix_false_mode_shape_and_rank() -> None:
    c = build_constraint_matrix(_SPEC, include_subtotals=False)
    assert c.shape == (len(_BASIS), 1)
    assert compute_rank(c) == 1


def test_constraint_matrix_true_mode_shape_and_rank() -> None:
    c = build_constraint_matrix(_SPEC, include_subtotals=True)
    dim = len(_BASIS) + len(_SUBTOTALS)
    k = len(_SUBTOTALS) + 1
    assert c.shape == (dim, k)
    assert compute_rank(c) == k


def test_effective_dimension_is_invariant_to_include_subtotals() -> None:
    c_false = build_constraint_matrix(_SPEC, include_subtotals=False)
    c_true = build_constraint_matrix(_SPEC, include_subtotals=True)
    d_false = c_false.shape[0] - compute_rank(c_false)
    d_true = c_true.shape[0] - compute_rank(c_true)
    assert d_false == d_true


@pytest.mark.parametrize("include_subtotals", [False, True])
def test_real_data_satisfies_constraints_exactly(include_subtotals: bool) -> None:
    names = state_vector_names(_SPEC, include_subtotals=include_subtotals)
    x = _vector(names)
    c = build_constraint_matrix(_SPEC, include_subtotals=include_subtotals)
    assert np.allclose(c.T @ x, 0.0, atol=1e-8)


@pytest.mark.parametrize("include_subtotals", [False, True])
def test_projection_is_idempotent_and_symmetric(include_subtotals: bool) -> None:
    c = build_constraint_matrix(_SPEC, include_subtotals=include_subtotals)
    p = build_projection(c)
    assert np.allclose(p @ p, p, atol=1e-8)
    assert np.allclose(p, p.T, atol=1e-8)


@pytest.mark.parametrize("include_subtotals", [False, True])
def test_projection_leaves_constraint_satisfying_data_unchanged(include_subtotals: bool) -> None:
    names = state_vector_names(_SPEC, include_subtotals=include_subtotals)
    x = _vector(names)
    c = build_constraint_matrix(_SPEC, include_subtotals=include_subtotals)
    p = build_projection(c)
    assert np.allclose(p @ x, x, atol=1e-6)


@pytest.mark.parametrize("include_subtotals", [False, True])
def test_orthonormal_basis_spans_null_space(include_subtotals: bool) -> None:
    c = build_constraint_matrix(_SPEC, include_subtotals=include_subtotals)
    basis = orthonormal_basis(c)
    dim = c.shape[0]
    rank = compute_rank(c)
    assert basis.shape == (dim, dim - rank)
    # 正規直交: Q^T Q = I
    assert np.allclose(basis.T @ basis, np.eye(dim - rank), atol=1e-8)
    # null(C^T) の基底であること: C^T @ basis ≈ 0
    assert np.allclose(c.T @ basis, 0.0, atol=1e-8)


def test_build_projection_uses_svd_not_explicit_inverse() -> None:
    """C の列がほぼ従属(ランク落ち)でも例外を出さず安定に計算できること。"""
    c = build_constraint_matrix(_SPEC, include_subtotals=True)
    # 最後の列を最初の列のほぼコピーにして意図的にランク落ちさせる。
    degenerate = c.copy()
    degenerate[:, -1] = degenerate[:, 0] * (1 + 1e-12)
    p = build_projection(degenerate)
    assert np.allclose(p @ p, p, atol=1e-6)


def test_load_canonical_spec_preserves_order(tmp_path: Path) -> None:
    doc = {
        "name": "test",
        "basis": [
            {
                "name": "A",
                "members": ["jppfs_cor_A"],
                "parent_subtotal": "jppfs_cor_S",
                "side": "assets",
                "sign": "free",
                "ratio": 1.0,
            },
            {
                "name": "B",
                "members": ["jppfs_cor_B"],
                "parent_subtotal": "jppfs_cor_S",
                "side": "assets",
                "sign": "free",
                "ratio": 0.5,
            },
        ],
        "subtotals": [{"element_id": "jppfs_cor_S", "side": "assets", "ratio": 1.0}],
        "totals": {"assets": "jppfs_cor_Assets"},
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    spec = load_canonical_spec(path)
    assert [b.name for b in spec.basis] == ["A", "B"]
    assert spec.subtotals[0].element_id == "jppfs_cor_S"
