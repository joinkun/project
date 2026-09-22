"""balance_sheet.py のテスト。合成 canonical スペックで検証する(実データ非依存)。"""

from __future__ import annotations

import numpy as np

from fsgen.accounting.constraints import BasisSpec, CanonicalSpec, SubtotalSpec
from fsgen.viz.balance_sheet import LabelMap, build_bs_lines, render_markdown, render_text

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
_NAMES = tuple(b.name for b in _BASIS)

_LABELS = LabelMap(
    accounts={b.name: b.name for b in _BASIS}
    | {
        "jppfs_cor_CurrentAssets": "流動資産",
        "jppfs_cor_NoncurrentAssets": "固定資産",
        "jppfs_cor_CurrentLiabilities": "流動負債",
        "jppfs_cor_NoncurrentLiabilities": "固定負債",
        "jppfs_cor_ShareholdersEquity": "株主資本",
        "jppfs_cor_ValuationAndTranslationAdjustments": "評価・換算差額等",
        "jppfs_cor_Assets": "資産",
        "jppfs_cor_Liabilities": "負債",
        "jppfs_cor_NetAssets": "純資産",
        "jppfs_cor_LiabilitiesAndNetAssets": "負債純資産",
    },
    sections={"assets": "資産の部", "liabilities": "負債の部", "net_assets": "純資産の部"},
)


def _make_x(cash: float, retained: float) -> np.ndarray:
    values = dict.fromkeys(_NAMES, 0.0)
    values["Cash"] = cash
    values["RetainedEarnings"] = retained
    return np.array([values[n] for n in _NAMES])


def test_build_bs_lines_subtotals_equal_sum_of_children() -> None:
    x = _make_x(cash=1000.0, retained=-500.0)
    lines = build_bs_lines(_SPEC, _NAMES, [x], _LABELS)

    by_label = {line.label: line for line in lines}
    current_assets_total = by_label["流動資産合計"]
    assert current_assets_total.values is not None
    assert current_assets_total.values[0] == 1000.0  # Cash(1000) + OTHER(0)

    assets_total = by_label["資産合計"]
    assert assets_total.values is not None
    assert assets_total.values[0] == 1000.0  # CurrentAssets(1000) + NoncurrentAssets(0) + OTHER(0)


def test_build_bs_lines_diff_row_reflects_balance() -> None:
    x = _make_x(cash=1000.0, retained=1000.0)  # Assets=1000, NetAssets=1000, Liabilities=0
    lines = build_bs_lines(_SPEC, _NAMES, [x], _LABELS)
    diff_line = next(line for line in lines if line.kind == "diff")
    assert diff_line.values is not None
    assert diff_line.values[0] == 0.0  # 1000 - (0 + 1000) = 0


def test_build_bs_lines_diff_row_nonzero_when_unbalanced() -> None:
    x = _make_x(cash=1000.0, retained=500.0)  # Assets=1000, NetAssets=500, Liabilities=0
    lines = build_bs_lines(_SPEC, _NAMES, [x], _LABELS)
    diff_line = next(line for line in lines if line.kind == "diff")
    assert diff_line.values is not None
    assert diff_line.values[0] == 500.0


def test_render_text_contains_company_and_sections() -> None:
    x = _make_x(cash=1_000_000.0, retained=500_000.0)
    lines = build_bs_lines(_SPEC, _NAMES, [x], _LABELS)
    text = render_text(lines, (0.0,), "テスト株式会社", "S100TEST")
    assert "テスト株式会社" in text
    assert "資産の部" in text
    assert "負債の部" in text
    assert "純資産の部" in text
    assert "貸借差額" in text


def test_render_markdown_is_pipe_table() -> None:
    x = _make_x(cash=1_000_000.0, retained=500_000.0)
    lines = build_bs_lines(_SPEC, _NAMES, [x], _LABELS)
    md = render_markdown(lines, (0.0,), "テスト株式会社", "S100TEST")
    assert md.startswith("# テスト株式会社")
    assert "| 科目 |" in md
    assert "|---|" in md


def test_multiple_t_columns_and_change() -> None:
    x0 = _make_x(cash=1000.0, retained=0.0)
    x1 = _make_x(cash=2000.0, retained=0.0)
    lines = build_bs_lines(_SPEC, _NAMES, [x0, x1], _LABELS)
    by_label = {line.label: line for line in lines}
    cash_line = by_label["Cash"]
    assert cash_line.values == (1000.0, 2000.0)
