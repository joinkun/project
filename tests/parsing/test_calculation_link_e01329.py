"""E01329 (S100W9LX) 固有の実測値テスト。

企業コード・実測値をハードコードしてよい唯一の箇所。値は手で数えて確認済み
(spec 3.4, 9)。これらの数値が出ない場合はパーサの実装を疑うこと。
"""

from __future__ import annotations

from pathlib import Path

from fsgen.parsing.calculation_link import CalcTree, find_cal_file, parse_calculation_link

_DOC_DIR = Path(__file__).parents[1] / "fixtures" / "S100W9LX"
_ROLE_MAP = {
    "rol_ConsolidatedBalanceSheet": "consolidated",
    "rol_BalanceSheet": "nonconsolidated",
}
_EXTENSION_ELEMENT = "jpcrp030000-asr_E01329-000_InvestmentsInCapitalAndGuaranteeDeposits"


def _trees() -> dict[str, CalcTree]:
    cal_path = find_cal_file(_DOC_DIR)
    return parse_calculation_link(cal_path, _ROLE_MAP)


def test_consolidated_element_and_arc_counts() -> None:
    consolidated = _trees()["consolidated"]
    assert len(consolidated.elements) == 67
    assert len(consolidated.arcs) == 65


def test_nonconsolidated_element_and_arc_counts() -> None:
    nonconsolidated = _trees()["nonconsolidated"]
    assert len(nonconsolidated.elements) == 48
    assert len(nonconsolidated.arcs) == 46


def test_consolidated_roots() -> None:
    consolidated = _trees()["consolidated"]
    assert set(consolidated.roots) == {
        "jppfs_cor_Assets",
        "jppfs_cor_LiabilitiesAndNetAssets",
    }


def test_assets_and_liabilities_and_net_assets_are_not_directly_linked() -> None:
    consolidated = _trees()["consolidated"]
    assert not any(
        arc.parent == "jppfs_cor_Assets" and arc.child == "jppfs_cor_LiabilitiesAndNetAssets"
        for arc in consolidated.arcs
    )


def test_extension_element_is_detected() -> None:
    consolidated = _trees()["consolidated"]
    assert _EXTENSION_ELEMENT in consolidated.diagnostics.extension_elements
