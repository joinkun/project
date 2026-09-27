"""extension.py のテスト。局所名の切り出しと、cal 上の位置による写像の判定。"""

from __future__ import annotations

from fsgen.parsing._calc_types import CalcArc
from fsgen.parsing.calculation_link import CalcTree
from fsgen.parsing.extension import (
    EDINET_INDUSTRY_SUFFIXES,
    ExtensionRule,
    extension_local_name,
    nearest_subtotal,
)

_EXT = "jpcrp030000-asr_E01329-000_AccountsReceivableTradeAndContractAssets"
_CA = "jppfs_cor_CurrentAssets"
_NCA = "jppfs_cor_NoncurrentAssets"
_ASSETS = "jppfs_cor_Assets"
_SUBTOTALS = frozenset({_ASSETS, _CA, _NCA})


def _tree(arcs: list[tuple[str, str]]) -> CalcTree:
    elements = tuple(sorted({e for a in arcs for e in a}))
    return CalcTree(
        role="consolidated",
        role_uri="rol_ConsolidatedBalanceSheet",
        arcs=tuple(CalcArc(p, c, 1.0, None) for p, c in arcs),
        elements=elements,
        roots=(_ASSETS,),
    )


def _rule() -> ExtensionRule:
    return ExtensionRule.from_members(
        {
            "jppfs_cor_AccountsReceivableTradeAndContractAssets": "receivables_trade",
            "jppfs_cor_AccountsReceivableTrade": "receivables_trade",
        },
        {"receivables_trade": _CA},
    )


def test_extension_local_name() -> None:
    assert extension_local_name(_EXT) == "AccountsReceivableTradeAndContractAssets"
    assert extension_local_name("jpcrp030200-asr_E38190-000_DepositsReceivedCL") == (
        "DepositsReceivedCL"
    )
    assert extension_local_name("jppfs_cor_AccountsReceivableTrade") is None
    assert extension_local_name("jpcrp_cor_Something") is None


def test_nearest_subtotal() -> None:
    tree = _tree([(_ASSETS, _CA), (_CA, "x_Group"), ("x_Group", _EXT)])
    assert nearest_subtotal(_EXT, tree, _SUBTOTALS) == _CA
    assert nearest_subtotal(_ASSETS, tree, _SUBTOTALS) is None


def test_resolve_maps_exact_local_name_under_parent_subtotal() -> None:
    tree = _tree([(_ASSETS, _CA), (_CA, _EXT)])
    assert _rule().resolve(_EXT, tree, _SUBTOTALS) == "receivables_trade"


def test_resolve_rejects_different_parent_subtotal() -> None:
    tree = _tree([(_ASSETS, _NCA), (_NCA, _EXT)])
    assert _rule().candidate_group(_EXT) == "receivables_trade"
    assert _rule().resolve(_EXT, tree, _SUBTOTALS) is None


def test_resolve_requires_exact_match() -> None:
    near = "jpcrp030000-asr_E01329-000_AccountsReceivableTradeAndContractAssetsCA"
    tree = _tree([(_ASSETS, _CA), (_CA, near)])
    assert _rule().resolve(near, tree, _SUBTOTALS) is None
    assert _rule().resolve("jppfs_cor_AccountsReceivableTrade", tree, _SUBTOTALS) is None


def test_suffix_removal_uses_industry_codes_only() -> None:
    rule = ExtensionRule.from_members(
        {"jppfs_cor_AccountsReceivableTrade": "receivables_trade"},
        {"receivables_trade": _CA},
        EDINET_INDUSTRY_SUFFIXES,
    )
    cns = "jpcrp030000-asr_E00001-000_AccountsReceivableTradeCNS"
    ca = "jpcrp030000-asr_E00001-000_AccountsReceivableTradeCA"
    assert rule.match(cns) == ("receivables_trade", "CNS")
    assert rule.match(ca) is None  # 表示区分の略号は仕様に一覧が無いので除去しない
    tree = _tree([(_ASSETS, _NCA), (_NCA, cns)])
    assert rule.resolve(cns, tree, _SUBTOTALS) is None  # 親小計が違えば写像しない


def test_industry_suffixes_match_the_official_table() -> None:
    assert len(EDINET_INDUSTRY_SUFFIXES) == 21
    assert len(set(EDINET_INDUSTRY_SUFFIXES)) == 21
