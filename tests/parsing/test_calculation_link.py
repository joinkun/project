"""_cal.xml パーサの不変条件テスト。fixture を glob して自動 parametrize する。

企業コード固有の実測値は test_calculation_link_e01329.py に分離してある。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from fsgen.parsing.calculation_link import (
    CalcArc,
    CalcTree,
    find_cal_file,
    is_extension_element,
    parse_calculation_link,
    select_tree,
)

_TESTS_DIR = Path(__file__).parents[1]
_FIXTURES_DIR = _TESTS_DIR / "fixtures"
_CONFIG_PATH = _TESTS_DIR.parent / "conf" / "config.yaml"

FIXTURE_DIRS = sorted(p for p in _FIXTURES_DIR.iterdir() if p.is_dir())

with _CONFIG_PATH.open() as _f:
    _CALC_LINK_CONFIG = yaml.safe_load(_f)["calculation_link"]

ROLE_MAP: dict[str, str] = _CALC_LINK_CONFIG["balance_sheet_roles"]
STANDARD_PREFIXES: tuple[str, ...] = tuple(_CALC_LINK_CONFIG["standard_prefixes"])

_LINK_NS = "http://www.xbrl.org/2003/linkbase"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_BS_ROLE_URI = "http://disclosure.edinet-fsa.go.jp/role/jppfs/rol_ConsolidatedBalanceSheet"


def _parse_fixture(doc_dir: Path) -> dict[str, CalcTree]:
    cal_path = find_cal_file(doc_dir)
    return parse_calculation_link(cal_path, ROLE_MAP, STANDARD_PREFIXES)


def _write_cal_xml(tmp_path: Path, body: str) -> Path:
    public_doc = tmp_path / "DOC" / "XBRL" / "PublicDoc"
    public_doc.mkdir(parents=True)
    cal_path = public_doc / "sample_cal.xml"
    cal_path.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<link:linkbase xmlns:link="{_LINK_NS}" xmlns:xlink="{_XLINK_NS}">\n'
        f"{body}\n"
        f"</link:linkbase>\n"
    )
    return cal_path


# --- fixture 不変条件 (spec section 7) ---------------------------------


@pytest.mark.parametrize("doc_dir", FIXTURE_DIRS, ids=lambda p: p.name)
def test_at_least_one_tree(doc_dir: Path) -> None:
    trees = _parse_fixture(doc_dir)
    assert len(trees) >= 1


@pytest.mark.parametrize("doc_dir", FIXTURE_DIRS, ids=lambda p: p.name)
def test_all_arc_endpoints_are_known_elements(doc_dir: Path) -> None:
    trees = _parse_fixture(doc_dir)
    for tree in trees.values():
        for arc in tree.arcs:
            assert arc.parent in tree.elements
            assert arc.child in tree.elements


@pytest.mark.parametrize("doc_dir", FIXTURE_DIRS, ids=lambda p: p.name)
def test_no_cycles(doc_dir: Path) -> None:
    trees = _parse_fixture(doc_dir)
    for tree in trees.values():
        assert tree.diagnostics.cycles == ()


@pytest.mark.parametrize("doc_dir", FIXTURE_DIRS, ids=lambda p: p.name)
def test_roots_nonempty(doc_dir: Path) -> None:
    trees = _parse_fixture(doc_dir)
    for tree in trees.values():
        assert len(tree.roots) > 0


@pytest.mark.parametrize("doc_dir", FIXTURE_DIRS, ids=lambda p: p.name)
def test_depth_is_finite_for_every_element(doc_dir: Path) -> None:
    trees = _parse_fixture(doc_dir)
    for tree in trees.values():
        for element in tree.elements:
            assert tree.depth_of(element) >= 0


@pytest.mark.parametrize("doc_dir", FIXTURE_DIRS, ids=lambda p: p.name)
def test_select_tree_always_returns_a_tree(doc_dir: Path) -> None:
    trees = _parse_fixture(doc_dir)
    tree, kind = select_tree(trees)
    assert trees[kind] is tree


@pytest.mark.parametrize("doc_dir", FIXTURE_DIRS, ids=lambda p: p.name)
def test_no_zero_weight_arcs(doc_dir: Path) -> None:
    trees = _parse_fixture(doc_dir)
    for tree in trees.values():
        for arc in tree.arcs:
            assert arc.weight != 0


# --- find_cal_file --------------------------------------------------------


def test_find_cal_file_raises_when_missing(tmp_path: Path) -> None:
    (tmp_path / "XBRL" / "PublicDoc").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        find_cal_file(tmp_path)


def test_find_cal_file_raises_when_multiple(tmp_path: Path) -> None:
    public_doc = tmp_path / "XBRL" / "PublicDoc"
    public_doc.mkdir(parents=True)
    (public_doc / "a_cal.xml").write_text("<x/>")
    (public_doc / "b_cal.xml").write_text("<x/>")
    with pytest.raises(ValueError, match="expected exactly one"):
        find_cal_file(tmp_path)


# --- is_extension_element --------------------------------------------------


def test_is_extension_element() -> None:
    prefixes = ("jppfs_cor", "jpcrp_cor", "jpdei_cor")
    assert not is_extension_element("jppfs_cor_Assets", prefixes)
    assert is_extension_element(
        "jpcrp030000-asr_E01329-000_InvestmentsInCapitalAndGuaranteeDeposits", prefixes
    )


# --- 合成データによる単体テスト --------------------------------------------

_ARCROLE = "http://www.xbrl.org/2003/arcrole/summation-item"


def _loc(label: str, href: str | None = None) -> str:
    return f'  <link:loc xlink:href="t.xsd#{href or label}" xlink:label="{label}" />'


def _arc(from_label: str, to_label: str, weight: str = "1", extra: str = "") -> str:
    return (
        f'  <link:calculationArc xlink:from="{from_label}" xlink:to="{to_label}"\n'
        f'      xlink:arcrole="{_ARCROLE}" order="1" weight="{weight}" {extra}/>'
    )


def _link(*lines: str) -> str:
    inner = "\n".join(lines)
    return f'<link:calculationLink xlink:role="{_BS_ROLE_URI}">\n{inner}\n</link:calculationLink>'


def test_label_and_href_can_differ(tmp_path: Path) -> None:
    body = _link(
        _loc("loc_1", href="jppfs_cor_Assets"),
        _loc("loc_2", href="jppfs_cor_CurrentAssets"),
        _arc("loc_1", "loc_2"),
    )
    cal_path = _write_cal_xml(tmp_path, body)
    trees = parse_calculation_link(cal_path, {"rol_ConsolidatedBalanceSheet": "consolidated"})
    assert trees["consolidated"].arcs == (
        CalcArc(parent="jppfs_cor_Assets", child="jppfs_cor_CurrentAssets", weight=1.0, order=1.0),
    )


def test_negative_weight_is_read(tmp_path: Path) -> None:
    body = _link(
        _loc("jppfs_cor_NetAssets"),
        _loc("jppfs_cor_TreasuryStock"),
        _arc("jppfs_cor_NetAssets", "jppfs_cor_TreasuryStock", weight="-1"),
    )
    cal_path = _write_cal_xml(tmp_path, body)
    trees = parse_calculation_link(cal_path, {"rol_ConsolidatedBalanceSheet": "consolidated"})
    assert trees["consolidated"].arcs[0].weight == -1.0


def test_prohibited_arc_is_excluded_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    body = _link(
        _loc("jppfs_cor_A"),
        _loc("jppfs_cor_B"),
        _arc("jppfs_cor_A", "jppfs_cor_B", extra='use="prohibited" '),
    )
    cal_path = _write_cal_xml(tmp_path, body)
    with caplog.at_level(logging.WARNING):
        trees = parse_calculation_link(cal_path, {"rol_ConsolidatedBalanceSheet": "consolidated"})
    tree = trees["consolidated"]
    assert tree.arcs == ()
    assert len(tree.diagnostics.prohibited_arcs) == 1
    assert any("prohibited" in record.message for record in caplog.records)


def test_multiple_parents_detected(tmp_path: Path) -> None:
    body = _link(
        _loc("jppfs_cor_A"),
        _loc("jppfs_cor_B"),
        _loc("jppfs_cor_C"),
        _arc("jppfs_cor_A", "jppfs_cor_C"),
        _arc("jppfs_cor_B", "jppfs_cor_C"),
    )
    cal_path = _write_cal_xml(tmp_path, body)
    trees = parse_calculation_link(cal_path, {"rol_ConsolidatedBalanceSheet": "consolidated"})
    assert trees["consolidated"].diagnostics.multi_parent_elements == ("jppfs_cor_C",)


def test_cycle_detected(tmp_path: Path) -> None:
    body = _link(
        _loc("jppfs_cor_A"),
        _loc("jppfs_cor_B"),
        _arc("jppfs_cor_A", "jppfs_cor_B"),
        _arc("jppfs_cor_B", "jppfs_cor_A"),
    )
    cal_path = _write_cal_xml(tmp_path, body)
    trees = parse_calculation_link(cal_path, {"rol_ConsolidatedBalanceSheet": "consolidated"})
    assert len(trees["consolidated"].diagnostics.cycles) >= 1


# --- 実データが無く検証できない挙動 (spec section 8) ------------------------


@pytest.mark.skip(
    reason="連結なし企業(nonconsolidated のみ)の fixture が無く未検証。"
    "select_tree のフォールバック自体は合成データで確認済み。"
)
def test_select_tree_falls_back_when_no_consolidated_company_exists() -> None:
    ...


@pytest.mark.skip(
    reason="貸借対照表 role 内で weight=-1 が実際に使われる fixture が無く未検証。"
    "PL/CF 等の他 role には実例があるが、BS 限定では確認できていない。"
)
def test_negative_weight_real_example_in_balance_sheet() -> None:
    ...
