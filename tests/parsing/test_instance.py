"""instance.py のテスト。合成XBRLインスタンスで検証する(実データ非依存)。"""

from __future__ import annotations

import pytest
from lxml import etree

from fsgen.parsing.instance import (
    CONSOLIDATED_CONTEXT,
    NONCONSOLIDATED_CONTEXT,
    PRIOR_CONSOLIDATED_CONTEXT,
    PRIOR_NONCONSOLIDATED_CONTEXT,
    FactValue,
    context_id_for_role,
    context_instant,
    duration_context_id_for_role,
    read_fact,
    read_text_block_present,
    resolve_element_tag,
    split_element_id,
    verify_context_period,
)

_INSTANCE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl
    xmlns:xbrli="http://www.xbrl.org/2003/instance"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:jppfs_cor="http://disclosure.edinet-fsa.go.jp/taxonomy/jppfs/2023-11-01/jppfs_cor"
    xmlns:jpcrp_cor="http://disclosure.edinet-fsa.go.jp/taxonomy/jpcrp/2023-11-01/jpcrp_cor"
    xmlns:jpcrp030000-asr_E04641-000="http://disclosure.edinet-fsa.go.jp/jpcrp030000/asr/001/E04641-000">
  <xbrli:context id="CurrentYearInstant">
    <xbrli:entity>
      <xbrli:identifier scheme="http://disclosure.edinet-fsa.go.jp">E00001-000</xbrli:identifier>
    </xbrli:entity>
    <xbrli:period>
      <xbrli:instant>2024-03-31</xbrli:instant>
    </xbrli:period>
  </xbrli:context>
  <xbrli:context id="CurrentYearInstant_NonConsolidatedMember">
    <xbrli:entity>
      <xbrli:identifier scheme="http://disclosure.edinet-fsa.go.jp">E00001-000</xbrli:identifier>
    </xbrli:entity>
    <xbrli:period>
      <xbrli:instant>2024-03-31</xbrli:instant>
    </xbrli:period>
  </xbrli:context>
  <xbrli:context id="Prior1YearInstant">
    <xbrli:entity>
      <xbrli:identifier scheme="http://disclosure.edinet-fsa.go.jp">E00001-000</xbrli:identifier>
    </xbrli:entity>
    <xbrli:period>
      <xbrli:instant>2023-03-31</xbrli:instant>
    </xbrli:period>
  </xbrli:context>
  <xbrli:unit id="JPY">
    <xbrli:measure>iso4217:JPY</xbrli:measure>
  </xbrli:unit>
  {body}
</xbrli:xbrl>
"""


def _parse(body: str) -> etree._Element:
    return etree.fromstring(_INSTANCE_TEMPLATE.format(body=body).encode("utf-8"))


def test_context_id_for_role() -> None:
    assert context_id_for_role("consolidated") == CONSOLIDATED_CONTEXT
    assert context_id_for_role("nonconsolidated") == NONCONSOLIDATED_CONTEXT


def test_context_id_for_role_prior() -> None:
    assert context_id_for_role("consolidated", "prior") == PRIOR_CONSOLIDATED_CONTEXT
    assert context_id_for_role("nonconsolidated", "prior") == PRIOR_NONCONSOLIDATED_CONTEXT
    assert PRIOR_CONSOLIDATED_CONTEXT == "Prior1YearInstant"
    assert PRIOR_NONCONSOLIDATED_CONTEXT == "Prior1YearInstant_NonConsolidatedMember"


def test_context_id_for_role_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="role"):
        context_id_for_role("segment")
    with pytest.raises(ValueError, match="period"):
        context_id_for_role("consolidated", "prior2")  # type: ignore[arg-type]


def test_split_element_id() -> None:
    assert split_element_id("jppfs_cor_Assets") == ("jppfs_cor", "Assets")
    assert split_element_id("jpcrp_cor_SomeTextBlock") == ("jpcrp_cor", "SomeTextBlock")


def test_context_instant_and_verify() -> None:
    root = _parse("")
    assert context_instant(root, "CurrentYearInstant") == "2024-03-31"
    assert context_instant(root, "NoSuchContext") is None
    assert verify_context_period(root, "CurrentYearInstant", "2024-03-31") is True
    assert verify_context_period(root, "CurrentYearInstant", "2023-03-31") is False


def test_read_fact_returns_integer_value_without_rounding() -> None:
    root = _parse(
        '<jppfs_cor:Assets contextRef="CurrentYearInstant" unitRef="JPY" '
        'decimals="-3">108594000000</jppfs_cor:Assets>'
    )
    fact = read_fact(root, "jppfs_cor_Assets", "CurrentYearInstant")
    assert fact == FactValue("jppfs_cor_Assets", "CurrentYearInstant", 108594000000, "-3", "JPY")
    assert fact.is_missing is False


def test_read_fact_missing_element_returns_none_value() -> None:
    root = _parse("")
    fact = read_fact(root, "jppfs_cor_Assets", "CurrentYearInstant")
    assert fact.value is None
    assert fact.is_missing is True


def test_read_fact_xsi_nil_is_missing_not_zero() -> None:
    root = _parse(
        '<jppfs_cor:Assets xsi:nil="true" contextRef="CurrentYearInstant" unitRef="JPY"/>'
    )
    fact = read_fact(root, "jppfs_cor_Assets", "CurrentYearInstant")
    assert fact.value is None
    assert fact.is_missing is True


def test_read_fact_only_matches_requested_context() -> None:
    root = _parse(
        '<jppfs_cor:Assets contextRef="CurrentYearInstant" unitRef="JPY">100</jppfs_cor:Assets>'
        '<jppfs_cor:Assets contextRef="CurrentYearInstant_NonConsolidatedMember" '
        'unitRef="JPY">50</jppfs_cor:Assets>'
    )
    consolidated = read_fact(root, "jppfs_cor_Assets", "CurrentYearInstant")
    nonconsolidated = read_fact(
        root, "jppfs_cor_Assets", "CurrentYearInstant_NonConsolidatedMember"
    )
    assert consolidated.value == 100
    assert nonconsolidated.value == 50


def test_read_fact_duplicate_tags_with_same_value_are_fine() -> None:
    root = _parse(
        '<jppfs_cor:Assets contextRef="CurrentYearInstant" unitRef="JPY">100</jppfs_cor:Assets>'
        '<jppfs_cor:Assets contextRef="CurrentYearInstant" unitRef="JPY">100</jppfs_cor:Assets>'
    )
    fact = read_fact(root, "jppfs_cor_Assets", "CurrentYearInstant")
    assert fact.value == 100


def test_read_fact_does_not_round_or_scale_value() -> None:
    root = _parse(
        '<jppfs_cor:Assets contextRef="CurrentYearInstant" unitRef="JPY" '
        'decimals="-6">123456789</jppfs_cor:Assets>'
    )
    fact = read_fact(root, "jppfs_cor_Assets", "CurrentYearInstant")
    assert fact.value == 123456789


def test_resolve_element_tag_handles_company_specific_extension_prefix() -> None:
    root = _parse(
        '<jpcrp030000-asr_E04641-000:CoursePPE contextRef="CurrentYearInstant" unitRef="JPY">'
        "12345</jpcrp030000-asr_E04641-000:CoursePPE>"
    )
    tag = resolve_element_tag(root, "jpcrp030000-asr_E04641-000_CoursePPE")
    assert tag == (
        "{http://disclosure.edinet-fsa.go.jp/jpcrp030000/asr/001/E04641-000}CoursePPE"
    )
    fact = read_fact(root, "jpcrp030000-asr_E04641-000_CoursePPE", "CurrentYearInstant")
    assert fact.value == 12345


def test_resolve_element_tag_returns_none_for_unresolvable_element() -> None:
    root = _parse("")
    assert resolve_element_tag(root, "totally_unknown_prefix_Something") is None
    fact = read_fact(root, "totally_unknown_prefix_Something", "CurrentYearInstant")
    assert fact.value is None


def test_read_text_block_present() -> None:
    root_present = _parse(
        '<jpcrp_cor:MaterialMattersRelatingToGoingConcernEtcBusinessRisksTextBlock '
        'contextRef="CurrentYearInstant">Some risk text.'
        "</jpcrp_cor:MaterialMattersRelatingToGoingConcernEtcBusinessRisksTextBlock>"
    )
    root_absent = _parse("")
    element_id = "jpcrp_cor_MaterialMattersRelatingToGoingConcernEtcBusinessRisksTextBlock"
    assert read_text_block_present(root_present, element_id) is True
    assert read_text_block_present(root_absent, element_id) is False


def test_duration_context_id_for_role() -> None:
    assert duration_context_id_for_role("consolidated") == "CurrentYearDuration"
    assert duration_context_id_for_role("nonconsolidated", "prior") == (
        "Prior1YearDuration_NonConsolidatedMember"
    )
