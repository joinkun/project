"""XBRL インスタンス(*.xbrl)から、指定した勘定科目・コンテキストの値を読む。

書類取得APIで取れる「素の」インスタンスファイル(要素が直接タグとして現れる
形式。インライン XBRL の ix:nonFraction ではない)を対象とする。

値は円単位の整数のまま扱い、丸めない。``decimals`` 属性は報告精度の
表明であって値のスケールではないため無視する(§4.3)。

コンテキストIDの規約(実データで確認済み):
- 連結の当期末: ``CurrentYearInstant``
- 単体の当期末: ``CurrentYearInstant_NonConsolidatedMember``

要素IDから接頭辞・要素名を切り出す際の注意:
標準要素(jppfs_cor/jpcrp_cor/jpdei_cor)はプレフィックスが既知だが、拡張要素
(例: ``jpcrp030000-asr_E04641-000_CoursePPE``)は接頭辞と要素名の境界が
文字列だけでは一意に決まらない(``calculation_link.py`` 参照)。そこで、
文字列を推測するのではなく、**インスタンス文書自体が宣言している名前空間
接頭辞**の中から最長一致するものを探して解決する。文書が実際に使っている
接頭辞の集合は有限かつ厳密にわかるため、この方法なら拡張要素も安全に解決
できる。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from lxml import etree

_XBRLI_NS = "http://www.xbrl.org/2003/instance"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_XBRLI = f"{{{_XBRLI_NS}}}"
_XSI_NIL = f"{{{_XSI_NS}}}nil"

CONSOLIDATED_CONTEXT = "CurrentYearInstant"
NONCONSOLIDATED_CONTEXT = "CurrentYearInstant_NonConsolidatedMember"

_STANDARD_PREFIXES = ("jppfs_cor", "jpcrp_cor", "jpdei_cor")


def context_id_for_role(used_role: str) -> str:
    """"consolidated"/"nonconsolidated" -> 対応する当期末コンテキストID。"""
    if used_role == "consolidated":
        return CONSOLIDATED_CONTEXT
    if used_role == "nonconsolidated":
        return NONCONSOLIDATED_CONTEXT
    raise ValueError(f"unknown role: {used_role!r}")


def split_element_id(
    element_id: str, standard_prefixes: tuple[str, ...] = _STANDARD_PREFIXES
) -> tuple[str, str]:
    """"jppfs_cor_Assets" -> ("jppfs_cor", "Assets")。標準要素のみ対応。"""
    for prefix in standard_prefixes:
        if element_id.startswith(f"{prefix}_"):
            return prefix, element_id[len(prefix) + 1 :]
    raise ValueError(f"element_id {element_id!r} does not start with a known standard prefix")


def resolve_element_tag(root: etree._Element, element_id: str) -> str | None:
    """element_id をこの文書の実際の名前空間宣言に照らして完全修飾タグへ解決する。

    複数の接頭辞が前方一致しうる場合は最長一致(最も具体的なもの)を採用する。
    解決できなければ None(この文書にその要素は存在しない)。
    """
    candidates = [
        prefix
        for prefix in root.nsmap
        if prefix is not None
        and (element_id == prefix or element_id.startswith(f"{prefix}_"))
    ]
    if not candidates:
        return None
    prefix = max(candidates, key=len)
    local_name = element_id[len(prefix) + 1 :] if element_id != prefix else ""
    ns_uri = root.nsmap[prefix]
    return f"{{{ns_uri}}}{local_name}"


def context_instant(root: etree._Element, context_id: str) -> str | None:
    """指定コンテキストの instant 日付(YYYY-MM-DD)。無ければ None。"""
    context = root.find(f"{_XBRLI}context[@id='{context_id}']")
    if context is None:
        return None
    instant_el = context.find(f"{_XBRLI}period/{_XBRLI}instant")
    if instant_el is None or instant_el.text is None:
        return None
    text: str = instant_el.text
    return text.strip()


def verify_context_period(root: etree._Element, context_id: str, expected_instant: str) -> bool:
    """指定コンテキストの instant が期待する日付(periodEnd)と一致するか。"""
    return context_instant(root, context_id) == expected_instant


@dataclass(frozen=True)
class FactValue:
    element_id: str
    context_ref: str
    value: int | None  # None = 欠損(要素が無い、またはxsi:nil、または空値)
    decimals: str | None
    unit_ref: str | None

    @property
    def is_missing(self) -> bool:
        return self.value is None


def read_fact(root: etree._Element, element_id: str, context_id: str) -> FactValue:
    """(element_id, context_id) に対応する事実を読む。無ければ value=None。

    同じ (element_id, context_id) で異なる値を持つ事実が複数見つかった場合は
    ValueError(同一書類内で同じ事実が複数箇所にタグ付けされること自体はある
    が、値は常に一致するはず)。
    """
    tag = resolve_element_tag(root, element_id)
    if tag is None:
        return FactValue(element_id, context_id, None, None, None)

    matches = [el for el in root.iter(tag) if el.get("contextRef") == context_id]
    if not matches:
        return FactValue(element_id, context_id, None, None, None)

    values: set[str] = set()
    decimals: str | None = None
    unit_ref: str | None = None
    for el in matches:
        if el.get(_XSI_NIL) == "true":
            continue
        text = (el.text or "").strip()
        if not text:
            continue
        values.add(text)
        decimals = el.get("decimals")
        unit_ref = el.get("unitRef")

    if not values:
        return FactValue(element_id, context_id, None, decimals, unit_ref)
    if len(values) > 1:
        raise ValueError(f"inconsistent values for {element_id} @ {context_id}: {sorted(values)}")

    raw = values.pop()
    try:
        value = int(Decimal(raw))
    except InvalidOperation as exc:
        raise ValueError(f"non-numeric value for {element_id} @ {context_id}: {raw!r}") from exc
    return FactValue(element_id, context_id, value, decimals, unit_ref)


def read_text_block_present(root: etree._Element, element_id: str) -> bool:
    """テキストブロック要素(内容がある事実)がインスタンス中に存在するか(コンテキスト不問)。"""
    tag = resolve_element_tag(root, element_id)
    if tag is None:
        return False
    for el in root.iter(tag):
        if el.get(_XSI_NIL) == "true":
            continue
        if (el.text or "").strip():
            return True
    return False
