"""EDINET XBRL の calculation linkbase (``_cal.xml``) を読み、貸借対照表の
集計構造(親→子の有向グラフ)を取り出す。

会計恒等式の制約行列の構築や階層集約はここでは行わない(ステージ3)。
このモジュールは「読むだけ」であり、リンクベースに書かれている内容を
忠実にデータ構造へ写すことだけを責務とする。

構造的な異常検出(循環・複数親・weight 異常・拡張要素)は
``_calc_diagnostics`` に分離してある。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from fsgen.parsing._calc_diagnostics import CalcTreeDiagnostics, build_diagnostics
from fsgen.parsing._calc_diagnostics import is_extension_element as is_extension_element
from fsgen.parsing._calc_types import CalcArc as CalcArc

log = logging.getLogger(__name__)

__all__ = [
    "CalcArc",
    "CalcTree",
    "CalcTreeDiagnostics",
    "find_cal_file",
    "is_extension_element",
    "parse_calculation_link",
    "parse_calculation_link_bytes",
    "select_tree",
]

_LINK_NS = "http://www.xbrl.org/2003/linkbase"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_LINK = f"{{{_LINK_NS}}}"
_XLINK = f"{{{_XLINK_NS}}}"


@dataclass(frozen=True)
class CalcTree:
    """1つの role(連結 or 単体)分の calculation linkbase を表す有向グラフ。"""

    role: str
    role_uri: str
    arcs: tuple[CalcArc, ...]
    elements: tuple[str, ...]
    roots: tuple[str, ...]
    diagnostics: CalcTreeDiagnostics = field(
        default_factory=lambda: CalcTreeDiagnostics((), (), (), (), ())
    )

    def children_of(self, element: str) -> tuple[CalcArc, ...]:
        return tuple(a for a in self.arcs if a.parent == element)

    def parents_of(self, element: str) -> tuple[CalcArc, ...]:
        return tuple(a for a in self.arcs if a.child == element)

    def depth_of(self, element: str) -> int:
        """根からの最短距離。根自身は 0。到達不能なら ValueError。"""
        if element in self.roots:
            return 0
        depths: dict[str, int] = dict.fromkeys(self.roots, 0)
        queue: deque[str] = deque(self.roots)
        while queue:
            current = queue.popleft()
            next_depth = depths[current] + 1
            for arc in self.children_of(current):
                if arc.child not in depths:
                    depths[arc.child] = next_depth
                    queue.append(arc.child)
        if element not in depths:
            raise ValueError(f"element {element!r} is unreachable from any root")
        return depths[element]

    def leaves(self) -> tuple[str, ...]:
        parents = {a.parent for a in self.arcs}
        return tuple(sorted(e for e in self.elements if e not in parents))

    def depth_histogram(self) -> dict[int, int]:
        histogram: dict[int, int] = {}
        for element in self.elements:
            depth = self.depth_of(element)
            histogram[depth] = histogram.get(depth, 0) + 1
        return histogram


def find_cal_file(doc_dir: Path) -> Path:
    """``<doc_dir>/XBRL/PublicDoc/*_cal.xml`` を探す。"""
    matches = sorted((doc_dir / "XBRL" / "PublicDoc").glob("*_cal.xml"))
    if not matches:
        raise FileNotFoundError(f"no *_cal.xml found under {doc_dir / 'XBRL' / 'PublicDoc'}")
    if len(matches) > 1:
        raise ValueError(f"expected exactly one *_cal.xml, found {len(matches)}: {matches}")
    return matches[0]


def _role_segment(role_uri: str) -> str:
    return role_uri.rsplit("/", 1)[-1]


class _Accumulator:
    """1つの出力キー(連結/単体)に対応する arc の集積バッファ。"""

    def __init__(self, role_uri: str) -> None:
        self.role_uri = role_uri
        self.arcs: list[CalcArc] = []
        self.prohibited: list[CalcArc] = []


def _parse_weight(raw: str | None) -> float:
    if raw is None:
        raise ValueError("calculationArc is missing a weight attribute")
    return float(raw)


def _parse_order(raw: str | None) -> float | None:
    return float(raw) if raw is not None else None


def _element_id_from_href(href: str) -> str:
    return href.rsplit("#", 1)[-1] if "#" in href else href


def _collect_loc_map(link_el: etree._Element) -> dict[str, str]:
    """xlink:label -> 要素ID(href の # 以降)。"""
    loc_map: dict[str, str] = {}
    for loc in link_el.findall(f"{_LINK}loc"):
        href = loc.get(f"{_XLINK}href")
        label = loc.get(f"{_XLINK}label")
        if href is None or label is None:
            continue
        loc_map[label] = _element_id_from_href(href)
    return loc_map


def parse_calculation_link(
    cal_path: Path,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...] = ("jppfs_cor", "jpcrp_cor", "jpdei_cor"),
) -> dict[str, CalcTree]:
    """``_cal.xml`` を読み、role_map に載っている role だけを CalcTree にして返す。

    role_map: {"rol_ConsolidatedBalanceSheet": "consolidated", ...}
    該当する role が1つも無い場合、そのキーは返り値に含まれない。
    同じ role の calculationLink が複数回現れる場合はマージする。
    """
    tree = etree.parse(str(cal_path))
    return _parse_calculation_link_root(tree.getroot(), str(cal_path), role_map, standard_prefixes)


def parse_calculation_link_bytes(
    cal_bytes: bytes,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...] = ("jppfs_cor", "jpcrp_cor", "jpdei_cor"),
    source_label: str = "<bytes>",
) -> dict[str, CalcTree]:
    """``_cal.xml`` の中身(bytes)を直接パースする版。ZIPから展開せず読む用途。

    ディスク上のファイルを経由しない以外は ``parse_calculation_link`` と同じ。
    ``source_label`` はログメッセージにのみ使う(どのZIP/企業由来かの手がかり)。
    """
    root = etree.fromstring(cal_bytes)
    return _parse_calculation_link_root(root, source_label, role_map, standard_prefixes)


def _parse_calculation_link_root(
    root: etree._Element,
    source_label: str,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...],
) -> dict[str, CalcTree]:
    accumulators: dict[str, _Accumulator] = {}

    for link_el in root.findall(f"{_LINK}calculationLink"):
        role_uri = link_el.get(f"{_XLINK}role")
        if role_uri is None:
            continue
        output_key = role_map.get(_role_segment(role_uri))
        if output_key is None:
            continue

        loc_map = _collect_loc_map(link_el)
        accumulator = accumulators.setdefault(output_key, _Accumulator(role_uri))

        for arc_el in link_el.findall(f"{_LINK}calculationArc"):
            from_label = arc_el.get(f"{_XLINK}from")
            to_label = arc_el.get(f"{_XLINK}to")
            if from_label is None or to_label is None:
                continue
            parent = loc_map.get(from_label)
            child = loc_map.get(to_label)
            if parent is None or child is None:
                log.warning(
                    "unresolved calculationArc label in %s: from=%r to=%r",
                    source_label,
                    from_label,
                    to_label,
                )
                continue

            calc_arc = CalcArc(
                parent=parent,
                child=child,
                weight=_parse_weight(arc_el.get("weight")),
                order=_parse_order(arc_el.get("order")),
            )

            if arc_el.get("use") == "prohibited":
                log.warning(
                    "prohibited calculationArc excluded in %s: %s -> %s",
                    source_label,
                    parent,
                    child,
                )
                accumulator.prohibited.append(calc_arc)
                continue

            accumulator.arcs.append(calc_arc)

    trees: dict[str, CalcTree] = {}
    for output_key, accumulator in accumulators.items():
        arcs = tuple(accumulator.arcs)
        elements = tuple(sorted({a.parent for a in arcs} | {a.child for a in arcs}))
        parent_ids = {a.parent for a in arcs}
        child_ids = {a.child for a in arcs}
        roots = tuple(sorted(parent_ids - child_ids))

        diagnostics = build_diagnostics(arcs, tuple(accumulator.prohibited), standard_prefixes)
        if diagnostics.multi_parent_elements:
            log.warning(
                "%s: elements with multiple parents (DAG, not a tree): %s",
                output_key,
                diagnostics.multi_parent_elements,
            )
        if diagnostics.cycles:
            log.warning("%s: cycles detected: %s", output_key, diagnostics.cycles)
        if diagnostics.non_unit_weight_arcs:
            log.warning(
                "%s: arcs with weight other than +-1: %s",
                output_key,
                diagnostics.non_unit_weight_arcs,
            )

        trees[output_key] = CalcTree(
            role=output_key,
            role_uri=accumulator.role_uri,
            arcs=arcs,
            elements=elements,
            roots=roots,
            diagnostics=diagnostics,
        )

    return trees


def select_tree(
    trees: dict[str, CalcTree], prefer: str = "consolidated"
) -> tuple[CalcTree, str]:
    """prefer を優先し、無ければ他の種別へフォールバックする。

    戻り値の2要素目は実際に選ばれた種別("consolidated" / "nonconsolidated" 等)。
    """
    if not trees:
        raise ValueError("no calculation trees available to select from")
    if prefer in trees:
        return trees[prefer], prefer
    fallback_key = next(iter(trees))
    return trees[fallback_key], fallback_key
