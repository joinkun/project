"""calculation linkbase の構造的な異常検出ロジック。

``calculation_link.py`` から内部的に使われる。公開 API はそちらの
re-export 経由で ``fsgen.parsing.calculation_link`` として使うこと。
"""

from __future__ import annotations

from dataclasses import dataclass

from fsgen.parsing._calc_types import CalcArc


@dataclass(frozen=True)
class CalcTreeDiagnostics:
    """パース時に検出した構造的な異常・注記事項。例外にはせず、ここに集める。"""

    multi_parent_elements: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]
    non_unit_weight_arcs: tuple[CalcArc, ...]
    prohibited_arcs: tuple[CalcArc, ...]
    extension_elements: tuple[str, ...]


def is_extension_element(element_id: str, standard_prefixes: tuple[str, ...]) -> bool:
    """標準タクソノミの接頭辞で始まらなければ会社独自の拡張要素とみなす。"""
    return not any(
        element_id == prefix or element_id.startswith(f"{prefix}_")
        for prefix in standard_prefixes
    )


def find_cycles(arcs: tuple[CalcArc, ...]) -> tuple[tuple[str, ...], ...]:
    """深さ優先探索で有向閉路を検出する。white/gray/black の3色法。"""
    adjacency: dict[str, list[str]] = {}
    for arc in arcs:
        adjacency.setdefault(arc.parent, []).append(arc.child)

    color: dict[str, int] = {}
    path: list[str] = []
    cycles: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        color[node] = 1
        path.append(node)
        for nxt in adjacency.get(node, []):
            state = color.get(nxt, 0)
            if state == 0:
                visit(nxt)
            elif state == 1:
                start = path.index(nxt)
                cycles.append(tuple(path[start:]))
        path.pop()
        color[node] = 2

    nodes = sorted({a.parent for a in arcs} | {a.child for a in arcs})
    for node in nodes:
        if color.get(node, 0) == 0:
            visit(node)
    return tuple(cycles)


def build_diagnostics(
    arcs: tuple[CalcArc, ...],
    prohibited: tuple[CalcArc, ...],
    standard_prefixes: tuple[str, ...],
) -> CalcTreeDiagnostics:
    parents_by_child: dict[str, set[str]] = {}
    for arc in arcs:
        parents_by_child.setdefault(arc.child, set()).add(arc.parent)
    multi_parent = tuple(sorted(c for c, ps in parents_by_child.items() if len(ps) > 1))

    non_unit_weight = tuple(a for a in arcs if abs(a.weight) != 1.0)

    elements = sorted({a.parent for a in arcs} | {a.child for a in arcs})
    extensions = tuple(e for e in elements if is_extension_element(e, standard_prefixes))

    return CalcTreeDiagnostics(
        multi_parent_elements=multi_parent,
        cycles=find_cycles(arcs),
        non_unit_weight_arcs=non_unit_weight,
        prohibited_arcs=prohibited,
        extension_elements=extensions,
    )
