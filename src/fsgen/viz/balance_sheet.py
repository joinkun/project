"""貸借対照表(BS)を表形式で出力する(ステージ6 §4)。

生成された経路 x(t)(円単位、基礎科目のみの状態ベクトル)を、階層構造
(小計と内訳の入れ子)を保った BS として表示する。小計は常に配下の
基礎科目の和として再計算する(表示上も検算できる形にする)。科目の
日本語ラベルは要素IDから機械的に生成せず、`conf/accounts/labels_ja.yaml`
の手書き対応表を使う。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from fsgen.accounting.constraints import (
    ASSETS_ID,
    LIABILITIES_ID,
    NET_ASSETS_ID,
    SUBTOTAL_CHILDREN,
    CanonicalSpec,
)

_YEN_PER_MILLION = 1_000_000.0
_LIABILITIES_AND_NET_ASSETS_ID = "jppfs_cor_LiabilitiesAndNetAssets"

LineKind = Literal["section", "group", "item", "subtotal", "total", "diff"]


@dataclass(frozen=True)
class LabelMap:
    accounts: dict[str, str]
    sections: dict[str, str]


def load_labels(path: Path) -> LabelMap:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return LabelMap(accounts=dict(doc["accounts"]), sections=dict(doc["sections"]))


def _label(labels: LabelMap, key: str, as_total: bool = False) -> str:
    base = labels.accounts.get(key, key)
    return f"{base}合計" if as_total else base


@dataclass(frozen=True)
class BSLine:
    label: str
    indent: int
    kind: LineKind
    values: tuple[float, ...] | None  # 円。section見出し行は None。


def build_bs_lines(
    spec: CanonicalSpec, names: tuple[str, ...], x_by_t: list[np.ndarray], labels: LabelMap
) -> list[BSLine]:
    """spec.basis の parent_subtotal 構造から階層的なBS行リストを組み立てる。

    小計・合計は全て「配下の和」として計算し直す(reported な小計値は
    使わない。x(t) は基礎科目のみの状態ベクトルであり、そもそも小計値を
    持たないため)。
    """
    index = {n: i for i, n in enumerate(names)}
    n_t = len(x_by_t)

    def value_of(name: str) -> tuple[float, ...]:
        i = index[name]
        return tuple(float(x[i]) for x in x_by_t)

    def add(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(x + y for x, y in zip(a, b, strict=True))

    lines: list[BSLine] = []
    zero = (0.0,) * n_t
    side_tops = (
        ("assets", ASSETS_ID),
        ("liabilities", LIABILITIES_ID),
        ("net_assets", NET_ASSETS_ID),
    )
    totals: dict[str, tuple[float, ...]] = {}

    for side, top_id in side_tops:
        lines.append(BSLine(f"【{labels.sections[side]}】", 0, "section", None))
        top_sum = zero
        for child_id in SUBTOTAL_CHILDREN.get(top_id, ()):
            lines.append(BSLine(_label(labels, child_id), 1, "group", None))
            child_sum = zero
            # ステージ39: 開いた小計(有形固定資産など)は、その配下を 1 段深く表示して小計を付ける
            for sub in (s for s in spec.subtotals if s.parent == child_id):
                lines.append(BSLine(_label(labels, sub.element_id), 2, "group", None))
                sub_sum = zero
                for b in spec.basis:
                    if b.parent_subtotal != sub.element_id:
                        continue
                    vals = value_of(b.name)
                    lines.append(BSLine(_label(labels, b.name), 3, "item", vals))
                    sub_sum = add(sub_sum, vals)
                lines.append(
                    BSLine(_label(labels, sub.element_id, as_total=True), 2, "subtotal", sub_sum)
                )
                child_sum = add(child_sum, sub_sum)
            for b in spec.basis:
                if b.parent_subtotal != child_id:
                    continue
                vals = value_of(b.name)
                lines.append(BSLine(_label(labels, b.name), 2, "item", vals))
                child_sum = add(child_sum, vals)
            lines.append(BSLine(_label(labels, child_id, as_total=True), 1, "subtotal", child_sum))
            top_sum = add(top_sum, child_sum)
        for b in spec.basis:
            if b.parent_subtotal != top_id:
                continue
            vals = value_of(b.name)
            lines.append(BSLine(_label(labels, b.name), 1, "item", vals))
            top_sum = add(top_sum, vals)
        lines.append(BSLine(_label(labels, top_id, as_total=True), 0, "total", top_sum))
        totals[top_id] = top_sum

    liab_and_net = add(totals[LIABILITIES_ID], totals[NET_ASSETS_ID])
    liab_and_net_label = _label(labels, _LIABILITIES_AND_NET_ASSETS_ID, as_total=True)
    lines.append(BSLine(liab_and_net_label, 0, "total", liab_and_net))
    diff = tuple(a - b for a, b in zip(totals[ASSETS_ID], liab_and_net, strict=True))
    lines.append(BSLine("貸借差額", 0, "diff", diff))
    return lines


def _t_column_label(t: float) -> str:
    if t == 0.0:
        return "現状(t=0)"
    if t == 1.0:
        return "目標(t=1.0)"
    return f"t={t:g}"


def _fmt_yen_millions(v: float) -> str:
    millions = v / _YEN_PER_MILLION
    text = f"{millions:,.0f}"
    return "0" if text == "-0" else text  # 丸めて0になる微小な負値は「-0」と出さない


def _fmt_change(v0: float, v1: float) -> str:
    change = (v1 - v0) / _YEN_PER_MILLION
    sign = "+" if change > 0 else ""
    text = f"{sign}{change:,.0f}"
    return "0" if text == "-0" else text


def render_text(
    lines: list[BSLine], ts: tuple[float, ...], company_name: str, doc_id: str
) -> str:
    """標準出力向けの固定幅テキスト表。"""
    col_labels = [_t_column_label(t) for t in ts] + ["変化"]
    name_width = 26
    col_width = 12

    out = [f"{company_name}(docID: {doc_id})  単位: 百万円", ""]
    header = " " * name_width + "".join(f"{c:>{col_width}}" for c in col_labels)
    out.append(header)
    for line in lines:
        indent_str = "  " * line.indent
        label = f"{indent_str}{line.label}"
        if line.values is None:
            out.append(label)
            continue
        cells = [_fmt_yen_millions(v) for v in line.values]
        cells.append(_fmt_change(line.values[0], line.values[-1]))
        out.append(f"{label:<{name_width}}" + "".join(f"{c:>{col_width}}" for c in cells))
    return "\n".join(out)


def render_markdown(
    lines: list[BSLine], ts: tuple[float, ...], company_name: str, doc_id: str
) -> str:
    """Markdownファイル向けの表(パイプ区切り)。"""
    col_labels = [_t_column_label(t) for t in ts] + ["変化"]
    out = [f"# {company_name}(docID: {doc_id})", "", "単位: 百万円", ""]
    out.append("| 科目 | " + " | ".join(col_labels) + " |")
    out.append("|---|" + "---|" * len(col_labels))
    for line in lines:
        indent_str = "&nbsp;&nbsp;" * line.indent
        label = f"{indent_str}{line.label}"
        if line.values is None:
            out.append(f"| **{line.label}** |" + " |" * len(col_labels))
            continue
        cells = [_fmt_yen_millions(v) for v in line.values]
        cells.append(_fmt_change(line.values[0], line.values[-1]))
        weight = "**" if line.kind in ("subtotal", "total", "diff") else ""
        out.append(f"| {weight}{label}{weight} | " + " | ".join(cells) + " |")
    return "\n".join(out)
