#!/usr/bin/env python
"""calculation linkbase の調査スクリプト。

fixtures 配下の全企業ディレクトリを走査し、貸借対照表の計算階層構造
(要素数・アーク数・根・深さ分布など)を標準出力へ表示する。階層レベル
(D の次元)を決めるための判断材料を出すのが目的で、集約や制約行列の
構築は行わない(ステージ3)。

使い方:
    uv run python scripts/inspect_cal.py tests/fixtures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from fsgen.parsing.calculation_link import (
    CalcTree,
    find_cal_file,
    parse_calculation_link,
    select_tree,
)

_CONFIG_PATH = Path(__file__).parents[1] / "conf" / "config.yaml"


def _load_calc_link_config() -> tuple[dict[str, str], tuple[str, ...]]:
    with _CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)["calculation_link"]
    return config["balance_sheet_roles"], tuple(config["standard_prefixes"])


def _count_leaves_cut_at_depth(tree: CalcTree, max_depth: int) -> int:
    """「深さ d 以下の要素のうち、深さ d 以下に子を持たないもの」の数。"""
    count = 0
    for element in tree.elements:
        if tree.depth_of(element) > max_depth:
            continue
        children_within = [
            arc for arc in tree.children_of(element) if tree.depth_of(arc.child) <= max_depth
        ]
        if not children_within:
            count += 1
    return count


def _format_list(items: tuple[object, ...]) -> str:
    return ", ".join(str(item) for item in items) if items else "(なし)"


def _print_tree_summary(kind: str, tree: CalcTree) -> None:
    print(
        f"  {kind:<15}: elements={len(tree.elements)} "
        f"arcs={len(tree.arcs)} roots={list(tree.roots)}"
    )


def _print_selected_tree_detail(kind: str, tree: CalcTree) -> None:
    print(f"  [{kind}] 使用")
    histogram = tree.depth_histogram()
    max_depth = max(histogram)
    print(f"    最大深さ: {max_depth}")
    print(f"    深さ別要素数: {dict(sorted(histogram.items()))}")
    print(f"    葉の数: {len(tree.leaves())}")
    print(f"    weight=-1 のアーク: {_format_list(tuple(a for a in tree.arcs if a.weight < 0))}")
    print(f"    拡張要素: {_format_list(tree.diagnostics.extension_elements)}")
    print(f"    複数親を持つ要素: {_format_list(tree.diagnostics.multi_parent_elements)}")
    if tree.diagnostics.cycles:
        print(f"    循環: {tree.diagnostics.cycles}")
    if tree.diagnostics.prohibited_arcs:
        print(f"    除外された use=prohibited アーク数: {len(tree.diagnostics.prohibited_arcs)}")
    for threshold in (2, 3):
        if threshold > max_depth:
            continue
        cut_leaves = _count_leaves_cut_at_depth(tree, threshold)
        print(f"    深さ<={threshold} で切った場合の葉ノード数: {cut_leaves}")


def inspect_one(
    doc_dir: Path, role_map: dict[str, str], standard_prefixes: tuple[str, ...]
) -> None:
    print(f"=== {doc_dir.name} ===")
    try:
        cal_path = find_cal_file(doc_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  スキップ: {exc}")
        return

    trees = parse_calculation_link(cal_path, role_map, standard_prefixes)
    if not trees:
        print("  貸借対照表の role が見つかりませんでした")
        return

    for kind, tree in trees.items():
        _print_tree_summary(kind, tree)

    selected_tree, selected_kind = select_tree(trees)
    _print_selected_tree_detail(selected_kind, selected_tree)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixtures_root", type=Path, help="fixture ディレクトリ群の親 (例: tests/fixtures/)"
    )
    args = parser.parse_args()

    role_map, standard_prefixes = _load_calc_link_config()
    doc_dirs = sorted(p for p in args.fixtures_root.iterdir() if p.is_dir())
    for doc_dir in doc_dirs:
        inspect_one(doc_dir, role_map, standard_prefixes)


if __name__ == "__main__":
    main()
