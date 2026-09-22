#!/usr/bin/env python
"""全社の _cal.xml を読み、貸借対照表の勘定科目の出現頻度と階層構造を集計する。

ZIP は展開せず zipfile でメモリ上から直接読む。canonical 勘定科目リストを
決めるための材料を出すのが目的で、リストの決定自体はここでは行わない。

使い方:
    uv run python scripts/survey_accounts.py
    uv run python scripts/survey_accounts.py --n-jobs 4
"""

from __future__ import annotations

import argparse
import csv
import functools
import logging
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from fsgen.parsing.calculation_link import (
    CalcTree,
    is_extension_element,
    parse_calculation_link_bytes,
    select_tree,
)
from fsgen.parsing.zip_reader import read_cal_xml

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[1] / "conf" / "config.yaml"
_DEFAULT_DOCUMENT_LIST_CSV = Path("data/raw/index/document_list.csv")
_DEFAULT_DOCUMENTS_DIR = Path("data/raw/documents")
_DEFAULT_OUTPUT_DIR = Path("data/interim/survey")


def _load_calc_link_config() -> tuple[dict[str, str], tuple[str, ...]]:
    with _CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)["calculation_link"]
    return config["balance_sheet_roles"], tuple(config["standard_prefixes"])


def _element_prefix(element_id: str, standard_prefixes: tuple[str, ...]) -> str:
    for prefix in standard_prefixes:
        if element_id == prefix or element_id.startswith(f"{prefix}_"):
            return prefix
    return "extension"


def _count_leaves_at_or_below_depth(tree: CalcTree, max_depth: int) -> int:
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


@dataclass(frozen=True)
class ElementOccurrence:
    element_id: str
    depth: int
    is_leaf: bool


@dataclass(frozen=True)
class ArcOccurrence:
    parent: str
    child: str
    weight: float


@dataclass(frozen=True)
class CompanyResult:
    doc_id: str
    edinet_code: str
    parse_status: str  # "ok" | "failed"
    error: str | None
    has_consolidated: bool
    has_nonconsolidated: bool
    used_role: str | None
    n_elements: int | None
    n_arcs: int | None
    n_roots: int | None
    roots: tuple[str, ...]
    max_depth: int | None
    n_leaves: int | None
    n_extension_elements: int | None
    has_negative_weight: bool | None
    n_leaves_depth2: int | None
    n_leaves_depth3: int | None
    negative_weight_arcs: tuple[tuple[str, str], ...]
    element_occurrences: tuple[ElementOccurrence, ...]
    arc_occurrences: tuple[ArcOccurrence, ...]


def _failed_result(doc_id: str, edinet_code: str, error: Exception) -> CompanyResult:
    return CompanyResult(
        doc_id=doc_id,
        edinet_code=edinet_code,
        parse_status="failed",
        error=f"{type(error).__name__}: {error}",
        has_consolidated=False,
        has_nonconsolidated=False,
        used_role=None,
        n_elements=None,
        n_arcs=None,
        n_roots=None,
        roots=(),
        max_depth=None,
        n_leaves=None,
        n_extension_elements=None,
        has_negative_weight=None,
        n_leaves_depth2=None,
        n_leaves_depth3=None,
        negative_weight_arcs=(),
        element_occurrences=(),
        arc_occurrences=(),
    )


def process_one(
    doc_id: str,
    edinet_code: str,
    documents_dir: Path,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...],
) -> CompanyResult:
    zip_path = documents_dir / f"{doc_id}.zip"
    try:
        cal_bytes = read_cal_xml(zip_path)
        trees = parse_calculation_link_bytes(
            cal_bytes, role_map, standard_prefixes, source_label=doc_id
        )
        tree, used_role = select_tree(trees)
    except Exception as exc:  # noqa: BLE001 - 1社の失敗で全体を止めないため意図的に広く捕捉する
        return _failed_result(doc_id, edinet_code, exc)

    leaves = set(tree.leaves())
    element_occurrences = tuple(
        ElementOccurrence(e, tree.depth_of(e), e in leaves) for e in tree.elements
    )
    arc_occurrences = tuple(ArcOccurrence(a.parent, a.child, a.weight) for a in tree.arcs)
    negative_weight_arcs = tuple(sorted({(a.parent, a.child) for a in tree.arcs if a.weight < 0}))
    histogram = tree.depth_histogram()
    max_depth = max(histogram) if histogram else 0

    return CompanyResult(
        doc_id=doc_id,
        edinet_code=edinet_code,
        parse_status="ok",
        error=None,
        has_consolidated="consolidated" in trees,
        has_nonconsolidated="nonconsolidated" in trees,
        used_role=used_role,
        n_elements=len(tree.elements),
        n_arcs=len(tree.arcs),
        n_roots=len(tree.roots),
        roots=tree.roots,
        max_depth=max_depth,
        n_leaves=len(leaves),
        n_extension_elements=len(tree.diagnostics.extension_elements),
        has_negative_weight=len(negative_weight_arcs) > 0,
        n_leaves_depth2=_count_leaves_at_or_below_depth(tree, 2),
        n_leaves_depth3=_count_leaves_at_or_below_depth(tree, 3),
        negative_weight_arcs=negative_weight_arcs,
        element_occurrences=element_occurrences,
        arc_occurrences=arc_occurrences,
    )


class Aggregator:
    """1社ずつ結果を取り込み、生データ(要素・アークの列挙)は取り込んだ直後に捨てる。"""

    def __init__(self) -> None:
        self.n_ok = 0
        self.n_failed = 0
        self.company_rows: list[dict[str, object]] = []
        self.failed: list[tuple[str, str]] = []

        self.element_n_companies: Counter[str] = Counter()
        self.element_depths: dict[str, list[int]] = defaultdict(list)
        self.element_n_leaf: Counter[str] = Counter()
        self.element_n_internal: Counter[str] = Counter()

        self.arc_n_companies: Counter[tuple[str, str, float]] = Counter()

        self.has_consolidated_count = 0
        self.has_nonconsolidated_only_count = 0
        self.n_extension_per_company: list[int] = []
        self.roots_counter: Counter[tuple[str, ...]] = Counter()
        self.max_depth_counter: Counter[int] = Counter()
        self.n_leaves_list: list[int] = []
        self.n_leaves_depth2_list: list[int] = []
        self.n_leaves_depth3_list: list[int] = []
        self.negative_weight_companies = 0
        self.negative_weight_arc_set: set[tuple[str, str]] = set()

    def add(self, result: CompanyResult) -> None:
        self.company_rows.append(
            {
                "docID": result.doc_id,
                "edinetCode": result.edinet_code,
                "has_consolidated": result.has_consolidated,
                "has_nonconsolidated": result.has_nonconsolidated,
                "used_role": result.used_role,
                "n_elements": result.n_elements,
                "n_arcs": result.n_arcs,
                "n_roots": result.n_roots,
                "roots": ";".join(result.roots),
                "max_depth": result.max_depth,
                "n_leaves": result.n_leaves,
                "n_extension_elements": result.n_extension_elements,
                "has_negative_weight": result.has_negative_weight,
                "parse_status": result.parse_status,
                "error": result.error,
            }
        )

        if result.parse_status != "ok":
            self.n_failed += 1
            self.failed.append((result.doc_id, result.error or ""))
            return

        self.n_ok += 1
        if result.has_consolidated:
            self.has_consolidated_count += 1
        elif result.has_nonconsolidated:
            self.has_nonconsolidated_only_count += 1

        for occ in result.element_occurrences:
            self.element_n_companies[occ.element_id] += 1
            self.element_depths[occ.element_id].append(occ.depth)
            if occ.is_leaf:
                self.element_n_leaf[occ.element_id] += 1
            else:
                self.element_n_internal[occ.element_id] += 1

        for arc in result.arc_occurrences:
            self.arc_n_companies[(arc.parent, arc.child, arc.weight)] += 1

        assert result.n_extension_elements is not None
        self.n_extension_per_company.append(result.n_extension_elements)
        self.roots_counter[result.roots] += 1
        assert result.max_depth is not None
        self.max_depth_counter[result.max_depth] += 1
        assert result.n_leaves is not None
        self.n_leaves_list.append(result.n_leaves)
        assert result.n_leaves_depth2 is not None
        self.n_leaves_depth2_list.append(result.n_leaves_depth2)
        assert result.n_leaves_depth3 is not None
        self.n_leaves_depth3_list.append(result.n_leaves_depth3)
        if result.has_negative_weight:
            self.negative_weight_companies += 1
            self.negative_weight_arc_set.update(result.negative_weight_arcs)


def _iter_targets(document_list_csv: Path) -> list[tuple[str, str]]:
    with document_list_csv.open(newline="", encoding="utf-8") as f:
        return [(row["docID"], row["edinetCode"] or "") for row in csv.DictReader(f)]


def run_survey(
    document_list_csv: Path,
    documents_dir: Path,
    n_jobs: int,
    show_progress: bool = True,
) -> Aggregator:
    targets = _iter_targets(document_list_csv)
    role_map, standard_prefixes = _load_calc_link_config()
    aggregator = Aggregator()

    if n_jobs <= 1:
        for doc_id, edinet_code in tqdm(targets, desc="survey", disable=not show_progress):
            aggregator.add(
                process_one(doc_id, edinet_code, documents_dir, role_map, standard_prefixes)
            )
        return aggregator

    worker = functools.partial(
        process_one,
        documents_dir=documents_dir,
        role_map=role_map,
        standard_prefixes=standard_prefixes,
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(worker, doc_id, edinet_code) for doc_id, edinet_code in targets]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="survey", disable=not show_progress
        ):
            aggregator.add(future.result())
    return aggregator


def write_element_frequency_csv(
    aggregator: Aggregator, standard_prefixes: tuple[str, ...], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "element_id",
                "prefix",
                "is_extension",
                "n_companies",
                "ratio",
                "depth_mode",
                "depth_min",
                "depth_max",
                "n_as_leaf",
                "n_as_internal",
            ]
        )
        items = sorted(
            aggregator.element_n_companies.items(), key=lambda kv: (-kv[1], kv[0])
        )
        for element_id, n_companies in items:
            depths = aggregator.element_depths[element_id]
            ratio = n_companies / aggregator.n_ok if aggregator.n_ok else 0.0
            writer.writerow(
                [
                    element_id,
                    _element_prefix(element_id, standard_prefixes),
                    is_extension_element(element_id, standard_prefixes),
                    n_companies,
                    f"{ratio:.6f}",
                    statistics.mode(depths),
                    min(depths),
                    max(depths),
                    aggregator.element_n_leaf.get(element_id, 0),
                    aggregator.element_n_internal.get(element_id, 0),
                ]
            )


def write_arc_frequency_csv(aggregator: Aggregator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parent", "child", "weight", "n_companies"])
        items = sorted(aggregator.arc_n_companies.items(), key=lambda kv: (-kv[1], kv[0]))
        for (parent, child, weight), n_companies in items:
            writer.writerow([parent, child, weight, n_companies])


def write_company_summary_csv(aggregator: Aggregator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "docID",
        "edinetCode",
        "has_consolidated",
        "has_nonconsolidated",
        "used_role",
        "n_elements",
        "n_arcs",
        "n_roots",
        "roots",
        "max_depth",
        "n_leaves",
        "n_extension_elements",
        "has_negative_weight",
        "parse_status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregator.company_rows)


def _ratio_bucket(ratio: float) -> str:
    if ratio == 1.0:
        return "100%    "
    if ratio >= 0.9:
        return "90-100% "
    if ratio >= 0.5:
        return "50-90%  "
    if ratio >= 0.1:
        return "10-50%  "
    if ratio >= 0.01:
        return "1-10%   "
    return "1%未満   "


def print_summary(aggregator: Aggregator, standard_prefixes: tuple[str, ...]) -> None:
    total = aggregator.n_ok + aggregator.n_failed
    print(f"処理: {total}社(成功 {aggregator.n_ok} / 失敗 {aggregator.n_failed})\n")
    print(
        f"連結あり: {aggregator.has_consolidated_count}社 / "
        f"単体のみ: {aggregator.has_nonconsolidated_only_count}社\n"
    )

    print("要素の出現率分布(標準要素のみ):")
    bucket_counts: Counter[str] = Counter()
    for element_id, n_companies in aggregator.element_n_companies.items():
        if is_extension_element(element_id, standard_prefixes):
            continue
        ratio = n_companies / aggregator.n_ok if aggregator.n_ok else 0.0
        bucket_counts[_ratio_bucket(ratio)] += 1
    for bucket in ("100%    ", "90-100% ", "50-90%  ", "10-50%  ", "1-10%   ", "1%未満   "):
        print(f"  {bucket}: {bucket_counts.get(bucket, 0)} 要素")

    extension_elements = [
        e for e in aggregator.element_n_companies if is_extension_element(e, standard_prefixes)
    ]
    total_extension_occurrences = sum(
        aggregator.element_n_companies[e] for e in extension_elements
    )
    companies_with_extension = sum(1 for n in aggregator.n_extension_per_company if n > 0)
    pct = 100 * companies_with_extension / aggregator.n_ok if aggregator.n_ok else 0.0
    print(
        f"\n拡張要素: 延べ {total_extension_occurrences} 個 / ユニーク {len(extension_elements)} 個"
    )
    print(f"  拡張要素を持つ会社: {companies_with_extension} 社 ({pct:.0f}%)")
    nonzero = [n for n in aggregator.n_extension_per_company if n > 0]
    if nonzero:
        print(
            f"  1社あたり拡張要素数: 中央値 {statistics.median(nonzero):.0f}, 最大 {max(nonzero)}"
        )
    else:
        print("  1社あたり拡張要素数: (拡張要素を持つ会社なし)")

    print("\n根の構成:")
    standard_roots = ("jppfs_cor_Assets", "jppfs_cor_LiabilitiesAndNetAssets")
    standard_count = 0
    other: Counter[tuple[str, ...]] = Counter()
    for roots, count in aggregator.roots_counter.items():
        if set(roots) == set(standard_roots):
            standard_count += count
        else:
            other[roots] += count
    print(f"  {{Assets, LiabilitiesAndNetAssets}} : {standard_count}社")
    if other:
        print(f"  その他                             : {sum(other.values())}社")
        for roots, count in sorted(other.items(), key=lambda kv: -kv[1]):
            print(f"    {roots}: {count}社")

    print(f"\n最大深さの分布: {dict(sorted(aggregator.max_depth_counter.items()))}")

    print(f"\nBS内に weight=-1 のアークを持つ会社: {aggregator.negative_weight_companies}社")
    if aggregator.negative_weight_arc_set:
        print("  → 該当する (parent, child):")
        for parent, child in sorted(aggregator.negative_weight_arc_set):
            print(f"    {parent} -> {child}")

    if aggregator.n_leaves_list:
        q1, _, q3 = statistics.quantiles(aggregator.n_leaves_list, n=4)
        print(
            f"\n葉ノード数の分布: 中央値 {statistics.median(aggregator.n_leaves_list):.0f}, "
            f"四分位 [{q1:.0f}, {q3:.0f}], "
            f"範囲 [{min(aggregator.n_leaves_list)}, {max(aggregator.n_leaves_list)}]"
        )
    for label, values in (
        ("深さ<=2", aggregator.n_leaves_depth2_list),
        ("深さ<=3", aggregator.n_leaves_depth3_list),
    ):
        if values:
            print(
                f"{label} で切った場合の葉ノード数: 中央値 {statistics.median(values):.0f}, "
                f"範囲 [{min(values)}, {max(values)}]"
            )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-list-csv", type=Path, default=_DEFAULT_DOCUMENT_LIST_CSV)
    parser.add_argument("--documents-dir", type=Path, default=_DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="ProcessPoolExecutorの並列数(デフォルト1=逐次)"
    )
    args = parser.parse_args()

    _, standard_prefixes = _load_calc_link_config()
    aggregator = run_survey(args.document_list_csv, args.documents_dir, args.n_jobs)

    write_element_frequency_csv(
        aggregator, standard_prefixes, args.output_dir / "element_frequency.csv"
    )
    write_arc_frequency_csv(aggregator, args.output_dir / "arc_frequency.csv")
    write_company_summary_csv(aggregator, args.output_dir / "company_summary.csv")

    print_summary(aggregator, standard_prefixes)


if __name__ == "__main__":
    main()
