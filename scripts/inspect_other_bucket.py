#!/usr/bin/env python
"""__OTHER__ 残余バケツの中身を診断する(ステージ4.5 A)。

canonical で集約すると __OTHER__<subtotal> に丸められてしまう葉要素を、
集約する前の段階で記録し、以下を報告する。

- A.1: 残余バケツ別に、流れ込んだ元の葉要素を金額の大きい順に(上位30件)
- A.2: A.1 上位30要素のうち、同じ親(subtotal)を持ち同一社内での共起が
  稀な要素の組(排他性スコア = P(A and B) / min(P(A), P(B)) < 0.1)
- A.3: NetAssets 側で未分類のまま残る要素
- A.4: 金融業判定(unclassified_value / Assets の分布、閾値5%で切った場合の
  除外社数と filerName 上位50件)

B(canonical の修正)はこのスクリプトの範囲外。診断のみ行う。

使い方:
    uv run python scripts/inspect_other_bucket.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import yaml
from lxml import etree
from tqdm import tqdm

from fsgen.parsing.calculation_link import CalcTree, parse_calculation_link_bytes, select_tree
from fsgen.parsing.instance import context_id_for_role, read_fact, verify_context_period
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[1] / "conf" / "config.yaml"
_DEFAULT_DOCUMENT_LIST_CSV = Path("data/raw/index/document_list.csv")
_DEFAULT_DOCUMENTS_DIR = Path("data/raw/documents")
_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")

_TOP_N = 30
_EXCLUSIVITY_THRESHOLD = 0.1
_FINANCIAL_RATIO_THRESHOLD = 0.05


def _load_calc_link_config() -> tuple[dict[str, str], tuple[str, ...]]:
    with _CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)["calculation_link"]
    return config["balance_sheet_roles"], tuple(config["standard_prefixes"])


@dataclass(frozen=True)
class BasisAccount:
    element_id: str
    parent_subtotal: str | None
    side: str
    sign: str
    ratio: float | None


@dataclass(frozen=True)
class SubtotalAccount:
    element_id: str
    side: str
    ratio: float | None


@dataclass(frozen=True)
class CanonicalConfig:
    name: str
    basis: tuple[BasisAccount, ...]
    subtotals: tuple[SubtotalAccount, ...]
    totals: dict[str, str]

    @property
    def basis_ids(self) -> frozenset[str]:
        return frozenset(b.element_id for b in self.basis)

    @property
    def subtotal_ids(self) -> frozenset[str]:
        return frozenset(s.element_id for s in self.subtotals)


def load_canonical(path: Path) -> CanonicalConfig:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    basis = tuple(BasisAccount(**b) for b in doc["basis"])
    subtotals = tuple(SubtotalAccount(**s) for s in doc["subtotals"])
    return CanonicalConfig(name=doc["name"], basis=basis, subtotals=subtotals, totals=doc["totals"])


def _load_document_index(document_list_csv: Path) -> dict[str, dict[str, str]]:
    with document_list_csv.open(newline="", encoding="utf-8") as f:
        return {row["docID"]: row for row in csv.DictReader(f)}


def _walk_to_canonical_or_root(
    leaf: str, tree: CalcTree, canonical_ids: frozenset[str]
) -> tuple[str | None, list[str]]:
    """leaf から親を辿る。canonical に当たればそのIDと辿った経路を、
    当たらず根に着いたら (None, 経路) を返す。経路には leaf 自身も含む。"""
    u = leaf
    path = [leaf]
    while u not in canonical_ids:
        parents = tree.parents_of(u)
        if not parents:
            return None, path
        u = parents[0].parent
        path.append(u)
    return u, path


@dataclass
class LeafObservation:
    """1社1葉要素についての、__OTHER__ バケツへの寄与。"""

    bucket: str  # __OTHER__<subtotal>
    leaf: str
    value: int


@dataclass
class CompanyRecord:
    doc_id: str
    edinet_code: str
    filer_name: str
    parse_status: str
    other_observations: list[LeafObservation] = field(default_factory=list)
    unclassified_net_assets: list[tuple[str, int]] = field(default_factory=list)
    unclassified_value: int = 0
    assets_total: int | None = None


def _process_company(
    doc_id: str,
    edinet_code: str,
    filer_name: str,
    period_end: str,
    documents_dir: Path,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...],
    canonical: CanonicalConfig,
) -> CompanyRecord:
    try:
        zip_path = documents_dir / f"{doc_id}.zip"
        cal_bytes = read_cal_xml(zip_path)
        trees = parse_calculation_link_bytes(
            cal_bytes, role_map, standard_prefixes, source_label=doc_id
        )
        tree, used_role = select_tree(trees)

        xbrl_bytes = read_instance_xbrl(zip_path)
        root = etree.fromstring(xbrl_bytes)
        context_id = context_id_for_role(used_role)
        if not verify_context_period(root, context_id, period_end):
            pass  # 期間不一致は inspect_values.py 側で既に報告済み。ここでは値だけ読む。

        canonical_ids = canonical.basis_ids | canonical.subtotal_ids
        basis_ids = canonical.basis_ids
        net_assets_total_id = canonical.totals.get("net_assets")

        record = CompanyRecord(
            doc_id=doc_id,
            edinet_code=edinet_code,
            filer_name=filer_name,
            parse_status="ok",
        )

        for leaf in tree.leaves():
            bucket, path = _walk_to_canonical_or_root(leaf, tree, canonical_ids)
            fact = read_fact(root, leaf, context_id)
            value = fact.value if fact.value is not None else 0
            if bucket is None:
                record.unclassified_value += value
                if net_assets_total_id in path and value != 0:
                    record.unclassified_net_assets.append((leaf, value))
                continue
            if bucket in basis_ids:
                continue  # 名前付き基礎科目に分類済み。残余ではない。
            if value != 0:
                record.other_observations.append(
                    LeafObservation(bucket=f"__OTHER__{bucket}", leaf=leaf, value=value)
                )

        assets_fact = read_fact(root, canonical.totals["assets"], context_id)
        record.assets_total = assets_fact.value
        return record
    except Exception as exc:  # noqa: BLE001 - 1社の失敗で全体を止めないため意図的に広く捕捉する
        return CompanyRecord(
            doc_id=doc_id,
            edinet_code=edinet_code,
            filer_name=filer_name,
            parse_status=f"failed: {type(exc).__name__}: {exc}",
        )


def run(
    document_list_csv: Path,
    documents_dir: Path,
    canonical_path: Path,
    show_progress: bool = True,
) -> tuple[CanonicalConfig, list[CompanyRecord]]:
    role_map, standard_prefixes = _load_calc_link_config()
    canonical = load_canonical(canonical_path)
    index = _load_document_index(document_list_csv)

    results: list[CompanyRecord] = []
    for doc_id, row in tqdm(index.items(), desc="inspect_other_bucket", disable=not show_progress):
        results.append(
            _process_company(
                doc_id,
                row.get("edinetCode", ""),
                row.get("filerName", ""),
                row.get("periodEnd", ""),
                documents_dir,
                role_map,
                standard_prefixes,
                canonical,
            )
        )
    return canonical, results


# --- A.1 ---


@dataclass(frozen=True)
class ElementStats:
    element_id: str
    n_companies: int
    ratio: float
    median_share: float
    total_share: float
    total_value: int


def analyze_bucket_contents(
    results: list[CompanyRecord], bucket: str, n_ok: int
) -> list[ElementStats]:
    per_element_values: dict[str, list[int]] = defaultdict(list)
    per_company_bucket_total: dict[str, int] = defaultdict(int)

    for r in results:
        if r.parse_status != "ok":
            continue
        company_total = sum(o.value for o in r.other_observations if o.bucket == bucket)
        if company_total == 0:
            continue
        per_company_bucket_total[r.doc_id] = company_total
        seen_this_company: dict[str, int] = defaultdict(int)
        for o in r.other_observations:
            if o.bucket == bucket:
                seen_this_company[o.leaf] += o.value
        for element_id, value in seen_this_company.items():
            per_element_values[element_id].append(value)

    grand_total = sum(per_company_bucket_total.values())
    stats: list[ElementStats] = []
    for element_id, values in per_element_values.items():
        n_companies = len(values)
        total_value = sum(values)
        shares = []
        for r in results:
            if r.parse_status != "ok":
                continue
            bucket_total = per_company_bucket_total.get(r.doc_id, 0)
            if bucket_total == 0:
                continue
            leaf_value = sum(
                o.value
                for o in r.other_observations
                if o.bucket == bucket and o.leaf == element_id
            )
            if leaf_value > 0:
                shares.append(leaf_value / bucket_total)
        stats.append(
            ElementStats(
                element_id=element_id,
                n_companies=n_companies,
                ratio=n_companies / n_ok if n_ok else 0.0,
                median_share=statistics.median(shares) if shares else 0.0,
                total_share=(total_value / grand_total) if grand_total else 0.0,
                total_value=total_value,
            )
        )
    return sorted(stats, key=lambda s: -s.total_value)


# --- A.2 ---


def compute_exclusivity_candidates(
    results: list[CompanyRecord], bucket: str, top_elements: list[str]
) -> list[tuple[str, str, float]]:
    """top_elements 内のペアについて、共起の稀さ(排他性スコア)を求める。

    「出現」= その社の当該バケツ内でその要素が非ゼロ値を持つこと。
    """
    presence: dict[str, set[str]] = {e: set() for e in top_elements}
    for r in results:
        if r.parse_status != "ok":
            continue
        present_here = {
            o.leaf for o in r.other_observations if o.bucket == bucket and o.leaf in presence
        }
        for e in present_here:
            presence[e].add(r.doc_id)

    n = len({r.doc_id for r in results if r.parse_status == "ok"})
    if n == 0:
        return []

    candidates: list[tuple[str, str, float]] = []
    for a, b in combinations(top_elements, 2):
        pa, pb = presence[a], presence[b]
        if not pa or not pb:
            continue
        p_a = len(pa) / n
        p_b = len(pb) / n
        p_both = len(pa & pb) / n
        score = p_both / min(p_a, p_b)
        if score < _EXCLUSIVITY_THRESHOLD:
            candidates.append((a, b, score))
    return sorted(candidates, key=lambda x: x[2])


# --- A.4 ---


def compute_financial_sector_candidates(
    results: list[CompanyRecord], threshold: float
) -> list[tuple[str, str, str, float]]:
    """(doc_id, edinet_code, filer_name, unclassified_ratio) を比率降順で返す。"""
    out: list[tuple[str, str, str, float]] = []
    for r in results:
        if r.parse_status != "ok" or not r.assets_total:
            continue
        ratio = r.unclassified_value / r.assets_total
        if ratio >= threshold:
            out.append((r.doc_id, r.edinet_code, r.filer_name, ratio))
    return sorted(out, key=lambda x: -x[3])


def print_report(canonical: CanonicalConfig, results: list[CompanyRecord]) -> None:
    ok = [r for r in results if r.parse_status == "ok"]
    failed = [r for r in results if r.parse_status != "ok"]
    n_ok = len(ok)
    print(f"処理: {len(results)}社(成功 {n_ok} / 失敗 {len(failed)})\n")

    other_buckets = [s.element_id for s in canonical.subtotals]
    top_elements_by_bucket: dict[str, list[str]] = {}

    print("=" * 70)
    print("A.1 残余バケツ別の内訳(上位30件、金額の大きい順)")
    print("=" * 70)
    for subtotal_id in other_buckets:
        bucket = f"__OTHER__{subtotal_id}"
        stats = analyze_bucket_contents(ok, bucket, n_ok)
        if not stats:
            continue
        top = stats[:_TOP_N]
        top_elements_by_bucket[bucket] = [s.element_id for s in top]
        print(f"\n{bucket}:")
        print(
            f"  {'element_id':<55} {'n_companies':>11} {'ratio':>7} "
            f"{'median_share':>12} {'total_share':>11}"
        )
        for s in top:
            print(
                f"  {s.element_id:<55} {s.n_companies:>11} {s.ratio:>7.4f} "
                f"{s.median_share:>12.4f} {s.total_share:>11.4f}"
            )

    print("\n" + "=" * 70)
    print("A.2 概念グループの候補(排他性スコア < 0.1、同一 subtotal 内)")
    print("=" * 70)
    for subtotal_id in other_buckets:
        bucket = f"__OTHER__{subtotal_id}"
        top_elements = top_elements_by_bucket.get(bucket, [])
        if len(top_elements) < 2:
            continue
        candidates = compute_exclusivity_candidates(ok, bucket, top_elements)
        if not candidates:
            continue
        print(f"\n{bucket}:")
        for a, b, score in candidates:
            print(f"  {a}  <->  {b}:  score={score:.4f}")

    print("\n" + "=" * 70)
    print("A.3 純資産側の未分類要素")
    print("=" * 70)
    net_assets_leaf_values: dict[str, list[int]] = defaultdict(list)
    net_assets_companies: dict[str, set[str]] = defaultdict(set)
    for r in ok:
        for leaf, value in r.unclassified_net_assets:
            net_assets_leaf_values[leaf].append(value)
            net_assets_companies[leaf].add(r.doc_id)
    rows = sorted(
        (
            (leaf, len(net_assets_companies[leaf]), sum(values))
            for leaf, values in net_assets_leaf_values.items()
        ),
        key=lambda x: -x[2],
    )
    print(f"  {'element_id':<55} {'n_companies':>11} {'total_value':>18}")
    for leaf, n_companies, total_value in rows[:_TOP_N]:
        ratio = n_companies / n_ok if n_ok else 0.0
        print(f"  {leaf:<55} {n_companies:>11} ({ratio:.4f})  {total_value:>18}")

    print("\n" + "=" * 70)
    print(f"A.4 金融業判定(unclassified_value / Assets >= {_FINANCIAL_RATIO_THRESHOLD:.0%})")
    print("=" * 70)
    financial_candidates = compute_financial_sector_candidates(ok, _FINANCIAL_RATIO_THRESHOLD)
    print(f"  該当社数: {len(financial_candidates)} / {n_ok}")
    print(f"  {'docID':<12} {'edinetCode':<10} {'ratio':>8}  filerName")
    for doc_id, edinet_code, filer_name, ratio in financial_candidates[:50]:
        print(f"  {doc_id:<12} {edinet_code:<10} {ratio:>8.4f}  {filer_name}")

    filer_counter = Counter(name for _, _, name, _ in financial_candidates)
    print("\n  filerName 頻度上位(重複がある場合の参考):")
    for name, count in filer_counter.most_common(20):
        print(f"    {name}: {count}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-list-csv", type=Path, default=_DEFAULT_DOCUMENT_LIST_CSV)
    parser.add_argument("--documents-dir", type=Path, default=_DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    args = parser.parse_args()

    canonical, results = run(args.document_list_csv, args.documents_dir, args.canonical)
    print_report(canonical, results)


if __name__ == "__main__":
    main()
