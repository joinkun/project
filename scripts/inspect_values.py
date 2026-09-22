#!/usr/bin/env python
"""canonical (core/extended) で全社を集約し、カバレッジ・欠損・会計恒等式を検証する。

ZIP は展開しない。出力は data/interim/values/ 配下。

ステージ4.6 A で残余科目の定義を「積み上げ」から「差額」に変更した:

    __OTHER__<s> := 報告された小計 s の値
                    - Σ(s の直下の小計、報告値)
                    - Σ(parent_subtotal が s である基礎科目の値)

こうすると、小計・合計のファクトが全て揃っている限り、恒等式
(Assets = Σ資産側科目、等)は整数の減算だけで構成的に厳密に 0 になる。
小計自体が欠損している社(または直下の小計が欠損している社)は、
差額を計算できないため、旧来の「葉を積み上げる」方式にフォールバック
する(§A注意)。フォールバックが発生した社・小計は必ず記録して報告する。

基礎科目自体(__OTHER__ でないもの)は引き続き、葉から祖先を辿って
最初に基礎科目のメンバーか小計に当たったところで積み上げる(§2.1
「粗い方を取る」原則、ステージ4.5で復元済み)。

§C: 総資産に対する __OTHER__Assets の比率が閾値を超える社を「金融業」
として `is_financial` フラグ列に記録する(この段階では除外しない)。

使い方:
    uv run python scripts/inspect_values.py
    uv run python scripts/inspect_values.py --canonical conf/accounts/extended.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from lxml import etree
from tqdm import tqdm

from fsgen.parsing.calculation_link import CalcTree, parse_calculation_link_bytes, select_tree
from fsgen.parsing.instance import (
    context_id_for_role,
    read_fact,
    read_text_block_present,
    verify_context_period,
)
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[1] / "conf" / "config.yaml"
_DEFAULT_DOCUMENT_LIST_CSV = Path("data/raw/index/document_list.csv")
_DEFAULT_DOCUMENTS_DIR = Path("data/raw/documents")
_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_OUTPUT_DIR = Path("data/interim/values")

_OTHER_PREFIX = "__OTHER__"
_GOING_CONCERN_ELEMENT = "jpcrp_cor_MaterialMattersRelatingToGoingConcernEtcBusinessRisksTextBlock"
_FINANCIAL_RATIO_THRESHOLD = 0.5
_OLD_STYLE_FINANCIAL_RATIO_THRESHOLD = 0.05
_RESIDUAL_ANOMALY_THRESHOLD = 0.1

# 合計・小計の階層(直下の小計)。差額定義の残余計算に使う。
# 例: Assets の残余 = Assets - CurrentAssets - NoncurrentAssets - (Assets 直下の基礎科目)
_SUBTOTAL_CHILDREN: dict[str, tuple[str, ...]] = {
    "jppfs_cor_Assets": ("jppfs_cor_CurrentAssets", "jppfs_cor_NoncurrentAssets"),
    "jppfs_cor_Liabilities": ("jppfs_cor_CurrentLiabilities", "jppfs_cor_NoncurrentLiabilities"),
    "jppfs_cor_NetAssets": (
        "jppfs_cor_ShareholdersEquity",
        "jppfs_cor_ValuationAndTranslationAdjustments",
    ),
}


def _load_calc_link_config() -> tuple[dict[str, str], tuple[str, ...]]:
    with _CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)["calculation_link"]
    return config["balance_sheet_roles"], tuple(config["standard_prefixes"])


@dataclass(frozen=True)
class BasisAccount:
    name: str
    members: tuple[str, ...]
    parent_subtotal: str | None
    side: str
    sign: str
    ratio: float | None
    always_include: bool = False


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
    def member_to_name(self) -> dict[str, str]:
        return {m: b.name for b in self.basis for m in b.members}

    @property
    def basis_names(self) -> frozenset[str]:
        return frozenset(b.name for b in self.basis)

    @property
    def subtotal_ids(self) -> frozenset[str]:
        return frozenset(s.element_id for s in self.subtotals)

    def basis_names_for_side(self, side: str) -> tuple[str, ...]:
        return tuple(b.name for b in self.basis if b.side == side)

    def basis_names_for_parent(self, parent_subtotal: str) -> tuple[str, ...]:
        return tuple(b.name for b in self.basis if b.parent_subtotal == parent_subtotal)


def load_canonical(path: Path) -> CanonicalConfig:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    basis = tuple(
        BasisAccount(
            name=b["name"],
            members=tuple(b["members"]),
            parent_subtotal=b["parent_subtotal"],
            side=b["side"],
            sign=b["sign"],
            ratio=b["ratio"],
            always_include=b.get("always_include", False),
        )
        for b in doc["basis"]
    )
    subtotals = tuple(SubtotalAccount(**s) for s in doc["subtotals"])
    return CanonicalConfig(name=doc["name"], basis=basis, subtotals=subtotals, totals=doc["totals"])


def _load_document_index(document_list_csv: Path) -> dict[str, dict[str, str]]:
    with document_list_csv.open(newline="", encoding="utf-8") as f:
        return {row["docID"]: row for row in csv.DictReader(f)}


def find_bucket(
    leaf: str,
    tree: CalcTree,
    member_to_name: dict[str, str],
    subtotal_ids: frozenset[str],
) -> str | None:
    """leaf 自身、またはその祖先が最初に基礎科目のメンバーか小計に当たった
    ところで停止し、対応する集計先(基礎科目名、または __OTHER__<小計>)を
    返す。

    葉自身が直接 members に載っていなくても、その祖先(例:
    PropertyPlantAndEquipment のような集約科目)が基礎科目として登録されて
    いれば、そこへ丸める(§2.1「粗い方を取る」原則)。ここで __OTHER__ に
    積み上げられる値は、小計ファクトが欠損している場合のフォールバック
    としてのみ最終的に使われる(§A)。
    """
    u = leaf
    while u not in member_to_name and u not in subtotal_ids:
        parents = tree.parents_of(u)
        if not parents:
            return None
        u = parents[0].parent
    if u in member_to_name:
        return member_to_name[u]
    return f"{_OTHER_PREFIX}{u}"


@dataclass(frozen=True)
class CompanyValues:
    doc_id: str
    edinet_code: str
    filer_name: str
    parse_status: str
    error: str | None
    used_role: str | None
    period_ok: bool | None
    has_going_concern: bool | None
    values: dict[str, int]  # basis account name(__OTHER__含む) -> 集約値
    missing: set[str]  # basis account name -> 一度も実値が乗らなかった(欠損)
    subtotal_facts: dict[str, int | None]  # subtotal element_id -> 実際に報告された値
    fallback_subtotals: frozenset[str]  # 差額計算できず積み上げにフォールバックした小計
    old_style_other_assets: int  # 差額定義を使う前の、積み上げ方式での __OTHER__Assets 値
    residual_anomalies: tuple[tuple[str, int, int, float], ...]  # (小計, 残余, 報告値, 比率)
    is_financial: bool
    totals: dict[str, int | None]  # side -> Assets/Liabilities/NetAssets/LiabilitiesAndNetAssets
    assets_decimals: str | None
    assets_unit_ref: str | None


def _failed(doc_id: str, edinet_code: str, filer_name: str, error: Exception) -> CompanyValues:
    return CompanyValues(
        doc_id=doc_id,
        edinet_code=edinet_code,
        filer_name=filer_name,
        parse_status="failed",
        error=f"{type(error).__name__}: {error}",
        used_role=None,
        period_ok=None,
        has_going_concern=None,
        values={},
        missing=set(),
        subtotal_facts={},
        fallback_subtotals=frozenset(),
        old_style_other_assets=0,
        residual_anomalies=(),
        is_financial=False,
        totals={},
        assets_decimals=None,
        assets_unit_ref=None,
    )


def _process_company_inner(
    doc_id: str,
    edinet_code: str,
    filer_name: str,
    period_end: str,
    documents_dir: Path,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...],
    canonical: CanonicalConfig,
) -> CompanyValues:
    zip_path = documents_dir / f"{doc_id}.zip"
    cal_bytes = read_cal_xml(zip_path)
    trees = parse_calculation_link_bytes(
        cal_bytes, role_map, standard_prefixes, source_label=doc_id
    )
    tree, used_role = select_tree(trees)

    xbrl_bytes = read_instance_xbrl(zip_path)
    root = etree.fromstring(xbrl_bytes)
    context_id = context_id_for_role(used_role)
    period_ok = verify_context_period(root, context_id, period_end)

    member_to_name = canonical.member_to_name
    subtotal_ids = canonical.subtotal_ids
    values: dict[str, int] = defaultdict(int)
    has_value: dict[str, bool] = defaultdict(bool)
    has_leaf: dict[str, bool] = defaultdict(bool)

    # 1. 基礎科目(と、フォールバック用の積み上げ __OTHER__)を葉から計算する。
    for leaf in tree.leaves():
        fact = read_fact(root, leaf, context_id)
        bucket = find_bucket(leaf, tree, member_to_name, subtotal_ids)
        if bucket is None:
            # 全ての葉は Assets/Liabilities/NetAssets のいずれかに到達できる
            # はずなので、ここには通常到達しない。防御的に読み捨てる。
            continue
        has_leaf[bucket] = True
        if fact.value is not None:
            values[bucket] += fact.value
            has_value[bucket] = True

    old_style_other_assets = values.get(f"{_OTHER_PREFIX}jppfs_cor_Assets", 0)

    # 2. 小計ファクトを読む。
    subtotal_facts: dict[str, int | None] = {
        s.element_id: read_fact(root, s.element_id, context_id).value for s in canonical.subtotals
    }

    # 3. 残余科目を差額定義で上書きする。必要なファクトが揃わない場合のみ
    #    フォールバック(手順1で積み上げた値のまま)とする。
    fallback_subtotals: set[str] = set()
    for s in canonical.subtotals:
        other_key = f"{_OTHER_PREFIX}{s.element_id}"
        reported = subtotal_facts.get(s.element_id)
        children = _SUBTOTAL_CHILDREN.get(s.element_id, ())
        children_facts = [subtotal_facts.get(c) for c in children]
        if reported is None or any(v is None for v in children_facts):
            fallback_subtotals.add(s.element_id)
            continue
        direct_basis_names = [
            n for n in canonical.basis_names_for_parent(s.element_id) if n != other_key
        ]
        computed = (
            reported
            - sum(v for v in children_facts if v is not None)
            - sum(values.get(n, 0) for n in direct_basis_names)
        )
        values[other_key] = computed
        has_leaf[other_key] = True
        has_value[other_key] = True

    missing = {cid for cid in has_leaf if not has_value.get(cid, False)}
    # 一度も葉が乗らなかった basis も欠損として扱う。
    all_bucket_names = {b.name for b in canonical.basis}
    missing |= {cid for cid in all_bucket_names if not has_leaf.get(cid, False)}

    totals: dict[str, int | None] = {}
    assets_decimals: str | None = None
    assets_unit_ref: str | None = None
    for side_name, total_element_id in canonical.totals.items():
        fact = read_fact(root, total_element_id, context_id)
        totals[side_name] = fact.value
        if side_name == "assets":
            assets_decimals = fact.decimals
            assets_unit_ref = fact.unit_ref

    # 4. 金融業判定(§C): __OTHER__Assets(最終値、フォールバック込み) / Assets。
    assets_val = totals.get("assets")
    other_assets_final = values.get(f"{_OTHER_PREFIX}jppfs_cor_Assets", 0)
    is_financial = (
        assets_val is not None
        and assets_val != 0
        and (other_assets_final / assets_val) > _FINANCIAL_RATIO_THRESHOLD
    )

    # 5. 残余の異常検知(§A注意): |残余| / |小計| > 0.1 の(小計, 残余)を記録する。
    anomalies: list[tuple[str, int, int, float]] = []
    for s in canonical.subtotals:
        reported = subtotal_facts.get(s.element_id)
        if not reported:
            continue
        other_val = values.get(f"{_OTHER_PREFIX}{s.element_id}", 0)
        ratio = abs(other_val) / abs(reported)
        if ratio > _RESIDUAL_ANOMALY_THRESHOLD:
            anomalies.append((s.element_id, other_val, reported, ratio))

    has_going_concern = read_text_block_present(root, _GOING_CONCERN_ELEMENT)

    return CompanyValues(
        doc_id=doc_id,
        edinet_code=edinet_code,
        filer_name=filer_name,
        parse_status="ok",
        error=None,
        used_role=used_role,
        period_ok=period_ok,
        has_going_concern=has_going_concern,
        values=dict(values),
        missing=missing,
        subtotal_facts=subtotal_facts,
        fallback_subtotals=frozenset(fallback_subtotals),
        old_style_other_assets=old_style_other_assets,
        residual_anomalies=tuple(anomalies),
        is_financial=is_financial,
        totals=totals,
        assets_decimals=assets_decimals,
        assets_unit_ref=assets_unit_ref,
    )


def process_company(
    doc_id: str,
    edinet_code: str,
    filer_name: str,
    period_end: str,
    documents_dir: Path,
    role_map: dict[str, str],
    standard_prefixes: tuple[str, ...],
    canonical: CanonicalConfig,
) -> CompanyValues:
    try:
        return _process_company_inner(
            doc_id,
            edinet_code,
            filer_name,
            period_end,
            documents_dir,
            role_map,
            standard_prefixes,
            canonical,
        )
    except Exception as exc:  # noqa: BLE001 - 1社の失敗で全体を止めないため意図的に広く捕捉する
        return _failed(doc_id, edinet_code, filer_name, exc)


def run(
    document_list_csv: Path,
    documents_dir: Path,
    canonical_path: Path,
    show_progress: bool = True,
) -> tuple[CanonicalConfig, list[CompanyValues]]:
    role_map, standard_prefixes = _load_calc_link_config()
    canonical = load_canonical(canonical_path)
    index = _load_document_index(document_list_csv)

    results: list[CompanyValues] = []
    for doc_id, row in tqdm(index.items(), desc="inspect_values", disable=not show_progress):
        results.append(
            process_company(
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


def write_aggregated_csv(
    canonical: CanonicalConfig, results: list[CompanyValues], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        ["docID", "edinetCode", "used_role", "parse_status", "period_ok", "has_going_concern"]
        + [b.name for b in canonical.basis]
        + [s.element_id for s in canonical.subtotals]
        + [f"total_{k}" for k in canonical.totals]
        + ["is_financial"]
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in results:
            row: list[object] = [
                r.doc_id,
                r.edinet_code,
                r.used_role or "",
                r.parse_status,
                r.period_ok if r.period_ok is not None else "",
                r.has_going_concern if r.has_going_concern is not None else "",
            ]
            for b in canonical.basis:
                row.append("" if b.name in r.missing else r.values.get(b.name, 0))
            for s in canonical.subtotals:
                row.append(r.subtotal_facts.get(s.element_id, "") if r.parse_status == "ok" else "")
            for total_key in canonical.totals:
                row.append(r.totals.get(total_key, ""))
            row.append(r.is_financial if r.parse_status == "ok" else "")
            writer.writerow(row)


@dataclass(frozen=True)
class Residuals:
    doc_id: str
    r1: int | None  # |Σ(資産側基礎科目) - Assets|
    r2: int | None  # |Σ(負債側基礎科目) - Liabilities|
    r3: int | None  # |Σ(純資産側基礎科目) - NetAssets|
    r4: int | None  # |Assets - LiabilitiesAndNetAssets| (貸借一致)
    assets: int | None


def compute_residuals(canonical: CanonicalConfig, r: CompanyValues) -> Residuals | None:
    if r.parse_status != "ok":
        return None

    def side_sum(side: str) -> int:
        return sum(r.values.get(name, 0) for name in canonical.basis_names_for_side(side))

    assets_total = r.totals.get("assets")
    liabilities_total = r.totals.get("liabilities")
    net_assets_total = r.totals.get("net_assets")
    liab_and_net_total = r.totals.get("liabilities_and_net_assets")

    r1 = abs(side_sum("assets") - assets_total) if assets_total is not None else None
    r2 = abs(side_sum("liabilities") - liabilities_total) if liabilities_total is not None else None
    r3 = abs(side_sum("net_assets") - net_assets_total) if net_assets_total is not None else None
    r4 = (
        abs(assets_total - liab_and_net_total)
        if assets_total is not None and liab_and_net_total is not None
        else None
    )
    return Residuals(doc_id=r.doc_id, r1=r1, r2=r2, r3=r3, r4=r4, assets=assets_total)


def write_residuals_csv(
    canonical: CanonicalConfig, results: list[CompanyValues], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["docID", "r1", "r2", "r3", "r4", "assets"])
        for r in results:
            residuals = compute_residuals(canonical, r)
            if residuals is None:
                writer.writerow([r.doc_id, "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    residuals.doc_id,
                    residuals.r1 if residuals.r1 is not None else "",
                    residuals.r2 if residuals.r2 is not None else "",
                    residuals.r3 if residuals.r3 is not None else "",
                    residuals.r4 if residuals.r4 is not None else "",
                    residuals.assets if residuals.assets is not None else "",
                ]
            )


def compute_other_ratio(canonical: CanonicalConfig, r: CompanyValues, side: str) -> float | None:
    """side 側の基礎科目合計に占める、__OTHER__(残余)の比率。"""
    if r.parse_status != "ok":
        return None
    total = sum(r.values.get(name, 0) for name in canonical.basis_names_for_side(side))
    if total == 0:
        return None
    other = sum(
        r.values.get(name, 0)
        for name in canonical.basis_names_for_side(side)
        if name.startswith(_OTHER_PREFIX)
    )
    return other / total


def write_coverage_csv(
    canonical: CanonicalConfig, results: list[CompanyValues], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["docID", "other_ratio_assets", "other_ratio_liabilities", "other_ratio_net_assets"]
        )
        for r in results:
            row = [r.doc_id]
            for side in ("assets", "liabilities", "net_assets"):
                ratio = compute_other_ratio(canonical, r, side)
                row.append(f"{ratio:.6f}" if ratio is not None else "")
            writer.writerow(row)


def _summary_stats(values: list[float]) -> str:
    if not values:
        return "(データなし)"
    q1, _, q3 = statistics.quantiles(values, n=4)
    return (
        f"中央値 {statistics.median(values):.4f}, 四分位 [{q1:.4f}, {q3:.4f}], "
        f"最大 {max(values):.4f}"
    )


def print_summary(canonical: CanonicalConfig, results: list[CompanyValues]) -> None:
    ok = [r for r in results if r.parse_status == "ok"]
    failed = [r for r in results if r.parse_status != "ok"]
    n_ok = len(ok)
    print(f"処理: {len(results)}社(成功 {n_ok} / 失敗 {len(failed)})\n")

    print(
        f"基礎科目数={len(canonical.basis)}, 小計科目数={len(canonical.subtotals)}, "
        f"D={len(canonical.basis) + len(canonical.subtotals)}"
    )

    # --- カバレッジ(残余比率、資産・負債・純資産の各側) ---
    print("\n=== カバレッジ(「その他」への合算比率) ===")
    for side in ("assets", "liabilities", "net_assets"):
        ratios = [
            ratio for r in ok if (ratio := compute_other_ratio(canonical, r, side)) is not None
        ]
        print(f"  {side}: {_summary_stats(ratios)}")

    # --- 欠損 ---
    print("\n=== 欠損(canonical各科目の欠損率) ===")
    for b in canonical.basis:
        n_missing = sum(1 for r in ok if b.name in r.missing)
        rate = n_missing / n_ok if n_ok else 0.0
        print(f"  {b.name}: {rate:.4f} ({n_missing}/{n_ok})")

    # --- 恒等式の検証(差額定義) ---
    print("\n=== 恒等式の検証(差額定義、§A) ===")
    residual_list: list[Residuals] = [
        x for r in ok if (x := compute_residuals(canonical, r)) is not None
    ]
    metrics: list[tuple[str, Callable[[Residuals], int | None]]] = [
        ("r1 (資産)", lambda x: x.r1),
        ("r2 (負債)", lambda x: x.r2),
        ("r3 (純資産)", lambda x: x.r3),
        ("r4 (貸借一致)", lambda x: x.r4),
    ]
    for label, getter in metrics:
        n_nonzero = 0
        nonzero_examples: list[tuple[str, int]] = []
        for x in residual_list:
            val = getter(x)
            if val is None:
                continue
            if val != 0:
                n_nonzero += 1
                nonzero_examples.append((x.doc_id, val))
        n_total = len(residual_list)
        print(f"  {label}: 0でない社数={n_nonzero}/{n_total}(整数完全一致で判定)")
        if nonzero_examples:
            print(f"    例: {sorted(nonzero_examples, key=lambda t: -abs(t[1]))[:10]}")

    # --- 残余の異常検知(§A注意) ---
    print(f"\n=== 残余の異常検知(|残余|/小計 > {_RESIDUAL_ANOMALY_THRESHOLD:.0%}) ===")
    all_anomalies: list[tuple[str, str, int, int, float]] = [
        (r.doc_id, s, other, reported, ratio)
        for r in ok
        for s, other, reported, ratio in r.residual_anomalies
    ]
    n_companies_with_anomaly = len({a[0] for a in all_anomalies})
    print(f"  件数(社×小計): {len(all_anomalies)}件 / 該当社数: {n_companies_with_anomaly}社")
    anomaly_by_subtotal: Counter[str] = Counter(a[1] for a in all_anomalies)
    print("  小計別の内訳:")
    for subtotal_id, count in anomaly_by_subtotal.most_common():
        print(f"    {subtotal_id}: {count}件")
    for doc_id, subtotal_id, other, reported, ratio in sorted(
        all_anomalies, key=lambda a: -a[4]
    )[:10]:
        print(f"    {doc_id} {subtotal_id}: 残余={other} 小計={reported} 比率={ratio:.4f}")

    # --- 小計欠損によるフォールバック ---
    print("\n=== 小計欠損によるフォールバック件数(§A注意) ===")
    fallback_counts: Counter[str] = Counter()
    for r in ok:
        fallback_counts.update(r.fallback_subtotals)
    for subtotal in canonical.subtotals:
        print(f"  {subtotal.element_id}: {fallback_counts.get(subtotal.element_id, 0)}社")
    n_any_fallback = sum(1 for r in ok if r.fallback_subtotals)
    print(f"  いずれかの小計でフォールバックが発生した社: {n_any_fallback}/{n_ok}社")

    # --- 金融業判定(§C) ---
    print(f"\n=== 金融業判定(__OTHER__Assets / Assets > {_FINANCIAL_RATIO_THRESHOLD:.0%}) ===")
    financial = [r for r in ok if r.is_financial]
    print(f"  該当社数: {len(financial)}/{n_ok}社")
    def _other_assets(r: CompanyValues) -> int:
        return r.values.get(f"{_OTHER_PREFIX}jppfs_cor_Assets", 0)

    for r in sorted(financial, key=lambda r: -_other_assets(r))[:50]:
        print(f"    {r.doc_id} {r.edinet_code} {r.filer_name}")

    print(
        f"\n  旧方式(積み上げ __OTHER__Assets / Assets > "
        f"{_OLD_STYLE_FINANCIAL_RATIO_THRESHOLD:.0%})との重なり:"
    )
    old_style_flagged = set()
    for r in ok:
        assets = r.totals.get("assets")
        if assets and (r.old_style_other_assets / assets) > _OLD_STYLE_FINANCIAL_RATIO_THRESHOLD:
            old_style_flagged.add(r.doc_id)
    new_style_flagged = {r.doc_id for r in financial}
    both = old_style_flagged & new_style_flagged
    print(f"    旧方式該当: {len(old_style_flagged)}社")
    print(f"    新方式(is_financial)該当: {len(new_style_flagged)}社")
    print(f"    両方に該当: {len(both)}社")
    print(f"    旧方式のみ: {len(old_style_flagged - new_style_flagged)}社")
    print(f"    新方式のみ: {len(new_style_flagged - old_style_flagged)}社")

    # --- 単位の確認 ---
    print("\n=== 単位の確認(Assets ファクトの decimals/unitRef) ===")
    decimals_counts = Counter(r.assets_decimals for r in ok)
    unit_counts = Counter(r.assets_unit_ref for r in ok)
    print("  decimals別社数:")
    for value, count in sorted(decimals_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {value!r}: {count}")
    print("  unitRef別社数:")
    for value, count in sorted(unit_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {value!r}: {count}")

    # --- 連結/単体とラベルの関係 ---
    print("\n=== 連結/単体と重要事象の関係 ===")
    counts = Counter((r.used_role, r.has_going_concern) for r in ok)
    print(f"{'':10}重要事象あり  なし")
    consolidated_yes = counts.get(("consolidated", True), 0)
    consolidated_no = counts.get(("consolidated", False), 0)
    nonconsolidated_yes = counts.get(("nonconsolidated", True), 0)
    nonconsolidated_no = counts.get(("nonconsolidated", False), 0)
    print(f"連結あり  {consolidated_yes:>8} {consolidated_no:>8}")
    print(f"単体のみ  {nonconsolidated_yes:>8} {nonconsolidated_no:>8}")

    # --- コンテキスト期間の検証 ---
    period_mismatches = [r.doc_id for r in ok if r.period_ok is False]
    print(f"\ncontext期間の不一致: {len(period_mismatches)}社")
    if period_mismatches:
        print(f"  例: {period_mismatches[:20]}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-list-csv", type=Path, default=_DEFAULT_DOCUMENT_LIST_CSV)
    parser.add_argument("--documents-dir", type=Path, default=_DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    canonical, results = run(args.document_list_csv, args.documents_dir, args.canonical)

    csv_path = args.output_dir / "aggregated_core.csv"
    write_aggregated_csv(canonical, results, csv_path)
    write_residuals_csv(canonical, results, args.output_dir / "residuals.csv")
    write_coverage_csv(canonical, results, args.output_dir / "coverage.csv")

    # pyarrow があれば parquet も書く(ステージ6以降の下流処理が使う)。
    try:
        import pandas as pd

        pd.read_csv(csv_path).to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except ImportError:
        log.warning("pyarrow が無いため parquet 出力をスキップしました(CSVのみ)。")

    print_summary(canonical, results)


if __name__ == "__main__":
    main()
