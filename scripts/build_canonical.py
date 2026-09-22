#!/usr/bin/env python
"""canonical 勘定科目リストを生成する。

`data/interim/survey/{element_frequency,arc_frequency}.csv` (ステージ3.0の
集計結果)を読み、出現率のしきい値で「基礎科目」を選ぶ。加えて、階層上位の
小計科目(CurrentAssets 等)を固定リストとして扱い、各小計の下に「その他」を
受ける残余の基礎科目を自動生成する。

ステージ4.5 B で以下を追加した:

- Assets/Liabilities/NetAssets 自身も小計科目として扱い、直下に残余科目
  (``__OTHER__jppfs_cor_Assets`` 等)を生成する。これにより、どの基礎科目・
  下位小計にも当たらず合計科目まで到達した葉が必ずどこかの残余科目に
  収容されるようになり、「未分類」が構造的に発生しなくなる(§B.1)。
  ``totals:`` セクションからは外さず、小計としても機能する二重の役割を持つ。
- 出現率だけでは判断できない、複数の要素IDをまとめて1つの基礎科目として
  扱う「グループ化科目」を手動で定義できるようにした(§B.2)。実データに
  存在しない要素IDが指定されていれば警告する。グループの出現率は各構成
  要素の出現率の単純和(1.0で頭打ち)による近似値であり、正確な値は
  inspect_values.py が実データから計算する。
- ``always_include=True`` の科目は、出現率のしきい値を無視して core/extended
  の両方に必ず含める。借入金のように「0であること自体が意味を持つ」科目を
  出現率で足切りしないため。

使い方:
    uv run python scripts/build_canonical.py --min-ratio 0.9 --out conf/accounts/core.yaml
    uv run python scripts/build_canonical.py --min-ratio 0.5 --out conf/accounts/extended.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_DEFAULT_ELEMENT_FREQUENCY_CSV = Path("data/interim/survey/element_frequency.csv")
_DEFAULT_ARC_FREQUENCY_CSV = Path("data/interim/survey/arc_frequency.csv")

# 合計科目。状態ベクトルには含めず、totals: セクションに別枠で記録する(§4.2)。
_TOTALS = {
    "assets": "jppfs_cor_Assets",
    "liabilities": "jppfs_cor_Liabilities",
    "net_assets": "jppfs_cor_NetAssets",
    "liabilities_and_net_assets": "jppfs_cor_LiabilitiesAndNetAssets",
}

# 階層上位の小計科目。出現率では選ばず固定で含める。基礎科目の候補からは除く。
# Assets/Liabilities/NetAssets 自身も含める(§B.1) — totals: との二重登録になるが、
# 「未分類」を構造的に無くすためにあえてそうする。
_FIXED_SUBTOTALS = (
    "jppfs_cor_CurrentAssets",
    "jppfs_cor_NoncurrentAssets",
    "jppfs_cor_CurrentLiabilities",
    "jppfs_cor_NoncurrentLiabilities",
    "jppfs_cor_ShareholdersEquity",
    "jppfs_cor_ValuationAndTranslationAdjustments",
    "jppfs_cor_Assets",
    "jppfs_cor_Liabilities",
    "jppfs_cor_NetAssets",
)

_OTHER_PREFIX = "__OTHER__"


@dataclass(frozen=True)
class ManualGroup:
    """複数の要素IDをまとめて1つの基礎科目として扱う手動定義(§B.2)。"""

    name: str
    members: tuple[str, ...]
    parent_subtotal: str
    side: str
    always_include: bool = False
    include_reason: str | None = None


# 出現率の低さ・taxonomy 改訂による表記揺れのため単独では基礎科目に選ばれない
# が、実務的な意味(同一概念の別名、あるいは0自体に意味がある)から手動で
# まとめる科目群。ステージ4.5 A.2 で検出した排他的ペア、および A.3 で判明した
# 純資産直下の未分類要素(非支配株主持分・新株予約権)に基づく。
_MANUAL_GROUPS: tuple[ManualGroup, ...] = (
    ManualGroup(
        "receivables_trade",
        (
            "jppfs_cor_AccountsReceivableTrade",
            "jppfs_cor_NotesAndAccountsReceivableTrade",
            "jppfs_cor_NotesAndAccountsReceivableTradeAndContractAssets",
            "jppfs_cor_NotesReceivableTrade",
        ),
        "jppfs_cor_CurrentAssets",
        "assets",
    ),
    ManualGroup(
        "inventories",
        (
            "jppfs_cor_Inventories",
            "jppfs_cor_MerchandiseAndFinishedGoods",
            "jppfs_cor_Merchandise",
            "jppfs_cor_FinishedGoods",
            "jppfs_cor_WorkInProcess",
            "jppfs_cor_RawMaterialsAndSupplies",
            "jppfs_cor_RawMaterials",
            "jppfs_cor_Supplies",
        ),
        "jppfs_cor_CurrentAssets",
        "assets",
    ),
    ManualGroup(
        "short_term_investment_securities",
        ("jppfs_cor_ShortTermInvestmentSecurities",),
        "jppfs_cor_CurrentAssets",
        "assets",
    ),
    ManualGroup(
        "allowance_doubtful_current",
        (
            "jppfs_cor_AllowanceForDoubtfulAccountsCA",
            "jppfs_cor_AllowanceForDoubtfulAccountsLeaseReceivablesCA",
            "jppfs_cor_AllowanceForDoubtfulAccountsNotesAndAccountsReceivableTrade",
            "jppfs_cor_AllowanceForDoubtfulAccountsNotesAndAccountsReceivableTradeAndContractAssets",
            "jppfs_cor_AllowanceForDoubtfulAccountsAccountsReceivableTrade",
            "jppfs_cor_AllowanceForDoubtfulAccountsShortTermLoansReceivableFromSubsidiariesAndAffiliates",
        ),
        "jppfs_cor_CurrentAssets",
        "assets",
        always_include=True,
        include_reason="控除科目。残余(流動資産その他)が負になる主因(ステージ16)",
    ),
    ManualGroup(
        "allowance_doubtful_noncurrent",
        (
            "jppfs_cor_AllowanceForDoubtfulAccountsIOAByGroup",
            "jppfs_cor_AllowanceForInvestmentLoss",
            "jppfs_cor_AllowanceForDoubtfulAccountsClaimsInBankruptcyReorganizationClaimsAndOther",
            "jppfs_cor_AllowanceForDoubtfulAccountsLongTermLoansReceivable",
            "jppfs_cor_AllowanceForDoubtfulAccountsLongTermLoansReceivableFromEmployees",
        ),
        "jppfs_cor_NoncurrentAssets",
        "assets",
        always_include=True,
        include_reason="控除科目。従来は投資その他の資産に純額で含まれていた(ステージ16)",
    ),
    ManualGroup(
        "payables_trade",
        (
            "jppfs_cor_AccountsPayableTrade",
            "jppfs_cor_NotesAndAccountsPayableTrade",
            "jppfs_cor_NotesPayableTrade",
        ),
        "jppfs_cor_CurrentLiabilities",
        "liabilities",
        always_include=True,
        include_reason="運転資本の主要項目。出現率は表記分裂の名残",
    ),
    ManualGroup(
        "short_term_loans",
        (
            "jppfs_cor_ShortTermLoansPayable",
            "jppfs_cor_CurrentPortionOfLongTermLoansPayable",
        ),
        "jppfs_cor_CurrentLiabilities",
        "liabilities",
        always_include=True,
    ),
    ManualGroup(
        "long_term_loans",
        ("jppfs_cor_LongTermLoansPayable",),
        "jppfs_cor_NoncurrentLiabilities",
        "liabilities",
        always_include=True,
    ),
    ManualGroup(
        # ステージ4.6まではBondsPayable(NoncurrentLiabilities配下)と
        # CurrentPortionOfBonds(CurrentLiabilities配下)を「社債」として
        # 1つにまとめていたが、これがクロス小計の残余異常を大量に生む
        # 原因になった(ステージ4.6 F.2)。制約行列を書くには「各基礎科目は
        # 厳密に1つの小計に属する」という不変条件が必要なため(ステージ5
        # §A.1)、2つに分割する。
        "bonds_noncurrent",
        ("jppfs_cor_BondsPayable",),
        "jppfs_cor_NoncurrentLiabilities",
        "liabilities",
        always_include=True,
    ),
    ManualGroup(
        "bonds_current",
        ("jppfs_cor_CurrentPortionOfBonds",),
        "jppfs_cor_CurrentLiabilities",
        "liabilities",
        always_include=True,
    ),
    ManualGroup(
        "accounts_payable_other",
        ("jppfs_cor_AccountsPayableOther",),
        "jppfs_cor_CurrentLiabilities",
        "liabilities",
        always_include=True,
        include_reason="未払金。金額規模が大きい",
    ),
    ManualGroup(
        "valuation_difference_securities",
        ("jppfs_cor_ValuationDifferenceOnAvailableForSaleSecurities",),
        "jppfs_cor_ValuationAndTranslationAdjustments",
        "net_assets",
        always_include=True,
        include_reason="純資産の主要変動要因",
    ),
    ManualGroup(
        "foreign_currency_translation",
        ("jppfs_cor_ForeignCurrencyTranslationAdjustment",),
        "jppfs_cor_ValuationAndTranslationAdjustments",
        "net_assets",
    ),
    ManualGroup(
        # NetAssets 直下(ShareholdersEquity の下ではない)。
        "non_controlling_interests",
        ("jppfs_cor_NonControllingInterests",),
        "jppfs_cor_NetAssets",
        "net_assets",
        always_include=True,
        include_reason="連結BSの構造上必須",
    ),
    ManualGroup(
        "subscription_rights",
        ("jppfs_cor_SubscriptionRightsToShares",),
        "jppfs_cor_NetAssets",
        "net_assets",
    ),
)


@dataclass(frozen=True)
class BasisEntry:
    name: str
    members: tuple[str, ...]
    parent_subtotal: str | None
    side: str
    sign: str
    ratio: float | None
    always_include: bool = False
    include_reason: str | None = None


@dataclass(frozen=True)
class SubtotalEntry:
    element_id: str
    side: str
    ratio: float | None


@dataclass(frozen=True)
class BuildReport:
    missing_members: tuple[str, ...]
    side_mismatches: tuple[str, ...]
    manual_groups_included: tuple[str, ...]
    manual_groups_excluded: tuple[str, ...]


def load_element_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_children_map(arc_frequency_csv: Path) -> dict[str, set[str]]:
    """parent -> {child, ...}。全社の和集合(どの社にも一度でも現れた辺)。"""
    children: dict[str, set[str]] = defaultdict(set)
    with arc_frequency_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            children[row["parent"]].add(row["child"])
    return children


def build_parents_map(children: dict[str, set[str]]) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    for parent, child_set in children.items():
        for child in child_set:
            parents[child].add(parent)
    return parents


def descendants_of(root: str, children: dict[str, set[str]]) -> set[str]:
    """root から parent->child 辺を辿って到達できる要素の集合(root 自身は含まない)。"""
    seen: set[str] = set()
    queue: deque[str] = deque(children.get(root, ()))
    seen.update(queue)
    while queue:
        node = queue.popleft()
        for child in children.get(node, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def find_parent_subtotal(
    element_id: str, parents: dict[str, set[str]], fixed_subtotals: tuple[str, ...]
) -> str | None:
    """element_id から親を遡り、最初に見つかった固定小計科目を返す(表示用)。

    合計科目(Assets等)自身も fixed_subtotals に含まれるため、他に小計が
    見つからなければ合計科目自身が返る(§B.1)。複数経路がある場合は最短
    距離のものを選ぶ。同距離で複数見つかった場合は、Assets/Liabilities/
    NetAssets自身よりも、より具体的な下位小計(CurrentAssets等)を優先する
    (単に辞書順だと "Assets" が "NoncurrentAssets" より先に来てしまい、
    情報量の少ない表示になるため)。これは表示用のラベルに過ぎず、実際の
    小計整合性検証は各葉の木構造上の実際の位置(region)で行う。
    """
    subtotal_set = set(fixed_subtotals)
    total_level = set(_TOTALS.values())
    visited = {element_id}
    queue: deque[str] = deque([element_id])
    while queue:
        level_size = len(queue)
        found_at_this_level: list[str] = []
        for _ in range(level_size):
            node = queue.popleft()
            for parent in sorted(parents.get(node, ())):
                if parent in subtotal_set:
                    found_at_this_level.append(parent)
                    continue
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        if found_at_this_level:
            return min(found_at_this_level, key=lambda x: (x in total_level, x))
    return None


def select_basis_candidates(
    rows: list[dict[str, str]], min_ratio: float, excluded: set[str]
) -> list[dict[str, str]]:
    return [
        r
        for r in rows
        if r["is_extension"] == "False"
        and float(r["ratio"]) >= min_ratio
        and r["element_id"] not in excluded
    ]


def drop_ancestor_descendant_conflicts(
    candidates: list[dict[str, str]], children: dict[str, set[str]]
) -> list[dict[str, str]]:
    """基礎科目候補どうしで祖先・子孫の関係にあるペアから、祖先(粗い側)だけを残す。

    例: BuildingsAndStructuresNet(純額)と BuildingsAndStructures(総額)が
    どちらも候補に残った場合、純額側だけを残す(§2.1)。
    """
    ids = [r["element_id"] for r in candidates]
    n_companies = {r["element_id"]: int(r["n_companies"]) for r in candidates}
    desc_cache = {eid: descendants_of(eid, children) for eid in ids}

    excluded: set[str] = set()
    for i, x in enumerate(ids):
        for y in ids[i + 1 :]:
            x_ancestor_of_y = y in desc_cache[x]
            y_ancestor_of_x = x in desc_cache[y]
            if x_ancestor_of_y and y_ancestor_of_x:
                loser = x if n_companies[x] < n_companies[y] else y
                log.warning(
                    "mutual ancestor/descendant conflict between %s and %s "
                    "(inconsistent across companies); dropping %s (fewer companies)",
                    x,
                    y,
                    loser,
                )
                excluded.add(loser)

    remaining = [r for r in candidates if r["element_id"] not in excluded]
    remaining_ids = {r["element_id"] for r in remaining}

    kept: list[dict[str, str]] = []
    for r in remaining:
        eid = r["element_id"]
        has_candidate_ancestor = any(
            eid in desc_cache[other] for other in remaining_ids if other != eid
        )
        if has_candidate_ancestor:
            continue
        kept.append(r)
    return kept


def validate_antichain(element_ids: list[str], children: dict[str, set[str]]) -> None:
    """基礎科目(の構成要素)が反鎖(互いに祖先・子孫でない)であることを確認する。"""
    for i, x in enumerate(element_ids):
        desc_x = descendants_of(x, children)
        for y in element_ids[i + 1 :]:
            if y in desc_x:
                raise ValueError(f"basis accounts are not an antichain: {x} is an ancestor of {y}")


def validate_single_subtotal_membership(
    basis: list[BasisEntry], parents: dict[str, set[str]], fixed_subtotals: tuple[str, ...]
) -> None:
    """各基礎科目が、その全メンバーについて厳密に1つの小計にのみ属することを
    検証する(ステージ5 §A.1)。制約行列 C の各行(小計 = Σ配下基礎科目)を
    書くための前提条件。

    メンバーの実際の所属小計は、宣言された parent_subtotal ではなく、
    全社の和集合(arc_frequency.csv)上で find_parent_subtotal を辿って
    決定的に求め、それが宣言と一致するか確認する。1つでも不一致があれば
    エラーで止める(ステージ4.6の bonds のように、1つの基礎科目が複数の
    小計にまたがることを防ぐ)。
    """
    violations: list[str] = []
    for b in basis:
        if not b.members or b.parent_subtotal is None:
            continue
        for member in b.members:
            actual = find_parent_subtotal(member, parents, fixed_subtotals)
            if actual != b.parent_subtotal:
                violations.append(
                    f"{b.name}:{member} declared parent_subtotal={b.parent_subtotal!r} "
                    f"but actually belongs to {actual!r}"
                )
    if violations:
        raise ValueError(
            "basis accounts must each belong to exactly one subtotal; "
            f"found {len(violations)} violation(s): {violations}"
        )


def classify_side(
    element_id: str,
    descendants_assets: set[str],
    descendants_liabilities: set[str],
    descendants_net_assets: set[str],
) -> str | None:
    # Assets/Liabilities/NetAssets 自身は自分自身の子孫集合には含まれないため特別扱いする。
    if element_id == _TOTALS["assets"]:
        return "assets"
    if element_id == _TOTALS["liabilities"]:
        return "liabilities"
    if element_id == _TOTALS["net_assets"]:
        return "net_assets"
    if element_id in descendants_assets:
        return "assets"
    if element_id in descendants_liabilities:
        return "liabilities"
    if element_id in descendants_net_assets:
        return "net_assets"
    return None


def _build_manual_basis(
    ratio_by_id: dict[str, float],
    side_of: Callable[[str], str | None],
    min_ratio: float,
) -> tuple[list[BasisEntry], BuildReport]:
    manual_basis: list[BasisEntry] = []
    missing_members: list[str] = []
    side_mismatches: list[str] = []
    included: list[str] = []
    excluded_names: list[str] = []

    for group in _MANUAL_GROUPS:
        member_ratios: list[float] = []
        for member in group.members:
            if member not in ratio_by_id:
                missing_members.append(f"{group.name}:{member}")
                continue
            member_ratios.append(ratio_by_id[member])
            actual_side = side_of(member)
            if actual_side != group.side:
                side_mismatches.append(
                    f"{group.name}:{member} (declared={group.side}, actual={actual_side})"
                )
        approx_ratio = min(1.0, sum(member_ratios)) if member_ratios else 0.0
        is_included = group.always_include or approx_ratio >= min_ratio
        if not is_included:
            excluded_names.append(group.name)
            continue
        included.append(group.name)
        manual_basis.append(
            BasisEntry(
                name=group.name,
                members=group.members,
                parent_subtotal=group.parent_subtotal,
                side=group.side,
                sign="free",
                ratio=round(approx_ratio, 4),
                always_include=group.always_include,
                include_reason=group.include_reason,
            )
        )

    report = BuildReport(
        missing_members=tuple(missing_members),
        side_mismatches=tuple(side_mismatches),
        manual_groups_included=tuple(included),
        manual_groups_excluded=tuple(excluded_names),
    )
    return manual_basis, report


def build_canonical(
    element_frequency_csv: Path, arc_frequency_csv: Path, min_ratio: float
) -> tuple[list[BasisEntry], list[SubtotalEntry], BuildReport]:
    rows = load_element_rows(element_frequency_csv)
    ratio_by_id = {r["element_id"]: float(r["ratio"]) for r in rows}
    children = build_children_map(arc_frequency_csv)
    parents = build_parents_map(children)

    descendants_assets = descendants_of(_TOTALS["assets"], children)
    descendants_liabilities = descendants_of(_TOTALS["liabilities"], children)
    descendants_net_assets = descendants_of(_TOTALS["net_assets"], children)

    def side_of(eid: str) -> str | None:
        return classify_side(
            eid, descendants_assets, descendants_liabilities, descendants_net_assets
        )

    # --- 小計科目(固定。Assets/Liabilities/NetAssets 自身も含む、§B.1) ---
    subtotals: list[SubtotalEntry] = []
    for subtotal_id in _FIXED_SUBTOTALS:
        side = side_of(subtotal_id)
        if side is None:
            raise ValueError(
                f"fixed subtotal {subtotal_id} is not under Assets/Liabilities/NetAssets"
            )
        ratio = ratio_by_id.get(subtotal_id)
        if ratio is not None and ratio < 0.9:
            log.warning("fixed subtotal %s has ratio %.4f (below 90%%)", subtotal_id, ratio)
        subtotals.append(SubtotalEntry(element_id=subtotal_id, side=side, ratio=ratio))

    # --- グループ化科目(手動定義、§B.2) ---
    manual_basis, report = _build_manual_basis(ratio_by_id, side_of, min_ratio)
    manual_member_ids = {m for b in manual_basis for m in b.members}
    if report.missing_members:
        log.warning("manual group member(s) not found in real data: %s", report.missing_members)
    if report.side_mismatches:
        log.warning("manual group member(s) with unexpected side: %s", report.side_mismatches)

    # --- 基礎科目(出現率ベース、グループに取り込まれなかった要素のみ対象) ---
    excluded = set(_TOTALS.values()) | set(_FIXED_SUBTOTALS) | manual_member_ids
    candidates = select_basis_candidates(rows, min_ratio, excluded)
    kept = drop_ancestor_descendant_conflicts(candidates, children)
    validate_antichain([r["element_id"] for r in kept], children)

    auto_basis: list[BasisEntry] = []
    unclassified: list[str] = []
    for r in kept:
        eid = r["element_id"]
        side = side_of(eid)
        if side is None:
            unclassified.append(eid)
            continue
        parent_subtotal = find_parent_subtotal(eid, parents, _FIXED_SUBTOTALS)
        auto_basis.append(
            BasisEntry(
                name=eid,
                members=(eid,),
                parent_subtotal=parent_subtotal,
                side=side,
                # この段階では実データの値を見ていないため全て free とする(§3.2)。
                # 推測で nonneg/nonpos を振らないこと。
                sign="free",
                ratio=round(float(r["ratio"]), 4),
            )
        )

    if unclassified:
        log.warning(
            "%d candidate(s) could not be classified as assets/liabilities/net_assets "
            "(not reachable from any total) and were dropped: %s",
            len(unclassified),
            unclassified,
        )

    basis = manual_basis + auto_basis

    validate_single_subtotal_membership(basis, parents, _FIXED_SUBTOTALS)

    no_parent_subtotal = [b.name for b in basis if b.parent_subtotal is None]
    if no_parent_subtotal:
        log.warning(
            "%d basis account(s) have no fixed subtotal ancestor "
            "(directly under a total, e.g. NetAssets): %s",
            len(no_parent_subtotal),
            no_parent_subtotal,
        )

    # --- 残余(その他)科目: 小計ごとに1つ自動生成(Assets/Liabilities/NetAssets 直下も含む) ---
    for subtotal in subtotals:
        basis.append(
            BasisEntry(
                name=f"{_OTHER_PREFIX}{subtotal.element_id}",
                members=(),
                parent_subtotal=subtotal.element_id,
                side=subtotal.side,
                sign="free",
                ratio=None,
            )
        )

    # 反鎖検証: 実際に値を持ちうる全構成要素(グループメンバー含む、残余科目は
    # members=() なので対象外)で確認する。
    all_member_ids = sorted({m for b in basis for m in b.members})
    validate_antichain(all_member_ids, children)

    basis.sort(key=lambda b: (b.side, b.parent_subtotal or "", -(b.ratio or 0.0), b.name))
    subtotals.sort(key=lambda s: (s.side, s.element_id))
    return basis, subtotals, report


def write_yaml(
    basis: list[BasisEntry],
    subtotals: list[SubtotalEntry],
    name: str,
    min_ratio: float,
    element_frequency_csv: Path,
    out_path: Path,
) -> None:
    document = {
        "name": name,
        "min_ratio": min_ratio,
        "generated_from": str(element_frequency_csv),
        "basis": [
            {
                "name": b.name,
                "members": list(b.members),
                "parent_subtotal": b.parent_subtotal,
                "side": b.side,
                "sign": b.sign,
                "ratio": b.ratio,
                "always_include": b.always_include,
                "include_reason": b.include_reason,
            }
            for b in basis
        ],
        "subtotals": [asdict(s) for s in subtotals],
        "totals": dict(_TOTALS),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(document, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-ratio", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--element-frequency-csv", type=Path, default=_DEFAULT_ELEMENT_FREQUENCY_CSV
    )
    parser.add_argument("--arc-frequency-csv", type=Path, default=_DEFAULT_ARC_FREQUENCY_CSV)
    parser.add_argument(
        "--name", type=str, default=None, help="省略時は --out のファイル名(拡張子抜き)"
    )
    args = parser.parse_args()

    name = args.name or args.out.stem
    basis, subtotals, report = build_canonical(
        args.element_frequency_csv, args.arc_frequency_csv, args.min_ratio
    )
    write_yaml(basis, subtotals, name, args.min_ratio, args.element_frequency_csv, args.out)

    n_other = sum(1 for b in basis if b.name.startswith(_OTHER_PREFIX))
    manual_group_names = {g.name for g in _MANUAL_GROUPS}
    n_manual = sum(1 for b in basis if b.name in manual_group_names)
    print(
        f"{args.out}: 基礎科目 {len(basis)}(うち残余 {n_other}, グループ化 {n_manual}) "
        f"+ 小計科目 {len(subtotals)} = D={len(basis) + len(subtotals)}"
    )
    print(f"  グループ化科目のうち採用: {list(report.manual_groups_included)}")
    print(f"  グループ化科目のうち出現率不足で不採用: {list(report.manual_groups_excluded)}")
    if report.missing_members:
        print(f"  実データに存在しなかったメンバー要素: {list(report.missing_members)}")
    if report.side_mismatches:
        print(f"  side不一致のメンバー要素: {list(report.side_mismatches)}")


if __name__ == "__main__":
    main()
