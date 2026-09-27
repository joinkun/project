#!/usr/bin/env python
"""ステージ21-C B.1: 取りこぼし候補 60 要素の割り当て(canonical の統合計画)。

``results/stage21/contract_assets/leak_candidates.csv``(__OTHER__ に落ちていた標準要素で、
既存グループのキーワードに該当し、5 社以上が使うもの)の各要素について、統合先の
グループ(または統合しない)と理由を ``PLAN`` に明示する。D = 32 は変えない。

- 計画表を ``results/stage21/integration_plan.md`` に書く(論文の付録用)
- 候補 60 要素がすべて判定済みかを検査する(判定漏れは例外)
- ``--apply``: ``conf/accounts/core.yaml`` の各グループの members に追記する
- 追記後、core.yaml の members が計画と一致するか、統合した要素が全 4 期で
  グループの親の小計の下にあるか(``leaves.csv`` の集計先)を検査する

    uv run python scripts/integration_plan_stage21.py            # 検査と計画表
    uv run python scripts/integration_plan_stage21.py --apply    # core.yaml に追記
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

_P = "jppfs_cor_"
_MARK = "  # ステージ21-C で追加(scripts/integration_plan_stage21.py)"


@dataclass(frozen=True)
class Decision:
    element: str  # 接頭辞 jppfs_cor_ を除いた要素名
    group: str | None  # 統合先の canonical 科目。None は統合しない
    reason: str


R, P, INV, APO = "receivables_trade", "payables_trade", "inventories", "accounts_payable_other"
STL, LTL, BNC = "short_term_loans", "long_term_loans", "bonds_noncurrent"

PLAN: tuple[Decision, ...] = (
    # --- 電子記録債権・債務 ---
    Decision(
        "ElectronicallyRecordedMonetaryClaimsOperatingCA",
        R,
        "営業取引から生じた電子記録債権。受取手形の代替手段で、売上債権と同じ概念",
    ),
    Decision(
        "ElectronicallyRecordedObligationsOperatingCL",
        P,
        "営業取引から生じた電子記録債務。支払手形の代替手段で、仕入債務と同じ概念",
    ),
    Decision(
        "ElectronicallyRecordedObligationsFacilitiesCL",
        APO,
        "設備投資に係る電子記録債務。営業外の未払で、未払金と同質",
    ),
    Decision(
        "ElectronicallyRecordedObligationsNonOperatingCL",
        APO,
        "営業外の電子記録債務。未払金と同質",
    ),
    # --- 契約資産 ---
    Decision(
        "ContractAssets",
        R,
        "収益認識基準の契約資産。合算表示の要素(…TradeAndContractAssets)と排他で、"
        "売上債権を細かく分けた表示の一部",
    ),
    Decision(
        "AccountsReceivableTradeAndContractAssets",
        R,
        "売掛金と契約資産の合算表示。現行の売上債権 members とすべて排他な別表記"
        "(2023-12-01 版タクソノミで新設)",
    ),
    Decision("AccountsReceivableTradeNet", R, "売掛金の純額表示。売掛金の別表記"),
    # --- 営業債権・債務(業種特有を含む) ---
    Decision("OperatingAccountsReceivableCA", R, "営業未収入金。本業の対価に係る債権"),
    Decision("NotesAndOperatingAccountsReceivableCA", R, "受取手形及び営業未収入金"),
    Decision("OperatingAccountsPayable", P, "営業未払金。本業の仕入・役務に係る債務"),
    Decision("NotesAndOperatingAccountsPayableTrade", P, "支払手形及び営業未払金"),
    Decision(
        "AccountsPayableOperatingSpecific",
        P,
        "業種別様式の営業未払金。本業に係る債務",
    ),
    Decision("AccountsPayableConsignment", P, "委託販売に係る未払金。商品仕入の対価"),
    Decision(
        "AccountsReceivableInstallment",
        R,
        "割賦売掛金。割賦販売による売上債権",
    ),
    Decision(
        "AccountsReceivableInstallmentSalesCALEA",
        R,
        "リース業の割賦債権。割賦販売の対価で、本業の営業債権",
    ),
    Decision(
        "AccountsReceivableLeaseCALEA",
        R,
        "リース業のリース料の未収。本業の営業債権",
    ),
    Decision("AccountsReceivableCARWY", R, "鉄道業の未収運賃。本業の営業債権"),
    Decision(
        "OperatingAccountsReceivableOfHighwayBusinessCAHWY",
        R,
        "高速道路事業の営業未収入金。本業の営業債権",
    ),
    Decision(
        "OperatingAccountsPayableHighwayBusinessCLHWY",
        P,
        "高速道路事業の営業未払金。本業の営業債務",
    ),
    # --- 建設業 ---
    Decision(
        "NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherCNS",
        R,
        "受取手形・完成工事未収入金等。建設業の売上債権",
    ),
    Decision(
        "AccountsReceivableFromCompletedConstructionContractsCNS",
        R,
        "完成工事未収入金。建設業の売掛金",
    ),
    Decision(
        "NotesPayableAccountsPayableForConstructionContractsAndOtherCNS",
        P,
        "支払手形・工事未払金等。建設業の仕入債務",
    ),
    Decision(
        "NotesPayableAccountsPayableForConstructionContractsCNS",
        P,
        "支払手形・工事未払金。建設業の仕入債務",
    ),
    Decision("AccountsPayableForConstructionContractsCNS", P, "工事未払金。建設業の買掛金"),
    Decision(
        "CostsOnUncompletedConstructionContractsCNS",
        INV,
        "未成工事支出金。建設業の仕掛品",
    ),
    Decision(
        "CostsOnUncompletedConstructionContractsAndOtherCNS",
        INV,
        "未成工事支出金等。建設業の仕掛品",
    ),
    Decision("RawMaterialsAndSuppliesCNS", INV, "建設業の材料貯蔵品"),
    Decision("RealEstateForSaleCNS", INV, "建設業の販売用不動産。販売目的の棚卸資産"),
    Decision(
        "AdvancesReceivedOnUncompletedConstructionContractsCNS",
        None,
        "未成工事受入金。顧客からの前受で契約負債に相当し、仕入債務ではない"
        "(契約負債 ContractLiabilities も __OTHER__ に残している)",
    ),
    # --- 不動産業・その他の棚卸資産 ---
    Decision("RealEstateForSale", INV, "販売用不動産。不動産業の商品"),
    Decision("RealEstateForSaleInProcess", INV, "仕掛販売用不動産。不動産業の仕掛品"),
    Decision("OtherInventories", INV, "その他の棚卸資産"),
    Decision("SemiFinishedGoods", INV, "半製品"),
    # --- 借入・社債 ---
    Decision(
        "CommercialPapersLiabilities",
        STL,
        "コマーシャル・ペーパー。短期の資金調達手段で、短期借入金と同じ機能",
    ),
    Decision(
        "ShortTermBondsPayable",
        STL,
        "短期社債。電子 CP の法的形式で、CP と同じく短期の資金調達",
    ),
    Decision(
        "ShortTermLoansPayableToSubsidiariesAndAffiliates",
        STL,
        "関係会社短期借入金。借入先が関係会社であるだけの短期借入金",
    ),
    Decision(
        "CurrentPortionOfLongTermLoansPayableToSubsidiariesAndAffiliates",
        STL,
        "1 年内返済予定の関係会社長期借入金。short_term_loans は 1 年内返済予定の長期借入金を含む",
    ),
    Decision(
        "LongTermLoansPayableToSubsidiariesAndAffiliates",
        LTL,
        "関係会社長期借入金。借入先が関係会社であるだけの長期借入金",
    ),
    Decision(
        "LongTermLoansPayableToRoadBuildingNCLHWY",
        LTL,
        "高速道路事業の道路建設関係長期借入金",
    ),
    Decision(
        "RoadBuildingBondsPayableNCLHWY",
        BNC,
        "高速道路事業の道路建設関係社債。普通社債と同じ負債",
    ),
    Decision(
        "ConvertibleBondTypeBondsWithSubscriptionRightsToShares",
        None,
        "転換社債型新株予約権付社債。資本性を持ち、株式への転換は再建の資本政策そのもの"
        "なので普通社債と混ぜない(58 社、ステージ21-C A.1)",
    ),
    Decision(
        "BondsWithSubscriptionRightsToSharesNCL",
        None,
        "新株予約権付社債。同上の理由で普通社債と混ぜない",
    ),
    # --- 未払金 ---
    Decision("NotesPayableFacilities", APO, "設備関係支払手形。営業外の未払で、未払金と同質"),
    Decision("AccountsPayableFacilities", APO, "設備関係未払金。未払金と同質"),
    Decision(
        "AccountsPayableOtherAndAccruedExpenses",
        APO,
        "未払金及び未払費用の合算表示。未払金の別表記として扱う(未払費用の単独表示は "
        "__OTHER__ に残るため、合算表示の社は未払費用の分だけ未払金が大きくなる)",
    ),
    # --- 統合しない(営業外の債権・長期の未払) ---
    Decision(
        "AccountsReceivableOther",
        None,
        "未収入金。営業外の債権で、売上債権ではない",
    ),
    Decision(
        "AccountsReceivableOtherFromSubsidiariesAndAffiliates",
        None,
        "関係会社未収入金。営業外の債権",
    ),
    Decision(
        "IncomeTaxesReceivable", None, "未収法人税等。税金の還付請求で、対応するグループが無い"
    ),
    Decision(
        "LongTermAccountsPayableOther",
        None,
        "長期未払金。固定負債の営業外の未払で、対応するグループが無い",
    ),
    Decision(
        "LongTermAccountsPayableInstallmentPurchase",
        None,
        "長期割賦未払金。設備の割賦購入で、長期未払金と同じ扱い",
    ),
    Decision(
        "LongTermAccountsPayableToJapanRailwayConstructionTransportAndTechnologyAgencyNCLRWY",
        None,
        "鉄道・運輸機構長期未払金。鉄道施設の譲渡代金で、長期未払金と同じ扱い",
    ),
    Decision("BondIssuanceCostDA", None, "社債発行費。繰延資産で、社債ではない"),
    # --- 統合しない(貸付金・金融業の固有科目) ---
    Decision(
        "AccountsReceivableOperatingLoansCALEA",
        None,
        "リース業の営業貸付金。貸付金(金融資産)で、売上債権ではない",
    ),
    Decision(
        "AccountsReceivableOtherLoansToCustomersCALEA",
        None,
        "リース業のその他の営業貸付債権。貸付金で、売上債権ではない",
    ),
    Decision(
        "AccountsReceivableOperatingLoansCASPF",
        None,
        "特定金融業の営業貸付金。貸付金で、売上債権ではない",
    ),
    Decision(
        "ReserveForFinancialProductsTransactionLiabilitiesReservesUnderTheSpecialLawsSEC",
        None,
        "証券業の金融商品取引責任準備金。キーワードの誤検出(棚卸資産ではない)",
    ),
    Decision(
        "BorrowingsOnMarginTransactionsCLSEC",
        None,
        "証券業の信用取引借入金。金融業の固有科目",
    ),
    Decision(
        "CashCollateralPledgedForSecuritiesBorrowingOnMarginTransactionsCASEC",
        None,
        "証券業の信用取引借証券担保金。金融業の固有科目",
    ),
    Decision(
        "BorrowingsOnGensakiTransactionCLSEC",
        None,
        "証券業の現先取引借入金。金融業の固有科目",
    ),
    Decision("TradingProductsCASEC", None, "証券業のトレーディング商品。金融業の固有科目"),
)

GROUP_LABELS = {
    R: "売上債権",
    P: "仕入債務",
    INV: "棚卸資産",
    APO: "未払金",
    STL: "短期借入金",
    LTL: "長期借入金",
    BNC: "社債(固定)",
}


def check_complete(candidates: pd.DataFrame) -> None:
    """候補がすべて判定済みで、計画に候補外や重複が無いことを確かめる。"""
    planned = [f"{_P}{d.element}" for d in PLAN]
    dup = [e for e, n in Counter(planned).items() if n > 1]
    if dup:
        raise ValueError(f"duplicated decisions: {dup}")
    cand = set(candidates["element"])
    missing = sorted(cand - set(planned))
    extra = sorted(set(planned) - cand)
    if missing or extra:
        raise ValueError(f"plan does not match candidates: missing={missing} extra={extra}")


def integrated() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in PLAN:
        if d.group is not None:
            out.setdefault(d.group, []).append(f"{_P}{d.element}")
    return out


def apply_to_core(core_path: Path) -> None:
    """core.yaml の各グループの members の末尾に、未登録の要素を追記する(書式は保つ)。"""
    text = core_path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    current = {b["name"]: list(b["members"]) for b in doc["basis"]}
    lines = text.splitlines()
    for group, elements in integrated().items():
        if group not in current:
            raise ValueError(f"unknown group in core.yaml: {group}")
        new = [e for e in elements if e not in current[group]]
        if not new:
            continue
        start = lines.index(f"- name: {group}")
        members_at = next(i for i in range(start, len(lines)) if lines[i] == "  members:")
        end = members_at + 1
        while end < len(lines) and lines[end].startswith("  - "):
            end += 1
        lines[end:end] = [_MARK] + [f"  - {e}" for e in new]
    core_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_core(core_path: Path, originals: dict[str, list[str]]) -> list[str]:
    """core.yaml の members = 統合前の members + 計画の要素 になっているか。"""
    doc = yaml.safe_load(core_path.read_text(encoding="utf-8"))
    members = {b["name"]: list(b["members"]) for b in doc["basis"]}
    problems = []
    plan = integrated()
    for name, now in members.items():
        expected = originals.get(name, []) + [
            e for e in plan.get(name, []) if e not in originals.get(name, [])
        ]
        if now != expected:
            problems.append(f"{name}: core.yaml={now} expected={expected}")
    all_members = [m for ms in members.values() for m in ms]
    dup = [m for m, n in Counter(all_members).items() if n > 1]
    if dup:
        problems.append(f"複数のグループに属する要素: {dup}")
    return problems


def check_subtotals(core_path: Path, leaves_dirs: list[Path], financial: set[str]) -> pd.DataFrame:
    """統合した要素が、各期でグループの親の小計の下にあったか(統合前の集計先で判定)。

    ``n_other_subtotal_nonfinancial`` が 0 であることを合格とする。金融業は流動・固定の
    区分の無い BS があり、推定の対象外なので別に数える。
    """
    doc = yaml.safe_load(core_path.read_text(encoding="utf-8"))
    parent_of = {b["name"]: b["parent_subtotal"] for b in doc["basis"]}
    rows = []
    for d in leaves_dirs:
        leaves = pd.read_csv(
            d / "leaves.csv", usecols=["docID", "element", "bucket"], dtype={"docID": str}
        )
        for group, elements in integrated().items():
            expected = f"__OTHER__{parent_of[group]}"
            sub = leaves[leaves["element"].isin(elements)]
            for element, g in sub.groupby("element"):
                rows.append(
                    {
                        "period": d.name,
                        "element": element,
                        "group": group,
                        "n_docs": g["docID"].nunique(),
                        "n_other_subtotal": int((g["bucket"] != expected).sum()),
                        "n_other_subtotal_nonfinancial": int(
                            ((g["bucket"] != expected) & ~g["docID"].isin(financial)).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_markdown(candidates: pd.DataFrame, path: Path) -> None:
    info = candidates.set_index("element")
    order = {g: i for i, g in enumerate([*GROUP_LABELS, None])}
    decisions = sorted(
        PLAN, key=lambda d: (order[d.group], -int(info.at[f"{_P}{d.element}", "n_docs"]))
    )
    n_in = sum(d.group is not None for d in PLAN)
    lines = [
        "# canonical の統合計画(ステージ21-C)",
        "",
        "__OTHER__(残余科目)に落ちていた標準要素のうち、既存グループのキーワードに該当し、"
        "fy2024 提出の非金融企業で 5 社以上が使う 60 要素について、統合先を判定した。"
        f"統合 {n_in} 要素、統合しない {len(PLAN) - n_in} 要素。"
        "状態ベクトルの次元 D = 32 は変えない(既存グループの members を増やすだけ)。",
        "",
        "社数・総資産比は fy2024 提出(当期)の非金融企業での値。総資産比は、その要素を持つ社での"
        "中央値。",
        "",
        "| 要素(jppfs_cor_) | 社数 | 総資産比 | 統合先 | 理由 |",
        "|---|---:|---:|---|---|",
    ]
    for d in decisions:
        r = info.loc[f"{_P}{d.element}"]
        target = f"{GROUP_LABELS[d.group]}(`{d.group}`)" if d.group else "統合しない"
        lines.append(
            f"| `{d.element}` | {int(r['n_docs'])} | {float(r['median_abs_over_assets']):.3f} "
            f"| {target} | {d.reason} |"
        )
    lines += [
        "",
        "## 対象外としたもの",
        "",
        "- 会社独自の拡張要素(キーワード該当 475 要素、424 社): 要素名は各社が付けるため、"
        "要素名による写像では拾えない。金額規模はステージ21-C A.2 に記録した。",
        "- 5 社未満の標準要素: 判定の対象外(__OTHER__ に残る)。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("results/stage21/contract_assets/leak_candidates.csv"),
    )
    parser.add_argument("--core", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--original-core", type=str, default="HEAD:conf/accounts/core.yaml")
    parser.add_argument("--leaks", type=Path, default=Path("data/interim/stage21/leaks"))
    parser.add_argument("--out", type=Path, default=Path("results/stage21"))
    parser.add_argument(
        "--values",
        type=Path,
        action="append",
        default=[
            Path("data/interim/stage21/fy2024/values_current.csv"),
            Path("data/interim/stage21/2023_2024/values_current.csv"),
        ],
        help="is_financial を引くための values_current.csv",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    check_complete(candidates)
    write_markdown(candidates, args.out / "integration_plan.md")

    original_text = subprocess.run(
        ["git", "show", args.original_core], check=True, capture_output=True, text=True
    ).stdout
    originals = {b["name"]: list(b["members"]) for b in yaml.safe_load(original_text)["basis"]}
    if args.apply:
        apply_to_core(args.core)
    problems = check_core(args.core, originals)
    for p in problems:
        print(f"不一致: {p}")

    leaves_dirs = sorted(p for p in args.leaks.iterdir() if (p / "leaves.csv").exists())
    financial: set[str] = set()
    for values in args.values:
        v = pd.read_csv(values, usecols=["docID", "is_financial"], dtype={"docID": str})
        financial |= set(v.loc[v["is_financial"].astype(str) == "True", "docID"])
    sub = check_subtotals(args.core, leaves_dirs, financial)
    sub.to_csv(args.out / "integration_subtotal_check.csv", index=False)
    bad = sub[sub["n_other_subtotal_nonfinancial"] > 0]
    print(f"判定 {len(PLAN)} 要素(統合 {sum(d.group is not None for d in PLAN)})")
    print(f"core.yaml と計画の不一致: {len(problems)}")
    print(
        f"親の小計が異なる(要素, 期): 非金融 {len(bad)}、"
        f"金融を含めると {int((sub['n_other_subtotal'] > 0).sum())}"
    )
    if not bad.empty:
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
