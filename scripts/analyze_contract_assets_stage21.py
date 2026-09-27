#!/usr/bin/env python
"""ステージ21-B C.3: 契約資産・電子記録債権など、グループ科目の取りこぼしの影響を測る。

``survey_group_leaks_stage21.py`` の ``leaves.csv``(4 期分)と、
``verify_prior_stage21.py`` の ``values_current.csv``(fy2024)を使う。対象は fy2024 の
非金融・解析成功の社(``aggregated_core.csv`` の is_financial で判定)。

- (2) 排他性スコア P(A∩B) / min(P(A), P(B))(同一社・同一期で両方が非ゼロ)
- (4) 金額規模: 候補要素の和 / __OTHER__CurrentAssets、/ 総資産(候補を持つ社の中央値)
- 表示の切り替え: 同じ社で、候補要素の有無が期の間で変わる件数

    uv run python scripts/analyze_contract_assets_stage21.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from survey_group_leaks_stage21 import keyword_group  # noqa: E402

_P = "jppfs_cor_"

# 現行の receivables_trade の members
RECEIVABLES_MEMBERS: tuple[str, ...] = (
    f"{_P}AccountsReceivableTrade",
    f"{_P}NotesAndAccountsReceivableTrade",
    f"{_P}NotesAndAccountsReceivableTradeAndContractAssets",
    f"{_P}NotesReceivableTrade",
)
CONTRACT_ASSETS: tuple[str, ...] = (
    f"{_P}ContractAssets",
    f"{_P}AccountsReceivableTradeAndContractAssets",
)
ELECTRONIC_CLAIMS = f"{_P}ElectronicallyRecordedMonetaryClaimsOperatingCA"

# 金額規模を測る候補の束(統合を検討する要素)
CANDIDATE_SETS: dict[str, tuple[str, ...]] = {
    "contract_assets": CONTRACT_ASSETS,
    "electronic_claims": (ELECTRONIC_CLAIMS,),
    "construction_receivables": (
        f"{_P}NotesReceivableAccountsReceivableFromCompletedConstructionContractsAndOtherCNS",
        f"{_P}AccountsReceivableFromCompletedConstructionContractsCNS",
    ),
    "operating_receivables": (
        f"{_P}OperatingAccountsReceivableCA",
        f"{_P}NotesAndOperatingAccountsReceivableCA",
    ),
}

# (期の名前, leaves.csv のディレクトリ, 提出期間の接尾辞)。期の名前は決算年。
PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2024", "fy2024_current", ""),
    ("2023a", "fy2024_prior", ""),
    ("2023b", "2023_2024_current", "_2023_2024"),
    ("2022", "2023_2024_prior", "_2023_2024"),
)


def exclusivity(presence: dict[str, set[str]], n: int, a: str, b: str) -> dict[str, object]:
    pa, pb = presence.get(a, set()), presence.get(b, set())
    both = len(pa & pb)
    score = both / min(len(pa), len(pb)) if pa and pb else float("nan")
    return {
        "a": a.removeprefix(_P),
        "b": b.removeprefix(_P),
        "n_a": len(pa),
        "n_b": len(pb),
        "n_both": both,
        "p_a": len(pa) / n,
        "p_b": len(pb) / n,
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaks", type=Path, default=Path("data/interim/stage21/leaks"))
    parser.add_argument(
        "--aggregated", type=Path, default=Path("data/interim/values/aggregated_core.csv")
    )
    parser.add_argument(
        "--values", type=Path, default=Path("data/interim/stage21/fy2024/values_current.csv")
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/stage21/contract_assets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    agg = pd.read_csv(args.aggregated, dtype={"docID": str, "edinetCode": str})
    nf = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    nf_codes = set(nf["edinetCode"])
    doc_to_code: dict[str, str] = {}
    for _, _, suffix in PERIODS:
        with (args.raw_dir / "index" / f"document_list{suffix}.csv").open(encoding="utf-8") as f:
            doc_to_code.update({r["docID"]: r["edinetCode"] for r in csv.DictReader(f)})

    lines = ["# ステージ21-B C.3: 契約資産ほか、グループ科目の取りこぼし", ""]

    # (2) 排他性(fy2024 当期、非金融)
    leaves = pd.read_csv(args.leaks / "fy2024_current" / "leaves.csv", dtype={"docID": str})
    leaves = leaves[leaves["docID"].isin(set(nf["docID"]))]
    presence: dict[str, set[str]] = defaultdict(set)
    for doc_id, element in zip(leaves["docID"], leaves["element"], strict=True):
        presence[element].add(doc_id)
    n = len(nf)
    pairs = [
        (c, r) for c in (*CONTRACT_ASSETS, ELECTRONIC_CLAIMS) for r in RECEIVABLES_MEMBERS
    ] + list(combinations((*CONTRACT_ASSETS, ELECTRONIC_CLAIMS), 2))
    rows = [exclusivity(presence, n, a, b) for a, b in pairs]
    pd.DataFrame(rows).to_csv(args.out / "exclusivity.csv", index=False)
    lines += [
        f"## (2) 排他性スコア(fy2024 当期、非金融 {n} 社)",
        "",
        "| A | B | A の社数 | B の社数 | 併存 | スコア |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['a']} | {r['b']} | {r['n_a']} | {r['n_b']} | {r['n_both']} "
            f"| {float(str(r['score'])):.3f} |"
        )
    n_ca_without_ar = len(
        presence[f"{_P}ContractAssets"] - set().union(*(presence[m] for m in RECEIVABLES_MEMBERS))
    )
    lines += [
        "",
        f"ContractAssets を持つが、現行の売上債権 members を 1 つも持たない社: {n_ca_without_ar}",
    ]

    # (4) 金額規模(fy2024 当期、非金融)
    values = pd.read_csv(args.values, dtype={"docID": str}).set_index("docID")
    asset_cols = [c for c in values.columns if c.startswith("total_assets")]
    other_ca = values["__OTHER__jppfs_cor_CurrentAssets"].fillna(0)
    total_assets = values[asset_cols[0]]
    lines += [
        "",
        "## (4) 金額規模(fy2024 当期、候補を持つ非金融の社)",
        "",
        "| 候補 | 社数 | 候補 / __OTHER__CurrentAssets 中央値 | 候補 / 総資産 中央値 "
        "| 候補 / 総資産 p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    scale_rows = []
    for name, elements in CANDIDATE_SETS.items():
        amount = leaves[leaves["element"].isin(elements)].groupby("docID")["value"].sum()
        amount = amount[amount != 0]
        idx = amount.index.intersection(values.index)
        ratio_other = (amount[idx] / other_ca[idx].where(other_ca[idx] != 0)).dropna()
        ratio_assets = (amount[idx] / total_assets[idx]).dropna()
        scale_rows.append(
            {
                "candidate": name,
                "n": len(idx),
                "over_other_ca_median": ratio_other.median(),
                "over_assets_median": ratio_assets.median(),
                "over_assets_p90": ratio_assets.quantile(0.9),
            }
        )
        lines.append(
            f"| {name} | {len(idx)} | {ratio_other.median():.3f} | {ratio_assets.median():.4f} "
            f"| {ratio_assets.quantile(0.9):.4f} |"
        )
    pd.DataFrame(scale_rows).to_csv(args.out / "scale.csv", index=False)

    # 表示の切り替え: 社 × 期ごとに、候補要素・現行 members の有無
    tracked = (*RECEIVABLES_MEMBERS, *CONTRACT_ASSETS, ELECTRONIC_CLAIMS)
    has: dict[tuple[str, str], set[str]] = defaultdict(set)
    present_codes: dict[str, set[str]] = defaultdict(set)
    for period, dirname, _ in PERIODS:
        df = pd.read_csv(args.leaks / dirname / "leaves.csv", dtype={"docID": str})
        df["edinetCode"] = df["docID"].map(doc_to_code)
        df = df[df["edinetCode"].isin(nf_codes)]
        present_codes[period] = set(df["edinetCode"])
        for code, element in zip(df["edinetCode"], df["element"], strict=True):
            if element in tracked:
                has[(code, period)].add(element)
    lines += [
        "",
        "## 表示の切り替え(同じ社で、要素の有無が期の間で変わる件数)",
        "",
        "期: 2024 = fy2024 当期、2023a = fy2024 前期、2023b = 2023-24 当期、2022 = 2023-24 前期。",
        "",
        "| 要素 | 比較 | 両期にいる社 | 有→無 | 無→有 |",
        "|---|---|---:|---:|---:|",
    ]
    switch_rows = []
    for element in tracked:
        for old, new in (("2022", "2023b"), ("2023a", "2024"), ("2023b", "2023a")):
            codes = present_codes[old] & present_codes[new]
            lost = sum(
                1 for c in codes if element in has[(c, old)] and element not in has[(c, new)]
            )
            gained = sum(
                1 for c in codes if element not in has[(c, old)] and element in has[(c, new)]
            )
            switch_rows.append(
                {
                    "element": element,
                    "old": old,
                    "new": new,
                    "n": len(codes),
                    "lost": lost,
                    "gained": gained,
                }
            )
            lines.append(
                f"| {element.removeprefix(_P)} | {old}→{new} | {len(codes)} | {lost} | {gained} |"
            )
    pd.DataFrame(switch_rows).to_csv(args.out / "switching.csv", index=False)

    # (1) 取りこぼし候補の一覧: __OTHER__ に落ちる標準要素のうち、既存グループに相当しうるもの
    other = leaves[leaves["bucket"].str.startswith("__OTHER__")].copy()
    other["keyword_group"] = other["element"].map(keyword_group)
    other.loc[other["element"].str.startswith(f"{_P}ElectronicallyRecorded"), "keyword_group"] = (
        "electronically_recorded"
    )
    other["rel"] = other["value"].abs() / other["assets"].abs()
    flagged = other[(other["keyword_group"] != "") & other["element"].str.startswith(_P)]
    table = (
        flagged.groupby("element")
        .agg(
            keyword_group=("keyword_group", "first"),
            n_docs=("docID", "nunique"),
            bucket=("bucket", lambda x: x.mode().iloc[0]),
            parent=("parent", lambda x: x.mode().iloc[0]),
            share_negative=("value", lambda x: float((x < 0).mean())),
            median_abs_over_assets=("rel", "median"),
        )
        .sort_values("n_docs", ascending=False)
    )
    table.insert(2, "share", table["n_docs"] / n)
    table[table["n_docs"] >= 5].to_csv(args.out / "leak_candidates.csv")
    n_ext = other[(other["keyword_group"] != "") & ~other["element"].str.startswith(_P)]
    lines += [
        "",
        f"## (1) 取りこぼし候補(標準要素、5 社以上): {int((table['n_docs'] >= 5).sum())} 要素"
        " → leak_candidates.csv",
        f"拡張要素でキーワードに当たるもの: {n_ext['element'].nunique()} 要素"
        f"({n_ext['docID'].nunique()} 社)",
    ]

    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
