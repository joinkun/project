#!/usr/bin/env python
"""ステージ21-C A: canonical の統合前の確認(fy2024 当期、非金融)。

- A.1 新株予約権付社債: 要素・社数・総資産比・__OTHER__NoncurrentLiabilities に占める割合
- A.2 会社独自の拡張要素(キーワード該当)の金額規模
- A.3 パネルで 1 社にまとめた重複書類(同じ edinetCode の複数書類)とそのラベル

入力は統合前の出力(``leaves.csv``、``values_current.csv``、``aggregated_core.csv``、
``labels.csv``)。

    uv run python scripts/pre_integration_checks_stage21.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from survey_group_leaks_stage21 import keyword_group  # noqa: E402

_P = "jppfs_cor_"


def _q(s: pd.Series, q: float) -> str:
    return "-" if s.empty else f"{s.quantile(q):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaves", type=Path, default=Path("data/interim/stage21/leaks/fy2024_current/leaves.csv")
    )
    parser.add_argument(
        "--values", type=Path, default=Path("data/interim/stage21/fy2024/values_current.csv")
    )
    parser.add_argument(
        "--aggregated", type=Path, default=Path("data/interim/values/aggregated_core.csv")
    )
    parser.add_argument("--labels", type=Path, default=Path("data/interim/labels/labels.csv"))
    parser.add_argument(
        "--document-list", type=Path, default=Path("data/raw/index/document_list.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("results/stage21/integration_checks.md"))
    args = parser.parse_args()

    agg = pd.read_csv(args.aggregated, dtype={"docID": str, "edinetCode": str})
    nf = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    nf_docs = set(nf["docID"])
    leaves = pd.read_csv(args.leaves, dtype={"docID": str})
    leaves = leaves[leaves["docID"].isin(nf_docs)]
    values = pd.read_csv(args.values, dtype={"docID": str}).set_index("docID")
    lines = [f"# ステージ21-C A: 統合前の確認(fy2024 当期、非金融 {len(nf)} 社)", ""]

    # A.1 新株予約権付社債
    other = leaves[leaves["bucket"].str.startswith("__OTHER__")]
    is_cb = other["element"].str.contains("SubscriptionRightsToShares") & other[
        "element"
    ].str.contains("Bond")
    cb = other[is_cb]
    other_ncl = values["__OTHER__jppfs_cor_NoncurrentLiabilities"]
    lines += [
        "## A.1 新株予約権付社債",
        "",
        "| 要素 | 集計先 | 社数 | 総資産比 中央値 | p90 | __OTHER__NCL に占める割合 中央値 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (element, bucket), g in cb.groupby(["element", "bucket"]):
        rel = g["value"] / g["assets"]
        share = (
            g.set_index("docID")["value"]
            / other_ncl.reindex(g["docID"]).where(lambda s: s != 0).to_numpy()
        ).dropna()
        lines.append(
            f"| {element} | {bucket.removeprefix('__OTHER__')} | {g['docID'].nunique()} "
            f"| {_q(rel, 0.5)} | {_q(rel, 0.9)} "
            f"| {_q(share, 0.5) if bucket.endswith('NoncurrentLiabilities') else '-'} |"
        )
    lines.append(
        f"\n新株予約権付社債を持つ社: {cb['docID'].nunique()} / {len(nf)}"
        f"({cb['docID'].nunique() / len(nf):.2%})"
    )

    # A.2 拡張要素(キーワード該当)
    ext = other[~other["element"].str.startswith(_P)].copy()
    ext = ext[ext["element"].map(keyword_group) != ""]
    per_doc = ext.groupby("docID")["value"].apply(lambda s: s.abs().sum())
    buckets_of = ext.groupby("docID")["bucket"].apply(lambda s: sorted(set(s)))
    denom = pd.Series(
        {
            d: sum(abs(float(values.at[d, b])) for b in bs if pd.notna(values.at[d, b]))
            for d, bs in buckets_of.items()
        }
    )
    ratio_other = (per_doc / denom.where(denom != 0)).dropna()
    ratio_assets = per_doc / values["total_assets"].reindex(per_doc.index).abs()
    lines += [
        "",
        "## A.2 会社独自の拡張要素(既存グループのキーワードに該当するもの)",
        "",
        f"- 要素の種類: {ext['element'].nunique()}、該当社: {per_doc.size}",
        "- 分母の「__OTHER__ の合計」は、その社の拡張要素が落ちている __OTHER__ 科目の"
        "最終値(差額定義)の絶対値の和",
        f"- 拡張要素の合計 / __OTHER__ の合計: 中央値 {_q(ratio_other, 0.5)}、"
        f"p90 {_q(ratio_other, 0.9)}",
        f"- 拡張要素の合計 / 総資産: 中央値 {_q(ratio_assets, 0.5)}、p90 {_q(ratio_assets, 0.9)}"
        f"、最大 {_q(ratio_assets, 1.0)}",
        f"- 総資産比 5% 超の社: {(ratio_assets > 0.05).sum()}、"
        f"10% 超: {(ratio_assets > 0.1).sum()}",
    ]
    by_group = ext.assign(group=ext["element"].map(keyword_group)).groupby("group")
    lines += ["", "| キーワード | 要素数 | 社数 |", "|---|---:|---:|"]
    for group, g in by_group:
        lines.append(f"| {group} | {g['element'].nunique()} | {g['docID'].nunique()} |")

    # A.3 重複書類
    labels = pd.read_csv(args.labels, dtype={"docID": str, "edinetCode": str})
    doc_list = pd.read_csv(args.document_list, dtype={"docID": str, "edinetCode": str})
    nf_lab = (
        nf[["docID", "edinetCode"]]
        .merge(labels[["docID", "label"]], on="docID", how="left")
        .merge(doc_list[["docID", "filerName", "periodStart", "periodEnd", "submitDateTime"]])
    )
    dup = nf_lab[nf_lab.duplicated("edinetCode", keep=False)].sort_values(
        ["edinetCode", "periodEnd"]
    )
    kept = dup.groupby("edinetCode")["periodEnd"].transform("max") == dup["periodEnd"]
    lines += [
        "",
        "## A.3 同じ edinetCode の複数書類(パネルでは当期末が最も新しい書類を残す)",
        "",
        f"- 該当 {dup['edinetCode'].nunique()} 社・{len(dup)} 書類。"
        f"除いた書類 {int((~kept).sum())}",
        f"- 除いた書類に class 2 が含まれるか: {bool((dup[~kept]['label'] == 2).any())}",
        "",
        "| edinetCode | docID | 社名 | 期間 | 提出日時 | class | パネル |",
        "|---|---|---|---|---|---:|---|",
    ]
    for (_, r), keep in zip(dup.iterrows(), kept, strict=True):
        lines.append(
            f"| {r['edinetCode']} | {r['docID']} | {r['filerName']} "
            f"| {r['periodStart']}〜{r['periodEnd']} | {r['submitDateTime']} "
            f"| {r['label']} | {'残す' if keep else '除く'} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
