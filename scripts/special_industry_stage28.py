#!/usr/bin/env python
"""ステージ28 A: 会計構造が異なる業種の書類を機械的に判定し、除外リストを作る。

入力は ``scripts/extract_business_stage28.py`` の出力(事業の内容と 33 業種)。規則は生成結果を
見る前に、次のとおり定めた。

- ゴルフ場(会員制ゴルフ場の経営・保有): 33 業種がサービス業または不動産業で、かつ
  「事業の内容」の最初の文(主たる事業を述べる文)に「ゴルフ」を含むか、提出者名に「ゴルフ」
  「カントリー」「カンツリー」を含む。最初の文の「カントリー」「カンツリー」は数えない
  (子会社名の列挙に現れる。最初の版の規則では、ゴルフ場子会社を持つだけの企業集団
  (燦キャピタルマネージメント)が該当した。生成の前に修正した)。
- 金融: 33 業種が銀行業・証券、商品先物取引業・保険業・その他金融業(リース・信販・貸金を含む)。
  既存の ``is_financial``(未対応の資産の比率による判定)で除外済みの書類は、入力に含まれない。

出力: ``data/interim/stage28/excluded.csv``(生成・評価のスクリプトが ``--exclude`` で読む)、
``results/stage28/excluded_docs.csv``、``results/stage28/exclusion.md``。

    uv run python scripts/special_industry_stage28.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

GOLF_INDUSTRIES = ("サービス業", "不動産業")
GOLF_NAME_WORDS = re.compile("ゴルフ|カントリー|カンツリー")
GOLF_TEXT_WORDS = re.compile("ゴルフ")
FINANCIAL_INDUSTRIES = ("銀行業", "証券、商品先物取引業", "保険業", "その他金融業")
_OTHER_NCL = "__OTHER__jppfs_cor_NoncurrentLiabilities"


def first_sentence(text: str) -> str:
    """見出し「事業の内容】」の後の、最初の「。」までの文。"""
    body = re.split(r"事業の内容\s*】", text, maxsplit=1)[-1]
    return body.split("。")[0].strip()


def classify(df: pd.DataFrame) -> pd.Series:
    """書類ごとの除外理由("golf" / "financial" / "")。"""
    first = df["business"].fillna("").map(first_sentence)
    golf = df["industry33"].isin(GOLF_INDUSTRIES) & (
        first.str.contains(GOLF_TEXT_WORDS) | df["filerName"].str.contains(GOLF_NAME_WORDS)
    )
    fin = df["industry33"].isin(FINANCIAL_INDUSTRIES)
    reason = pd.Series("", index=df.index)
    reason[golf] = "golf"
    reason[fin] = "financial"
    return reason


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--business", type=Path, default=Path("data/interim/stage28/business.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--results", type=Path, default=Path("results/stage28"))
    args = ap.parse_args()

    df = pd.read_csv(args.business, dtype={"docID": str, "edinetCode": str})
    df["reason"] = classify(df)
    df["first_sentence"] = df["business"].fillna("").map(first_sentence).str.slice(0, 120)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.set_index("docID")
    df["other_ncl_over_assets"] = (
        agg.loc[df["docID"], _OTHER_NCL].fillna(0.0).to_numpy()
        / agg.loc[df["docID"], "total_assets"].to_numpy()
    )
    ex = df[df["reason"] != ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ex[["docID", "edinetCode", "filerName", "label", "reason"]].to_csv(args.out, index=False)
    args.results.mkdir(parents=True, exist_ok=True)
    cols = ["docID", "edinetCode", "filerName", "label", "industry33", "reason",
            "other_ncl_over_assets", "first_sentence"]
    ex[cols].sort_values(["reason", "label", "filerName"]).to_csv(
        args.results / "excluded_docs.csv", index=False
    )

    lines = ["# ステージ28 A.1: 除外の該当数", ""]
    lines += ["| 理由 | class 2(68 社中) | class 0 の書類(3,951 中) | class 0 の企業 |",
              "|---|---:|---:|---:|"]
    for r in ("golf", "financial"):
        s = ex[ex["reason"] == r]
        lines.append(
            f"| {r} | {int((s['label'] == 2).sum())} | {int((s['label'] == 0).sum())} "
            f"| {s[s['label'] == 0]['edinetCode'].nunique()} |"
        )
    lines.append(
        f"| 計 | {int((ex['label'] == 2).sum())} | {int((ex['label'] == 0).sum())} "
        f"| {ex[ex['label'] == 0]['edinetCode'].nunique()} |"
    )
    lines += [
        "", "## class 2 の該当", "",
        "| docID | 企業 | 33 業種 | 理由 | その他固定負債 / 総資産 |", "|---|---|---|---|---:|",
    ]
    for _, r in ex[ex["label"] == 2].sort_values(["reason", "docID"]).iterrows():
        lines.append(
            f"| {r['docID']} | {r['filerName']} | {r['industry33']} | {r['reason']} "
            f"| {r['other_ncl_over_assets']:.2f} |"
        )
    lines += ["", "## その他固定負債(預託金を含む残余)/ 総資産の分布(class 0)", "",
              "| 群 | 書類 | p25 | p50 | p75 | 0.3 超の割合 |", "|---|---:|---:|---:|---:|---:|"]
    c0 = df[df["label"] == 0]
    for name, g in (("ゴルフ場", c0[c0["reason"] == "golf"]),
                    ("金融", c0[c0["reason"] == "financial"]),
                    ("それ以外", c0[c0["reason"] == ""])):
        v = g["other_ncl_over_assets"]
        lines.append(
            f"| {name} | {len(g)} | {v.quantile(0.25):.3f} | {v.median():.3f} "
            f"| {v.quantile(0.75):.3f} | {(v > 0.3).mean():.3f} |"
        )
    (args.results / "exclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
