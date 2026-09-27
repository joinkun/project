#!/usr/bin/env python
"""ステージ28 A.1: 有価証券報告書の「事業の内容」と 33 業種を、評価対象の書類ごとに取り出す。

対象は輸送に使う書類(`precompute_stage13_transport.load_data` と同じ条件: 解析成功、
金融業でない、ラベルあり)の class 0 と class 2。XBRL(`data/raw/documents*/<docID>.zip`)の
`jpcrp_cor:DescriptionOfBusinessTextBlock` をタグを除いた文字列にする。data/raw は読むだけ。

    uv run python scripts/extract_business_stage28.py
"""

from __future__ import annotations

import argparse
import html
import re
import zipfile
from pathlib import Path

import pandas as pd

from fsgen.models.industry import load_industry_config

_RAW_DIRS = (
    Path("data/raw/documents"),
    Path("data/raw/documents_2022_2023"),
    Path("data/raw/documents_2023_2024"),
)
_TAG = re.compile(
    r"<jpcrp_cor:DescriptionOfBusinessTextBlock[^>]*>(.*?)</jpcrp_cor:DescriptionOfBusinessTextBlock>",
    re.S,
)


def business_text(doc: str) -> str | None:
    """事業の内容の文字列。zip が無いか、要素が無ければ None。"""
    for d in _RAW_DIRS:
        path = d / f"{doc}.zip"
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if "PublicDoc" in n and n.endswith(".xbrl")]
            for n in names:
                m = _TAG.search(z.read(n).decode("utf-8"))
                if m:
                    text = re.sub(r"<[^>]+>", " ", html.unescape(html.unescape(m.group(1))))
                    return re.sub(r"\s+", " ", text).strip()
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage28/business.csv"))
    args = ap.parse_args()
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg[["docID", "edinetCode"]].merge(
        lab[["docID", "filerName", "label"]], on="docID", how="inner"
    )
    m["label"] = m["label"].astype(int)
    m = m[m["label"].isin([0, 2])].reset_index(drop=True)
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    m["industry33"] = ind.labels(m["edinetCode"].astype(str).tolist())
    m["business"] = [business_text(d) for d in m["docID"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)
    print(m["label"].value_counts().to_string())
    print("事業の内容が取れない書類:", int(m["business"].isna().sum()))


if __name__ == "__main__":
    main()
