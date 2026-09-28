#!/usr/bin/env python
"""ステージ30 A.1・A.2: GC 注記の年度間の遷移(2022 → 2023 → 2024)を集計する。

ラベルはステージ4.6 の判定(``scripts/build_labels.py``)を 3 つの書類一覧に当てたもの:
2024 年度分 ``data/interim/labels/labels.csv``、
2022・2023 年度分 ``data/interim/stage30/labels_*.csv``。
特殊業種の除外は ``recovery_stage30.py`` で行う。
年度は決算期末の年(ステージ21〜23 のパネルと同じ)。同じ企業・年度に書類が複数あれば
提出が最新のものを使う。ラベルは class 2 = GC 注記あり、class 1 = 重要事象のみ、class 0 = なし。

出力: ``data/interim/stage30/firm_years.csv``(企業 × 年度のラベル)、
``results/stage30/transitions.md``。

    uv run python scripts/transitions_stage30.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCES = (
    ("data/interim/stage30/labels_2022_2023.csv", "data/raw/index/document_list_2022_2023.csv"),
    ("data/interim/stage30/labels_2023_2024.csv", "data/raw/index/document_list_2023_2024.csv"),
    ("data/interim/labels/labels.csv", "data/raw/index/document_list.csv"),
)
YEARS = (2022, 2023, 2024)
STATUS = {2: "注記あり", 1: "重要事象のみ", 0: "なし"}


def firm_years() -> pd.DataFrame:
    frames = []
    for lab_path, list_path in SOURCES:
        lab = pd.read_csv(lab_path, dtype={"docID": str, "edinetCode": str})
        lst = pd.read_csv(list_path, dtype={"docID": str, "edinetCode": str})
        frames.append(lab.merge(lst[["docID", "periodEnd", "submitDateTime"]], on="docID"))
    df = pd.concat(frames, ignore_index=True)
    df = df[pd.to_numeric(df["label"], errors="coerce").notna()].copy()
    df["label"] = df["label"].astype(int)
    df["year"] = pd.to_datetime(df["periodEnd"]).dt.year
    df = df[df["year"].isin(YEARS)]
    df = df.sort_values("submitDateTime").drop_duplicates(["edinetCode", "year"], keep="last")
    return df[["edinetCode", "filerName", "year", "docID", "periodEnd", "label"]]


def _status(v: float) -> str:
    return "欠測" if pd.isna(v) else STATUS[int(v)]


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude", type=Path, default=Path("data/interim/stage28/excluded.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/stage30"))
    args = ap.parse_args()
    fy = firm_years()
    Path("data/interim/stage30").mkdir(parents=True, exist_ok=True)
    fy.to_csv("data/interim/stage30/firm_years.csv", index=False)
    wide = fy.pivot(index="edinetCode", columns="year", values="label")
    names = fy.sort_values("year").drop_duplicates("edinetCode", keep="last").set_index(
        "edinetCode"
    )["filerName"]
    ex = pd.read_csv(args.exclude, dtype={"edinetCode": str})
    ex_codes = set(ex["edinetCode"])

    lines = ["# ステージ30 A.1・A.2: GC 注記の遷移(2022 → 2024)", ""]
    lines += ["## 年度ごとのラベル(企業数)", "", "| 年度 | 注記あり | 重要事象のみ | なし | 計 |",
              "|---|---:|---:|---:|---:|"]
    for y in YEARS:
        c = wide[y].value_counts()
        lines.append(f"| {y} | {int(c.get(2, 0))} | {int(c.get(1, 0))} | {int(c.get(0, 0))} "
                     f"| {int(wide[y].notna().sum())} |")

    for title, codes in (("全企業", wide.index),
                         ("ステージ28 の除外リスト(2024 年度の評価データ)を除く",
                          wide.index.difference(list(ex_codes)))):
        w = wide.loc[codes]
        lines += ["", f"## 2022 → 2024 の遷移({title})", "",
                  "| 2022 | 2024: 注記あり | 重要事象のみ | なし | 欠測 | 計 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for s in (2, 1, 0):
            base = w[w[2022] == s]
            to = base[2024].map(_status).value_counts()
            lines.append(
                f"| {STATUS[s]} | {int(to.get('注記あり', 0))} | {int(to.get('重要事象のみ', 0))} "
                f"| {int(to.get('なし', 0))} | {int(to.get('欠測', 0))} | {len(base)} |"
            )
        gc22 = w[w[2022] == 2]
        path = gc22[2023].map(_status) + " → " + gc22[2024].map(_status)
        lines += ["", f"2022 年に注記ありの {len(gc22)} 社の 2023・2024 年の状態:", "",
                  "| 2023 → 2024 | 社数 |", "|---|---:|"]
        for k, v in path.value_counts().items():
            lines.append(f"| {k} | {v} |")

    # 再建(2022 注記あり → 2024 注記なし・重要事象のみ)と非再建の一覧
    w = wide
    gc22 = w[w[2022] == 2]
    rec = gc22[gc22[2024].isin([0, 1])]
    lines += ["", "## 2022 注記あり → 2024 注記が外れた企業", "",
              "| edinetCode | 企業 | 2023 | 2024 |", "|---|---|---|---|"]
    for code, r in rec.iterrows():
        lines.append(
            f"| {code} | {names.get(code, '')} | {_status(r[2023])} | {_status(r[2024])} |"
        )
    # 往復(2022 あり → 2023 外れ → 2024 あり)
    rt = gc22[gc22[2023].isin([0, 1]) & (gc22[2024] == 2)]
    lines += ["", f"往復(2022 あり → 2023 外れ → 2024 あり): {len(rt)} 社"
              + (f"({', '.join(names.get(c, c) for c in rt.index)})" if len(rt) else "")]
    # 中間年に初めて外れた社(2023 で外れ、2024 は欠測)
    mid = gc22[gc22[2023].isin([0, 1]) & gc22[2024].isna()]
    lines.append(f"2023 で外れ、2024 は欠測: {len(mid)} 社")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transitions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
