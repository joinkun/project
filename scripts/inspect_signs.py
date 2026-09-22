#!/usr/bin/env python
"""canonical の各科目(基礎科目・小計科目)の符号分布を実測する(ステージ5.5 §E)。

`core.yaml` の `sign` フィールドを `free` から実際の値(`nonneg`/`nonpos`/
`free`)に確定させるための調査。全社(非金融)について集計し、加えて
class 2(GC注記あり、68社)だけの別集計も出す(債務超過企業が含まれる
可能性が高く、「純資産合計を非負にすると実データが破綻する」ことの
確認になる)。

判定は機械的に決めず、報告(特に「負が数社だけ」のケースの明細)を見て
人間が判断する。

使い方:
    uv run python scripts/inspect_signs.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from fsgen.accounting.constraints import load_canonical_spec

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")

# 負(または正)が「数社だけ」とみなして明細を出す閾値。
_FEW_THRESHOLD = 20

_SPECIAL_INTEREST = (
    "jppfs_cor_RetainedEarnings",
    "jppfs_cor_NetAssets",
    "jppfs_cor_TreasuryStock",
    "jppfs_cor_ValuationAndTranslationAdjustments",
)


def load_labels(labels_csv: Path) -> tuple[dict[str, int], dict[str, str]]:
    label_by_doc: dict[str, int] = {}
    filer_by_doc: dict[str, str] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] != "":
                label_by_doc[row["docID"]] = int(row["label"])
            filer_by_doc[row["docID"]] = row.get("filerName", "")
    return label_by_doc, filer_by_doc


def load_rows(
    aggregated_csv: Path, labels_csv: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(全社(非金融), class2(非金融))の行リストを返す。filerName は labels.csv
    から docID で結合する(aggregated_core.csv には無いため)。"""
    label_by_doc, filer_by_doc = load_labels(labels_csv)
    all_rows: list[dict[str, str]] = []
    class2_rows: list[dict[str, str]] = []
    with aggregated_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["parse_status"] != "ok" or row.get("is_financial") == "True":
                continue
            row["filerName"] = filer_by_doc.get(row["docID"], "")
            all_rows.append(row)
            if label_by_doc.get(row["docID"]) == 2:
                class2_rows.append(row)
    return all_rows, class2_rows


def summarize_column(
    rows: list[dict[str, str]], column: str
) -> tuple[int, int, int, float | None, float | None, float | None]:
    """(n_positive, n_zero, n_negative, min, median, max) を返す(欠損は除く)。"""
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        if raw == "":
            continue
        values.append(float(raw))
    n_pos = sum(1 for v in values if v > 0)
    n_zero = sum(1 for v in values if v == 0)
    n_neg = sum(1 for v in values if v < 0)
    if not values:
        return n_pos, n_zero, n_neg, None, None, None
    return n_pos, n_zero, n_neg, min(values), statistics.median(values), max(values)


def suggest_sign(n_pos: int, n_zero: int, n_neg: int) -> str:
    if n_neg == 0:
        return "nonneg候補"
    if n_pos == 0:
        return "nonpos候補"
    return "free"


def print_detail(rows: list[dict[str, str]], column: str, want_negative: bool) -> None:
    matches = []
    for row in rows:
        raw = row.get(column, "")
        if raw == "":
            continue
        v = float(raw)
        if (want_negative and v < 0) or (not want_negative and v > 0):
            matches.append((row["docID"], row.get("filerName") or row.get("edinetCode", ""), v))
    for doc_id, name, v in sorted(matches, key=lambda t: t[2]):
        print(f"      {doc_id} {name}: {v:.0f}")


def print_table(rows: list[dict[str, str]], columns: list[str], title: str) -> None:
    print(f"\n=== {title}({len(rows)}社) ===")
    header = (
        f"{'name':<55} {'n_pos':>7} {'n_zero':>7} {'n_neg':>7} "
        f"{'min':>16} {'median':>16} {'max':>16}  判定"
    )
    print(header)
    for column in columns:
        n_pos, n_zero, n_neg, vmin, vmed, vmax = summarize_column(rows, column)
        suggestion = suggest_sign(n_pos, n_zero, n_neg)
        min_s = f"{vmin:.0f}" if vmin is not None else ""
        med_s = f"{vmed:.0f}" if vmed is not None else ""
        max_s = f"{vmax:.0f}" if vmax is not None else ""
        print(
            f"{column:<55} {n_pos:>7} {n_zero:>7} {n_neg:>7} {min_s:>16} {med_s:>16} "
            f"{max_s:>16}  {suggestion}"
        )
        few_negative = 0 < n_neg <= _FEW_THRESHOLD
        few_positive = 0 < n_pos <= _FEW_THRESHOLD and n_neg > n_pos
        if few_negative:
            print(f"    [負が少数({n_neg}社)の明細]")
            print_detail(rows, column, want_negative=True)
        if few_positive:
            print(f"    [正が少数({n_pos}社)の明細]")
            print_detail(rows, column, want_negative=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    basis_columns = [b.name for b in spec.basis]
    subtotal_columns = [s.element_id for s in spec.subtotals]
    all_columns = basis_columns + subtotal_columns

    all_rows, class2_rows = load_rows(args.aggregated_csv, args.labels_csv)
    print(f"全社(非金融): {len(all_rows)}社, class2(非金融): {len(class2_rows)}社")

    print_table(all_rows, all_columns, "全社(非金融)の符号分布")
    print_table(class2_rows, all_columns, "class2(GC注記あり、非金融)だけの符号分布")

    print("\n=== 特に注目する科目(全社) ===")
    for column in _SPECIAL_INTEREST:
        n_pos, n_zero, n_neg, vmin, vmed, vmax = summarize_column(all_rows, column)
        print(
            f"  {column}: n_pos={n_pos} n_zero={n_zero} n_neg={n_neg} "
            f"min={vmin} median={vmed} max={vmax}"
        )


if __name__ == "__main__":
    main()
