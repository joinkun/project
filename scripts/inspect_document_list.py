#!/usr/bin/env python
"""document_list.csv の取得前チェック(API は叩かない)。

使い方:
    uv run python scripts/inspect_document_list.py data/raw/index/document_list.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_SPECIAL_ENTITY_KEYWORDS = ("投資法人", "REIT", "特定目的会社")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    return df


def print_monthly_distribution(df: pd.DataFrame, column: str, label: str) -> None:
    print(f"\n=== {label}({column})の月別分布 ===")
    parsed = pd.to_datetime(df[column], errors="coerce")
    month = parsed.dt.to_period("M")
    counts = month.value_counts().sort_index()
    n_unparsed = parsed.isna().sum()
    for period, count in counts.items():
        print(f"  {period}: {count}")
    if n_unparsed:
        print(f"  (パース不能/欠損: {n_unparsed}件)")


def print_null_counts(df: pd.DataFrame, column: str) -> None:
    print(f"\n=== {column} が null / 非null の件数 ===")
    n_null = df[column].isna().sum()
    n_non_null = df[column].notna().sum()
    print(f"  null    : {n_null}")
    print(f"  非null  : {n_non_null}")


def print_value_counts(df: pd.DataFrame, column: str) -> None:
    print(f"\n=== {column} の値ごとの件数 ===")
    counts = df[column].value_counts(dropna=False)
    for value, count in counts.items():
        print(f"  {value!r}: {count}")


def print_duplicate_edinet_codes(df: pd.DataFrame) -> None:
    print("\n=== 同一 edinetCode が複数行ある社の一覧 ===")
    counts = df["edinetCode"].value_counts()
    duplicated_codes = counts[counts > 1].index
    print(f"  該当 edinetCode 数: {len(duplicated_codes)}")
    subset = df[df["edinetCode"].isin(duplicated_codes)].sort_values(
        ["edinetCode", "periodEnd"]
    )
    columns = ["edinetCode", "filerName", "docID", "periodEnd", "submitDateTime"]
    for _, row in subset[columns].iterrows():
        print(
            f"  {row['edinetCode']} {row['filerName']!s:<20} {row['docID']} "
            f"periodEnd={row['periodEnd']} submitDateTime={row['submitDateTime']}"
        )


def print_special_entity_counts(df: pd.DataFrame) -> None:
    print("\n=== filerName に投資法人/REIT/特定目的会社を含む件数 ===")
    pattern = "|".join(_SPECIAL_ENTITY_KEYWORDS)
    mask = df["filerName"].fillna("").str.contains(pattern, regex=True)
    print(f"  該当件数: {mask.sum()} / 全{len(df)}件")
    for keyword in _SPECIAL_ENTITY_KEYWORDS:
        n = df["filerName"].fillna("").str.contains(keyword, regex=False).sum()
        print(f"    うち「{keyword}」を含む: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "document_list_csv",
        type=Path,
        nargs="?",
        default=Path("data/raw/index/document_list.csv"),
    )
    args = parser.parse_args()

    df = load(args.document_list_csv)
    print(f"document_list.csv: {len(df)}行")

    print_monthly_distribution(df, "periodEnd", "periodEnd")
    print_null_counts(df, "secCode")
    print_value_counts(df, "docInfoEditStatus")
    print_monthly_distribution(df, "submitDateTime", "submitDateTime")
    print_duplicate_edinet_codes(df)
    print_special_entity_counts(df)


if __name__ == "__main__":
    main()
