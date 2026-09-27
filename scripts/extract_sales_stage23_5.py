#!/usr/bin/env python
"""ステージ23.5 B.1: 売上高(と売上原価)を当期・前期で取り出す(PL の canonical は作らない)。

各書類で BS と同じ区分(連結/単体、``company_checks.csv`` の used_role)の期間コンテキスト
(``CurrentYearDuration`` / ``Prior1YearDuration``)から、次の優先順で最初に値がある要素を使う:
1. 損益計算書の標準要素(jppfs_cor): NetSales、OperatingRevenue1、OperatingRevenue2、
   NetSalesOfCompletedConstructionContracts
2. 主要な経営指標等(jpcrp_cor): NetSales / OperatingRevenue1 / OperatingRevenue2 /
   NetSalesAndOperatingRevenue / NetOperatingRevenue /
   NetSalesOfCompletedConstructionContracts の SummaryOfBusinessResults
売上原価は jppfs_cor_CostOfSales のみ(無ければ欠損)。当期と前期で同じ要素を使えたかも記録する。

    uv run python scripts/extract_sales_stage23_5.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from lxml import etree
from tqdm import tqdm

from fsgen.parsing.instance import Period, duration_context_id_for_role, read_fact
from fsgen.parsing.zip_reader import read_instance_xbrl

SALES_ELEMENTS: tuple[str, ...] = (
    "jppfs_cor_NetSales",
    "jppfs_cor_OperatingRevenue1",
    "jppfs_cor_OperatingRevenue2",
    "jppfs_cor_NetSalesOfCompletedConstructionContracts",
    "jpcrp_cor_NetSalesSummaryOfBusinessResults",
    "jpcrp_cor_OperatingRevenue1SummaryOfBusinessResults",
    "jpcrp_cor_OperatingRevenue2SummaryOfBusinessResults",
    "jpcrp_cor_NetSalesAndOperatingRevenueSummaryOfBusinessResults",
    "jpcrp_cor_NetOperatingRevenueSummaryOfBusinessResults",
    "jpcrp_cor_NetSalesOfCompletedConstructionContractsSummaryOfBusinessResults",
)
COGS_ELEMENT = "jppfs_cor_CostOfSales"
SETS: tuple[tuple[str, str], ...] = (
    ("fy2024", ""),
    ("2023_2024", "_2023_2024"),
    ("2022_2023", "_2022_2023"),
)


def _first(root: etree._Element, elements: tuple[str, ...], ctx: str) -> tuple[int | None, str]:
    for element in elements:
        try:
            value = read_fact(root, element, ctx).value
        except ValueError:  # 同じ事実に矛盾する値(ごく稀)。この要素は使わない
            continue
        if value is not None:
            return value, element
    return None, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim/stage23"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim/stage23_5"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for set_name, suffix in SETS:
        with (args.interim / set_name / "company_checks.csv").open(encoding="utf-8") as f:
            checks = list(csv.DictReader(f))
        out = args.out_dir / f"sales_{set_name}.csv"
        n_ok = 0
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "docID",
                    "sales_cur",
                    "sales_prev",
                    "source_cur",
                    "source_prev",
                    "cogs_cur",
                    "cogs_prev",
                ]
            )
            for row in tqdm(checks, desc=f"sales {set_name}"):
                role = row["used_role"]
                if role not in ("consolidated", "nonconsolidated"):
                    continue
                zip_path = args.raw_dir / f"documents{suffix}" / f"{row['docID']}.zip"
                try:
                    root = etree.fromstring(read_instance_xbrl(zip_path))
                except Exception:  # noqa: BLE001 - 1社の失敗で全体を止めない
                    continue
                vals: dict[Period, tuple[int | None, str, int | None]] = {}
                for period in ("current", "prior"):
                    ctx = duration_context_id_for_role(role, period)
                    sales, source = _first(root, SALES_ELEMENTS, ctx)
                    cogs, _ = _first(root, (COGS_ELEMENT,), ctx)
                    vals[period] = (sales, source, cogs)
                n_ok += vals["current"][0] is not None
                writer.writerow(
                    [
                        row["docID"],
                        vals["current"][0],
                        vals["prior"][0],
                        vals["current"][1],
                        vals["prior"][1],
                        vals["current"][2],
                        vals["prior"][2],
                    ]
                )
        print(f"{set_name}: {len(checks)} 書類、当期の売上高あり {n_ok}")


if __name__ == "__main__":
    main()
