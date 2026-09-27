#!/usr/bin/env python
"""ステージ21-B C: canonical のグループ科目の取りこぼし(__OTHER__ に落ちる要素)を調べる。

各書類の選択した BS ツリー(連結優先)の葉要素について、指定した期の値を読み、
``inspect_values.find_bucket`` でどの集計先に落ちるかを記録する。

出力(``--out-dir``):
- ``leaves.csv``: 残余(__OTHER__*)に落ちる葉、および売上債権・契約資産に関係する葉の
  (docID, element, bucket, parent, value, assets)。金額規模・排他性の分析用
- ``other_elements.csv``: __OTHER__* に落ちる要素ごとの出現社数、出現率、cal 上の親(最頻)、
  負値の割合、総資産比の中央値、キーワードから推定したグループ

    uv run python scripts/survey_group_leaks_stage21.py --suffix "" --period current \\
        --out-dir data/interim/stage21/leaks/fy2024_current
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inspect_values import (  # noqa: E402
    _OTHER_PREFIX,
    _load_calc_link_config,
    find_bucket,
    load_canonical,
)

from fsgen.parsing.calculation_link import (  # noqa: E402
    parse_calculation_link_bytes,
    select_tree,
)
from fsgen.parsing.instance import Period, context_id_for_role, read_fact  # noqa: E402
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl  # noqa: E402

# 既存のグループ科目に相当しうる要素のキーワード(要素名のローカル部分に対する正規表現)。
# 取りこぼし候補に印を付けるためだけに使う。判定は人が目で行う。
KEYWORD_GROUPS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("allowance", re.compile(r"AllowanceForDoubtful")),
    ("contract_assets", re.compile(r"ContractAsset")),
    ("receivables_trade", re.compile(r"(Notes|Accounts|Trade)\w*Receivable|ReceivableTrade")),
    ("payables_trade", re.compile(r"(Notes|Accounts|Trade)\w*Payable|PayableTrade")),
    (
        "inventories",
        re.compile(
            r"Inventor|Merchandise|FinishedGoods|WorkInProcess|RawMaterial|Supplies"
            r"|RealEstateForSale|UncompletedConstruction|Stored|SemiFinished|Products"
        ),
    ),
    ("loans", re.compile(r"Loans?Payable|Borrowing|CommercialPaper")),
    ("bonds", re.compile(r"Bond")),
    ("income_taxes", re.compile(r"IncomeTaxes")),
    ("non_controlling", re.compile(r"NonControlling|MinorityInterest")),
    ("valuation_difference", re.compile(r"ValuationDifference")),
)

_RECEIVABLE_RELATED = re.compile(r"Receivable|ContractAsset")


def keyword_group(element_id: str) -> str:
    local = element_id.rsplit("_", 1)[-1]
    for name, pattern in KEYWORD_GROUPS:
        if pattern.search(local):
            return name
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--period", choices=("current", "prior"), default="current")
    parser.add_argument("--canonical", type=Path, default=Path("conf/accounts/core.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    period: Period = args.period

    role_map, prefixes = _load_calc_link_config()
    canonical = load_canonical(args.canonical)
    member_to_name = canonical.member_to_name
    subtotal_ids = canonical.subtotal_ids
    assets_id = canonical.totals["assets"]
    zips = sorted((args.raw_dir / f"documents{args.suffix}").glob("*.zip"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    presence: dict[str, set[str]] = defaultdict(set)
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    parents: dict[str, Counter[str]] = defaultdict(Counter)
    negatives: Counter[str] = Counter()
    rel_values: dict[str, list[float]] = defaultdict(list)
    with (args.out_dir / "leaves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["docID", "used_role", "element", "bucket", "parent", "value", "assets"])
        for zip_path in tqdm(zips, desc=f"leaks{args.suffix}:{period}"):
            doc_id = zip_path.stem
            # 1 書類の葉を先に全部読み、失敗しなかった書類だけを集計に入れる
            # (inspect_values と同じく、値の矛盾などで 1 社が落ちても全体は止めない)。
            try:
                trees = parse_calculation_link_bytes(
                    read_cal_xml(zip_path), role_map, prefixes, source_label=doc_id
                )
                tree, used_role = select_tree(trees)
                root = etree.fromstring(read_instance_xbrl(zip_path))
                context_id = context_id_for_role(used_role, period)
                assets = read_fact(root, assets_id, context_id).value
                leaves = [(leaf, read_fact(root, leaf, context_id).value) for leaf in tree.leaves()]
            except Exception:  # noqa: BLE001 - 1社の失敗で全体を止めない
                continue
            if assets is None:
                continue  # この期の BS が無い(連結初年度の前期など)
            n_docs += 1
            for leaf, value in leaves:
                if value is None or value == 0:
                    continue
                bucket = find_bucket(leaf, tree, member_to_name, subtotal_ids) or ""
                arcs = tree.parents_of(leaf)
                parent = arcs[0].parent if arcs else ""
                is_other = bucket.startswith(_OTHER_PREFIX)
                if is_other or _RECEIVABLE_RELATED.search(leaf):
                    writer.writerow([doc_id, used_role, leaf, bucket, parent, value, assets])
                if not is_other:
                    continue
                presence[leaf].add(doc_id)
                buckets[leaf][bucket] += 1
                parents[leaf][parent] += 1
                negatives[leaf] += value < 0
                if assets:
                    rel_values[leaf].append(abs(value) / abs(assets))

    with (args.out_dir / "other_elements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "element",
                "keyword_group",
                "n_docs",
                "share",
                "bucket",
                "parent",
                "share_negative",
                "median_abs_over_assets",
            ]
        )
        for leaf, docs in sorted(presence.items(), key=lambda kv: -len(kv[1])):
            writer.writerow(
                [
                    leaf,
                    keyword_group(leaf),
                    len(docs),
                    f"{len(docs) / n_docs:.4f}",
                    buckets[leaf].most_common(1)[0][0],
                    parents[leaf].most_common(1)[0][0],
                    f"{negatives[leaf] / len(docs):.2f}",
                    f"{statistics.median(rel_values[leaf]):.4f}" if rel_values[leaf] else "",
                ]
            )
    print(f"対象 {len(zips)} 書類、この期の BS がある {n_docs} 書類")
    print(f"__OTHER__ に落ちる要素の種類: {len(presence)}")


if __name__ == "__main__":
    main()
