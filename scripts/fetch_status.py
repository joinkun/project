#!/usr/bin/env python
"""fetch_log.csv の進捗確認(別ターミナルから実行する用)。

使い方:
    uv run python scripts/fetch_status.py
    uv run python scripts/fetch_status.py --interval 30   # N秒ごとに再表示
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def load_target_doc_ids(document_list_csv: Path) -> set[str]:
    with document_list_csv.open(newline="", encoding="utf-8") as f:
        return {row["docID"] for row in csv.DictReader(f)}


def load_fetch_log(fetch_log_csv: Path) -> dict[str, dict[str, str]]:
    if not fetch_log_csv.exists():
        return {}
    with fetch_log_csv.open(newline="", encoding="utf-8") as f:
        return {row["docID"]: row for row in csv.DictReader(f)}


def print_status(document_list_csv: Path, fetch_log_csv: Path) -> None:
    if not document_list_csv.exists():
        print(f"見つかりません: {document_list_csv}")
        return

    target_ids = load_target_doc_ids(document_list_csv)
    log_rows = load_fetch_log(fetch_log_csv)

    ok = [d for d in target_ids if log_rows.get(d, {}).get("status") == "ok"]
    failed = [d for d in target_ids if log_rows.get(d, {}).get("status") == "failed"]
    pending = [d for d in target_ids if d not in log_rows]

    print(f"対象総数: {len(target_ids)}")
    print(f"  ok    : {len(ok)}")
    print(f"  failed: {len(failed)}")
    print(f"  未処理 : {len(pending)}")

    if failed:
        print("\n失敗した docID:")
        for doc_id in sorted(failed):
            row = log_rows[doc_id]
            print(f"  {doc_id}: http_status={row.get('http_status')} error={row.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--interval", type=float, default=None, help="この秒数ごとに再表示し続ける(省略時は1回だけ)"
    )
    parser.add_argument(
        "--suffix", default="", help="提出期間の接尾辞(conf/data/*.yaml の output_suffix)"
    )
    args = parser.parse_args()

    document_list_csv = args.raw_dir / "index" / f"document_list{args.suffix}.csv"
    fetch_log_csv = args.raw_dir / "index" / f"fetch_log{args.suffix}.csv"

    if args.interval is None:
        print_status(document_list_csv, fetch_log_csv)
        return

    while True:
        print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        print_status(document_list_csv, fetch_log_csv)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
