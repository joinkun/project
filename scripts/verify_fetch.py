"""ステージ21-A: 提出期間ごとの取得結果を検証する。

各提出期間について、ドライラン件数・取得の成否・``_cal.xml`` と ``AuditDoc/`` の
欠落・総容量・タクソノミのバージョン(``_cal.xml`` 内の jppfs 名前空間の日付)を集計する。

    uv run python scripts/verify_fetch.py --suffix "" --suffix _2023_2024 \\
        --out results/stage21/fetch_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path

_TAXONOMY_RE = re.compile(rb"http://disclosure\.edinet-fsa\.go\.jp/taxonomy/jppfs/(\d{4}-\d{2}-\d{2})")


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _status_counts(fetch_log: Path) -> Counter[str]:
    if not fetch_log.exists():
        return Counter()
    with fetch_log.open(newline="", encoding="utf-8") as f:
        return Counter(row["status"] for row in csv.DictReader(f))


def _inspect_zip(path: Path) -> tuple[bool, bool, str | None]:
    """(cal.xml あり, AuditDoc あり, jppfs タクソノミの日付) を返す。"""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        cal = [n for n in names if n.startswith("XBRL/PublicDoc/") and n.endswith("_cal.xml")]
        has_audit = any(n.startswith("XBRL/AuditDoc/") for n in names)
        version = None
        if cal:
            m = _TAXONOMY_RE.search(zf.read(cal[0]))
            version = m.group(1).decode() if m else None
    return bool(cal), has_audit, version


def summarize(raw_dir: Path, suffix: str) -> dict[str, str]:
    index_dir = raw_dir / "index"
    documents_dir = raw_dir / f"documents{suffix}"
    statuses = _status_counts(index_dir / f"fetch_log{suffix}.csv")
    zips = sorted(documents_dir.glob("*.zip")) if documents_dir.exists() else []

    missing_cal = 0
    missing_audit = 0
    versions: Counter[str] = Counter()
    for path in zips:
        has_cal, has_audit, version = _inspect_zip(path)
        missing_cal += not has_cal
        missing_audit += not has_audit
        versions[version or "不明"] += 1

    return {
        "suffix": suffix or "(fy2024)",
        "listed": str(_count_rows(index_dir / f"document_list{suffix}.csv")),
        "ok": str(statuses.get("ok", 0)),
        "failed": str(statuses.get("failed", 0)),
        "zip_files": str(len(zips)),
        "missing_cal": str(missing_cal),
        "missing_audit_doc": str(missing_audit),
        "total_gb": f"{sum(p.stat().st_size for p in zips) / 1e9:.2f}",
        "jppfs_versions": "; ".join(f"{v}={n}" for v, n in versions.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--suffix", action="append", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    suffixes: list[str] = args.suffix if args.suffix is not None else [""]
    rows = [summarize(args.raw_dir, s) for s in suffixes]
    for row in rows:
        print(row)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
