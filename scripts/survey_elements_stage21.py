#!/usr/bin/env python
"""ステージ21-A D: BS 要素の出現頻度とタクソノミのバージョンを、提出期間・期別に集計する。

各書類で、選択した BS ツリー(連結を優先)の要素のうち、当期・前期のコンテキストに
値(nil でも空でもないファクト)を持つものを数える。jppfs タクソノミの日付
(インスタンスの名前空間 URI)も記録する。

出力(``--out-dir``):
- ``elements.csv``: element_id, role 別の出現社数(当期 / 前期)
- ``taxonomy.csv``: jppfs タクソノミの日付ごとの社数

    uv run python scripts/survey_elements_stage21.py --suffix "" \\
        --out-dir data/interim/stage21/fy2024
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from lxml import etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inspect_values import _load_calc_link_config  # noqa: E402

from fsgen.parsing.calculation_link import (  # noqa: E402
    parse_calculation_link_bytes,
    select_tree,
)
from fsgen.parsing.instance import Period, context_id_for_role, read_fact  # noqa: E402
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl  # noqa: E402

_PERIODS: tuple[Period, ...] = ("current", "prior")
_JPPFS_DATE = re.compile(r"/taxonomy/jppfs/(\d{4}-\d{2}-\d{2})/")


def survey_document(
    zip_path: Path, role_map: dict[str, str], prefixes: tuple[str, ...]
) -> tuple[str, str | None, dict[Period, set[str]]]:
    """(使った role, jppfs の日付, 期 -> 値のある BS 要素の集合)。"""
    trees = parse_calculation_link_bytes(
        read_cal_xml(zip_path), role_map, prefixes, source_label=zip_path.stem
    )
    tree, used_role = select_tree(trees)
    root = etree.fromstring(read_instance_xbrl(zip_path))
    version = next(
        (m.group(1) for uri in root.nsmap.values() if (m := _JPPFS_DATE.search(uri or ""))),
        None,
    )
    present: dict[Period, set[str]] = {}
    for period in _PERIODS:
        context_id = context_id_for_role(used_role, period)
        present[period] = {
            e for e in tree.elements if read_fact(root, e, context_id).value is not None
        }
    return used_role, version, present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    role_map, prefixes = _load_calc_link_config()
    zips = sorted((args.raw_dir / f"documents{args.suffix}").glob("*.zip"))

    counts: Counter[tuple[str, str, Period]] = Counter()
    n_role: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    failed = 0
    for zip_path in tqdm(zips, desc=f"elements{args.suffix}"):
        try:
            role, version, present = survey_document(zip_path, role_map, prefixes)
        except Exception:  # noqa: BLE001 - 1社の失敗で全体を止めない
            failed += 1
            continue
        n_role[role] += 1
        versions[version or "不明"] += 1
        for period, present_elements in present.items():
            counts.update((e, role, period) for e in present_elements)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    elements = sorted({e for e, _, _ in counts})
    with (args.out_dir / "elements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["element_id"]
        for role in sorted(n_role):
            header += [f"{role}_current", f"{role}_prior", f"{role}_n_docs"]
        writer.writerow(header)
        for e in elements:
            row: list[object] = [e]
            for role in sorted(n_role):
                row += [counts[(e, role, "current")], counts[(e, role, "prior")], n_role[role]]
            writer.writerow(row)
    with (args.out_dir / "taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["jppfs_date", "n_docs"])
        for version, n in sorted(versions.items()):
            writer.writerow([version, n])

    print(f"対象 {len(zips)} 書類(失敗 {failed})、role 別: {dict(n_role)}")
    print(f"jppfs タクソノミ: {dict(sorted(versions.items()))}")
    for role in sorted(n_role):
        n_cur = sum(1 for e in elements if counts[(e, role, "current")])
        n_pri = sum(1 for e in elements if counts[(e, role, "prior")])
        print(f"  {role}: BS 要素の種類数 当期 {n_cur} / 前期 {n_pri}")


if __name__ == "__main__":
    main()
