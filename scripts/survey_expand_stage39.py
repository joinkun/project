#!/usr/bin/env python
"""ステージ39 A.2: 小計を開くための子要素の調査(有形固定資産・投資その他の資産)。

評価対象の書類(class 0・2、解析成功、金融業でない)について、選ばれた計算リンクの木で、
小計の **直下の子** とその下の子孫を集める。各要素について、当期末の値がある書類の割合
(全体・class 2)と、小計に占める金額の中央値を出す。値は ``inspect_values`` と同じ文脈で読む。

    uv run python scripts/survey_expand_stage39.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inspect_values import _load_calc_link_config  # noqa: E402

from fsgen.parsing.calculation_link import parse_calculation_link_bytes, select_tree  # noqa: E402
from fsgen.parsing.instance import context_id_for_role, read_fact  # noqa: E402
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl  # noqa: E402

TARGETS = ("jppfs_cor_PropertyPlantAndEquipment", "jppfs_cor_InvestmentsAndOtherAssets")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=Path("data/raw/documents"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/stage39/survey.csv"))
    args = ap.parse_args()
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    m = agg[["docID"]].merge(lab[["docID", "label"]], on="docID")
    m = m[pd.to_numeric(m["label"], errors="coerce").isin([0, 2])]
    role_map, prefixes = _load_calc_link_config()
    rows = []
    for doc, label in tqdm(zip(m["docID"], m["label"].astype(int), strict=True), total=len(m)):
        z = args.docs / f"{doc}.zip"
        try:
            trees = parse_calculation_link_bytes(read_cal_xml(z), role_map, prefixes,
                                                 source_label=doc)
            tree, role = select_tree(trees)
            root = etree.fromstring(read_instance_xbrl(z))
            ctx = context_id_for_role(role, "current")
        except Exception:  # noqa: BLE001 - 調査なので読めない書類は数えない
            continue
        for tgt in TARGETS:
            total = read_fact(root, tgt, ctx).value
            # 直下の子と、その深さ
            depth: dict[str, int] = defaultdict(int)
            frontier = [(a.child, 1) for a in tree.children_of(tgt)]
            while frontier:
                el, d = frontier.pop()
                if el in depth:
                    continue
                depth[el] = d
                frontier += [(a.child, d + 1) for a in tree.children_of(el)]
            for el, d in depth.items():
                v = read_fact(root, el, ctx).value
                rows.append({"docID": doc, "label": label, "subtotal": tgt, "element": el,
                             "depth": d, "value": v, "subtotal_value": total})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out.with_suffix(".parquet"), index=False)
    print(len(rows))


if __name__ == "__main__":
    main()
