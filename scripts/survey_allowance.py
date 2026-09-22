#!/usr/bin/env python
"""ステージ16 §A.2: 貸倒引当金など「AllowanceFor…」要素の実データ調査。

分析対象の全社(aggregated_core.csv の parse_status==ok かつ非金融)について、
calculation linkbase(連結優先)の葉のうち要素IDが AllowanceFor / DoubtfulAccounts を
含むものを列挙し、要素ごとに 出現社数・出現率・符号・親(cal上)を集計する。
標準タクソノミ以外(拡張要素)も含める。

出力: data/interim/survey/allowance_elements.csv(要素ごと)、allowance_facts.csv(社×要素)
使い方:
    uv run python scripts/survey_allowance.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inspect_values import _load_calc_link_config  # noqa: E402

from fsgen.parsing.calculation_link import parse_calculation_link_bytes, select_tree  # noqa: E402
from fsgen.parsing.instance import context_id_for_role, read_fact  # noqa: E402
from fsgen.parsing.zip_reader import read_cal_xml, read_instance_xbrl  # noqa: E402

_KEYS = ("AllowanceFor", "DoubtfulAccounts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    docs = agg["docID"].tolist()[: args.limit or None]
    role_map, prefixes = _load_calc_link_config()
    rows = []
    for doc in tqdm(docs, desc="survey_allowance"):
        zp = Path(f"data/raw/documents/{doc}.zip")
        try:
            trees = parse_calculation_link_bytes(read_cal_xml(zp), role_map, prefixes,
                                                 source_label=doc)
            tree, role = select_tree(trees)
            root = etree.fromstring(read_instance_xbrl(zp))
        except Exception as exc:  # noqa: BLE001 - 1社の失敗で止めない
            print(f"skip {doc}: {exc}", file=sys.stderr)
            continue
        ctx = context_id_for_role(role)
        for el in tree.elements:
            if not any(k in el for k in _KEYS):
                continue
            f = read_fact(root, el, ctx)
            parents = tree.parents_of(el)
            rows.append({
                "docID": doc, "element": el, "value": f.value,
                "parent": parents[0].parent if parents else "",
                "is_leaf": el not in {a.parent for a in tree.arcs},
            })
    facts = pd.DataFrame(rows)
    facts.to_csv("data/interim/survey/allowance_facts.csv", index=False)
    n = len(docs)
    out = []
    for el, g in facts.groupby("element"):
        v = g["value"].dropna()
        par = Counter(g["parent"]).most_common(2)
        out.append({
            "element": el, "n_companies": g["docID"].nunique(),
            "ratio": g["docID"].nunique() / n, "n_with_value": len(v),
            "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum()),
            "n_zero": int((v == 0).sum()), "top_parents": "; ".join(f"{p}:{c}" for p, c in par),
            "is_extension": not el.startswith("jppfs_cor"),
        })
    res = pd.DataFrame(out).sort_values("n_companies", ascending=False)
    res.to_csv("data/interim/survey/allowance_elements.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(f"分析対象 {n} 社")
    print(res.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
