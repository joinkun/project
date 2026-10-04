#!/usr/bin/env python
"""ステージ52 D: 密度 + 距離の対応先(版 I・II)と現行の比較。

``evaluate_stage51.py`` の評価を流用する。

規則は ``results/stage52/criteria_b.md`` §6(生成の前に記録)。入力は
``metrics_stage39.py --gen-dir data/interim/stage52/gen --out data/interim/stage52/metrics``。
設定: ``equal``(現行、ステージ50 へのリンク)、``indep``(版 I、採用の候補)、
``pattern``(版 II、比較)。

出力(``results/stage52/``): ``comparison_d.md``、``assets_extreme.md``、``bs_d.md``、
``summary_d.json``、``targets_d.md``(D.3)。

    uv run python scripts/evaluate_stage52.py
"""

from __future__ import annotations

from pathlib import Path

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from density_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

e51._M = Path("data/interim/stage52/metrics")
e51._GEN = Path("data/interim/stage52/gen")
e51.SETTINGS = {"equal": "1 最近傍(現行)", "indep": "2 版 I 密度 + 距離(32 科目・独立ゼロ)",
                "pattern": "3 版 II 密度 + 距離(18 科目、比較)"}
e51.BASE = "equal"
e51.BS_ORDER = ("indep", "pattern")
e51.PAIRS_F2 = ()
e51.STAGE = "ステージ52"
e51.TITLE = "ステージ52 D: 密度 + 距離の対応先と現行の比較"
e51.OUT = {"comparison": "comparison_d.md", "losses": "losses_d.md",
           "assets": "assets_extreme.md", "bs": "bs_d.md", "summary": "summary_d.json"}


def targets_section() -> None:
    """D.3: 対応先の払込資本比率と異なる対応先の数(全社の計画)。"""
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    paid = (agg["jppfs_cor_CapitalStock"].fillna(0) + agg["jppfs_cor_CapitalSurplus"].fillna(0)) \
        / agg["total_assets"]
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    _x, label, docs, _k = load_tilde(names)
    col_docs = np.array(docs)[label == 0]  # 計画の列の並び
    src_docs = np.array(docs)[label == 2]
    lines = ["# ステージ52 D.3: 対応先の性質(全社の計画、class 2 57 社)", "",
             f"始点の払込資本比率の中央値 {paid.loc[src_docs].median():.3f}。"
             "対応先は実在の class 0 企業なので、識別 AUC は行わない(`criteria_b.md` §6)。", "",
             "| 設定 | 対応先の払込資本比率(中央値) | 異なる対応先の数 |", "|---|---:|---:|"]
    plans = {"equal": "data/interim/stage28/plans_equal/plans_nearest_na0.npz",
             "indep": "data/interim/stage52/plans/plans_indep.npz",
             "pattern": "data/interim/stage52/plans/plans_pattern.npz"}
    for s, f in plans.items():
        tg = col_docs[np.argmax(np.load(f)["plan_full"], axis=1)]
        lines.append(f"| {e51.SETTINGS[s]} | {paid.loc[tg].median():.3f} | {len(set(tg))} |")
    Path("results/stage52/targets_d.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    import sys

    targets_section()
    sys.argv = [sys.argv[0], "--out", "results/stage52"]  # 出力先はステージ52(既定はステージ51)
    e51.main()
