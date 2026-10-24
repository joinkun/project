#!/usr/bin/env python
"""ステージ71: BS 表(class 2、ステージ65〜70 と同じ選び方の 5 社)。目で読むための表。

列は始点、設定 1、設定 2・F1・F2 の終点(事前の規則の λ̃*_0 と、感度の λ̃ = 0.3)。

出力: ``results/stage71/bs.md``。

    uv run python scripts/stage71_bs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_stage51 as e51
import numpy as np
from evaluate_stage65 import bs_table
from tables_stage69 import INTERIM

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage71")


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    star = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["star"]
    e51._GEN = INTERIM / "stage67/all"
    g1 = (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", "s1__l1"))
    e51._GEN = INTERIM / "stage71/gen"
    cols_def = [(f"{lab} λ̃={lt:g}", f"{st}__l{lt:g}")
                for st, lab in (("a2", "設定 2"), ("f1", "F1"), ("f2", "F2"))
                for lt in dict.fromkeys((float(star[st]), 0.3))]
    gens = {k: (lambda g: g[g["seed"] == 0].set_index("docID"))(e51.load_gen("c2", k))
            for _, k in cols_def}
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g1[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g1.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    out = ["# ステージ71: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65〜70 と同じ規則(乱数 seed 0、始点で債務超過から 2 社、残りから 3 社)。"
           "λ̃ = 0.3 は感度(QP の許容内の項目 5 を違反としない場合の λ̃*_0)。", ""]
    for d in picks:
        r0 = g1.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0), ("設定 1", r0[x1c].to_numpy(float) * a0),
                *((lab, gens[k].loc[d][x1c].to_numpy(float) * a0) for lab, k in cols_def)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        out += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
    (OUT / "bs.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
