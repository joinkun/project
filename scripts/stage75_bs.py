#!/usr/bin/env python
"""ステージ75: BS 表(class 2、ステージ65〜74 と同じ 5 社)。目で読むための表。

列は始点、ステージ69 の設定、D2(57 社の全体のコスト)の各設定。λ はそれぞれ D2 の λ̃*
(``results/stage75/summary.json``)。

出力: ``results/stage75/bs.md``。

    uv run python scripts/stage75_bs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from evaluate_stage65 import bs_table
from stage74_bs import load
from tables_stage69 import INTERIM

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage75")
SHOW = ("abs_argmax", "abs_f1", "R1_f1", "R2_f1", "R3_f1")


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    d2 = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["d2"]
    g1 = load(INTERIM / "stage67/all", "s1__l1")
    cols_def = [(f"{st.replace('_f1', '・F1').replace('_argmax', '・argmax')} "
                 f"λ̃={d2[st]['star']:g}",
                 load(INTERIM / "stage75/gen_full", f"{st}__l{d2[st]['star']:g}"))
                for st in SHOW if d2[st]["star"] is not None]
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g1[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g1.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    out = ["# ステージ75: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65〜74 と同じ 5 社。列は D2(57 社の全体のコスト)の各設定。", ""]
    for d in picks:
        r0 = g1.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0),
                ("ステージ69", r0[x1c].to_numpy(float) * a0),
                *((lab, g.loc[d][x1c].to_numpy(float) * a0) for lab, g in cols_def)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        out += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
    (OUT / "bs.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
