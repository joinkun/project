#!/usr/bin/env python
"""ステージ74: BS 表(class 2、ステージ65〜73 と同じ 5 社)。目で読むための表。

列は始点、ステージ69 の設定、ステージ72 の F1(72 社、リークあり)、A1(リークなし)、B1(相対)、
B2(ハイブリッド)。λ はそれぞれ class 1 で較正した値(``results/stage74/summary.json``)。

出力: ``results/stage74/bs.md``。

    uv run python scripts/stage74_bs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from evaluate_stage65 import bs_table
from tables_stage69 import INTERIM

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members
from fsgen.viz.balance_sheet import load_labels

OUT = Path("results/stage74")


def load(gen: Path, setting: str) -> pd.DataFrame:
    e51._GEN = gen
    g = e51.load_gen("c2", setting)
    out: pd.DataFrame = g[g["seed"] == 0].set_index("docID")
    return out


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    star = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["star"]
    g1 = load(INTERIM / "stage67/all", "s1__l1")
    cols_def = [("ステージ72 F1", load(INTERIM / "stage72/gen", "f1__l0.3"))] + [
        (f"{st} λ̃={star[st]:g}", load(INTERIM / "stage74/gen", f"{st}__l{star[st]:g}"))
        for st in ("A1", "B1", "B2") if star.get(st) is not None]
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g1[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    docs = np.array(g1.index)
    r_ = np.random.default_rng(0)
    ins = docs[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in docs if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    out = ["# ステージ74: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
           "企業はステージ65〜73 と同じ 5 社。A1 = F1-abs(リークなし)、B1 = F1-rel、"
           "B2 = F1-hyb。", ""]
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
