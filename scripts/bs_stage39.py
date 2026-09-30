#!/usr/bin/env python
"""ステージ39 D.3: 細分化(D = 38)の生成の BS 表(5 社、seed 0、単位 百万円)。

細分化用のルート(``data/stage39_root``)で実行する。企業は除外後の class 2(55 社)から
乱数(seed 0)で 5 社を選ぶ(結果を見て選ばない)。列は始点、``--settings`` の各設定の終点。

    cd data/stage39_root && uv run --project ../.. python ../../scripts/bs_stage39.py \\
        --settings nearest_na0 --out ../../results/stage39/bs.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from run_stage25 import OUTLIERS

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.viz.balance_sheet import build_bs_lines, load_labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", type=Path, default=Path("data/interim/stage39/gen/c2"))
    ap.add_argument("--settings", required=True, help="カンマ区切りの設定(ディレクトリ名)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    settings = args.settings.split(",")
    gens = {}
    for s in settings:
        (f,) = sorted((args.gen_dir / s).glob("*__seed0.parquet"))
        g = pd.read_parquet(f)
        gens[s] = g[g["method"] == "PCFM"].set_index("docID")
    pool = sorted(set(gens[settings[0]].index) - set(OUTLIERS))
    docs = [str(d) for d in np.random.default_rng(0).choice(pool, 5, replace=False)]
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    out = [f"# ステージ39 D.3: BS 表(D = {len(names)}、seed 0、単位 百万円、5 年 = t=1)", "",
           "企業は除外後の class 2(55 社)から乱数(seed 0)で選んだ 5 社。", ""]
    for d in docs:
        r = gens[settings[0]].loc[d]
        cols = [("始点(class 2)", r[x0c].to_numpy(float))]
        cols += [(f"終点 {s}", gens[s].loc[d][x1c].to_numpy(float)) for s in settings]
        assets = float(r["assets_start"])
        bl = build_bs_lines(spec, names, [v * assets for _, v in cols], labels)
        out += [f"## {r['filerName']}({d})", "",
                "| 科目 | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
        for line in bl:
            if line.values is None:
                out.append(f"| **{line.label}** |" + " |" * len(cols))
                continue
            lab = "&nbsp;&nbsp;" * line.indent + line.label
            bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
            out.append(f"| {bold}{lab}{bold} | "
                       + " | ".join(f"{v / 1e6:,.0f}" for v in line.values) + " |")
        out.append("")
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
