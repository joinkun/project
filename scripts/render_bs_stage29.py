#!/usr/bin/env python
"""ステージ29 C.4: 同じ 5 社の BS を 4 版(R1・R3・R3 + 上限・R4)で並べ、RE の変化の内訳を付ける。

対象(除外後の 57 社から、外れ値 2 社を除いて選ぶ): アマガサ(S100TBIV)、始点で債務超過の
企業から無作為に 2 社、残りから無作為に 2 社(いずれも numpy の seed 0)。seed 0 の生成を使う。
内訳は ``decompose_re_change``(払込資本の減少分を振替、残りを利益部分)。

    uv run python scripts/render_bs_stage29.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from render_bs_stage24 import select_pcfm
from run_stage25 import OUTLIERS

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import decompose_re_change, subtotal_members
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

VERSIONS = ("r1", "r3", "r3_cap", "r4")
LABELS = {
    "r1": "R1", "r3": "R3", "r3_cap": "R3+上限", "r4": "R4(γ=0)",
    "r3_cap076": "最終(R3+上限 0.757)",
    "equal": "等重み", "frequency": "頻度", "persistence": "持続性",
    "frequency_asym": "頻度+非対称",
    "final": "最終(R3+上限+不活性)", "r1_inactive": "R1+不活性",  # ステージ43
    "prev": "更新前(ステージ43)", "new": "更新後(ステージ45)",  # ステージ45
    "s47_final": "最終設定(ステージ47)", "s47_r1": "R1(感度分析)",  # ステージ47
}
AMAGASA = "S100TBIV"
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"


def _fmt(v: float) -> str:
    return f"{v / 1e6:,.0f}"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage29"))
    ap.add_argument("--out", type=Path, default=Path("results/stage29/bs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--versions", default=",".join(VERSIONS),
                    help="並べる版。企業の選択には先頭の版の始点を使う(始点は版によらない)")
    ap.add_argument("--title", default="ステージ29 C.4: 同じ 5 社の BS(4 版)")
    args = ap.parse_args()
    versions = tuple(args.versions.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]
    gen = {
        v: select_pcfm(pd.read_parquet(args.dir / v / f"nearest_na0__seed{args.seed}.parquet"))
        .set_index("docID")
        for v in versions
    }
    targets = {
        v: pd.read_csv(args.dir / v / "full_targets_nearest_na0.csv").set_index("source_docID")
        for v in versions
    }
    base = gen[versions[0]]
    pool = sorted(set(base.index) - set(OUTLIERS) - {AMAGASA})
    na0 = {d: float(base.loc[d, x0c].to_numpy(dtype=float) @ members["jppfs_cor_NetAssets"])
           for d in pool}
    insolvent = sorted(d for d in pool if na0[d] < 0)
    rng = np.random.default_rng(0)
    pick_ins = [str(d) for d in rng.choice(insolvent, size=2, replace=False)]
    rest = sorted(set(pool) - set(pick_ins))
    pick_rand = [str(d) for d in rng.choice(rest, size=2, replace=False)]
    chosen = [AMAGASA, *pick_ins, *pick_rand]
    kinds = ["指定", "債務超過(無作為)", "債務超過(無作為)", "無作為", "無作為"]

    index_lines = [f"# {args.title}", "",
                   f"始点で債務超過の社(外れ値とアマガサを除く): {len(insolvent)} 社", ""]
    for doc, kind in zip(chosen, kinds, strict=True):
        r0 = base.loc[doc]
        assets = float(r0["assets_start"])
        xs = r0[x0c].to_numpy(dtype=float)
        ends = [gen[v].loc[doc][x1c].to_numpy(dtype=float) for v in versions]
        lines = build_bs_lines(spec, names, [xs * assets, *(e * assets for e in ends)], labels)
        head = [f"# {r0['filerName']}(docID: {doc}、{kind})", "",
                "単位: 百万円。終点は 5 年後(t=1)。seed 0。", ""]
        for v in versions:
            t = targets[v].loc[doc]
            head.append(f"- {LABELS[v]}の対応先: {t['target_name']}(RE/資産 "
                        f"{t['target_RE_over_assets']:.2f})")
        cols = ["始点", *(LABELS[v] for v in versions)]
        table = ["", "| 科目 | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for line in lines:
            label = "&nbsp;&nbsp;" * line.indent + line.label
            if line.values is None:
                table.append(f"| **{line.label}** |" + " |" * len(cols))
                continue
            bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
            table.append(
                f"| {bold}{label}{bold} | " + " | ".join(_fmt(v) for v in line.values) + " |"
            )
        # RE の変化の内訳(振替・利益部分)と払込資本の変化
        i_cs, i_csur = names.index(_CS), names.index(_CSUR)
        rows: dict[str, list[float]] = {"Δ払込資本(資本金 + 資本剰余金)": [], "ΔRE のうち振替": [],
                "ΔRE のうち利益部分": [], "Δ純資産": []}
        for e in ends:
            tr, pr = decompose_re_change(xs, e, names)
            rows["Δ払込資本(資本金 + 資本剰余金)"].append(
                (e[i_cs] + e[i_csur] - xs[i_cs] - xs[i_csur]) * assets
            )
            rows["ΔRE のうち振替"].append(float(tr) * assets)
            rows["ΔRE のうち利益部分"].append(float(pr) * assets)
            rows["Δ純資産"].append(float((e - xs) @ members["jppfs_cor_NetAssets"]) * assets)
        table.append("| **内訳(終点 − 始点)** |" + " |" * len(cols))
        for k, vals in rows.items():
            table.append(f"| {k} | — | " + " | ".join(_fmt(v) for v in vals) + " |")
        (args.out / f"{doc}.md").write_text("\n".join(head + table) + "\n", encoding="utf-8")
        index_lines.append(f"- [{r0['filerName']}]({doc}.md)({kind})")
    (args.out / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print("\n".join(index_lines))


if __name__ == "__main__":
    main()
