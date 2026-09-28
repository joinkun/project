#!/usr/bin/env python
"""ステージ31 C.3: 実際に GC 注記が外れた企業(再建群)2 社の、2022 年度末と 2024 年度末の BS。

再建群 15 社(``results/stage30/recovery_firms.csv``)から、2022 年度末に債務超過で 2024 年度末に
解消した社から 1 社、残りから 1 社を無作為に選ぶ(numpy seed 0)。値はステージ23 のパネルの
2022 年度・2024 年度の書類の当期末(``cur_``)。内訳は ``decompose_re_change``(払込資本の減少分を
振替、残りを利益部分)で、生成の表と同じ規則。

    uv run python scripts/render_actual_bs_stage31.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import decompose_re_change, subtotal_members
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"


def _fmt(v: float) -> str:
    return f"{v / 1e6:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--firms", type=Path, default=Path("results/stage30/recovery_firms.csv"))
    ap.add_argument("--panel", type=Path, default=Path("data/interim/stage23/panel/panel.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/stage31/bs_actual"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.firms, dtype={"edinetCode": str}).set_index("edinetCode")
    y = df[(df["group"] == "Y") & (df["reason"].fillna("") == "") & df["in_panel"]]
    resolved = sorted(y.index[y["resolved"].astype(bool)])
    rng = np.random.default_rng(0)
    pick1 = str(rng.choice(resolved))
    rest = sorted(set(y.index) - {pick1})
    pick2 = str(rng.choice(rest))

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    panel = pd.read_csv(args.panel, dtype={"edinetCode": str})
    index_lines = ["# ステージ31 C.3: 実際に再建した企業の BS(2022 年度末 → 2024 年度末)", ""]
    for code, kind in ((pick1, "債務超過を解消(無作為)"), (pick2, "無作為")):
        p = panel[panel["edinetCode"] == code].set_index("t")
        x22 = p.loc[2022, [f"cur_{n}" for n in names]].fillna(0.0).to_numpy(dtype=float)
        x24 = p.loc[2024, [f"cur_{n}" for n in names]].fillna(0.0).to_numpy(dtype=float)
        lines = build_bs_lines(spec, names, [x22, x24], labels)
        r = y.loc[code]
        head = [f"# {r['filerName']}({code}、{kind})", "",
                "単位: 百万円。実データ(2022 年度末 → 2024 年度末、2 年)。GC 注記: 2022 あり → "
                f"2024 {'なし' if int(r['y2024']) == 0 else '重要事象のみ'}。", ""]
        table = ["| 科目 | 2022 年度末 | 2024 年度末 |", "|---|---:|---:|"]
        for line in lines:
            label = "&nbsp;&nbsp;" * line.indent + line.label
            if line.values is None:
                table.append(f"| **{line.label}** | | |")
                continue
            bold = "**" if line.kind in ("subtotal", "total", "diff") else ""
            table.append(f"| {bold}{label}{bold} | " + " | ".join(_fmt(v) for v in line.values)
                         + " |")
        tr, pr = decompose_re_change(x22, x24, names)
        i_cs, i_csur = names.index(_CS), names.index(_CSUR)
        d_paid = x24[i_cs] + x24[i_csur] - x22[i_cs] - x22[i_csur]
        d_na = float((x24 - x22) @ members["jppfs_cor_NetAssets"])
        a0 = float(x22 @ members["jppfs_cor_Assets"])
        table += [
            "| **内訳(2024 − 2022)** | | |",
            f"| Δ払込資本(資本金 + 資本剰余金) | — | {_fmt(d_paid)} |",
            f"| ΔRE のうち振替 | — | {_fmt(float(tr))} |",
            f"| ΔRE のうち利益部分 | — | {_fmt(float(pr))} |",
            f"| Δ純資産 | — | {_fmt(d_na)} |",
            f"| 参考: 利益部分 / 2022 年度末の総資産(2 年) | — | {float(pr) / a0:+.3f} |",
        ]
        (args.out / f"{code}.md").write_text("\n".join(head + table) + "\n", encoding="utf-8")
        index_lines.append(f"- [{r['filerName']}]({code}.md)({kind})")
    (args.out / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print("\n".join(index_lines))


if __name__ == "__main__":
    main()
