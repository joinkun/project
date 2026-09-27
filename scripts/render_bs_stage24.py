#!/usr/bin/env python
"""ステージ24 C.3: 同じ 5 社の BS を、コスト重みの版ごとに横に並べる(numpy/pandas/yaml のみ)。

対象: 旭川国際ゴルフ場(S100VFZB)、アマガサ(S100TBIV)、小山カントリー倶楽部(S100VI0I)と、
残りの 65 社から無作為(seed 0)に 2 社。生成結果は最終設定 W1_S5_a05_b03・PCFM・seed 0。
列: 始点、各版の終点。各版の対応先(輸送で選ばれた健全企業)を見出しに書く。

    uv run python scripts/render_bs_stage24.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.viz.balance_sheet import build_bs_lines, load_labels

_CONFIG = "W1_S5_a05_b03"
FIXED = ("S100VFZB", "S100TBIV", "S100VI0I")
VERSIONS = ("equal", "estimated", "qualitative", "old")
LABELS = {
    "equal": "等重み",
    "estimated": "推定ベース",
    "qualitative": "定性ベース",
    "old": "旧重み(参考)",
}


def _fmt(v: float) -> str:
    return f"{v / 1e6:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/interim/stage24"))
    ap.add_argument("--out", type=Path, default=Path("results/stage24/bs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    x0c = [f"x0_{n}" for n in names]
    x1c = [f"x1_{n}" for n in names]

    gen: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.DataFrame] = {}
    for v in VERSIONS:
        df = pd.read_parquet(args.dir / v / f"nearest_na0__seed{args.seed}.parquet")
        gen[v] = df[(df["method"] == "PCFM") & (df["config"] == _CONFIG)].set_index("docID")
        targets[v] = pd.read_csv(args.dir / v / "full_targets_nearest_na0.csv").set_index(
            "source_docID"
        )
    others = sorted(set(gen["equal"].index) - set(FIXED))
    rng = np.random.default_rng(0)
    chosen = [*FIXED, *(str(d) for d in rng.choice(others, size=2, replace=False))]

    index_lines = ["# ステージ24 C.3: 同じ 5 社の BS(版ごと)", ""]
    for doc in chosen:
        r0 = gen["equal"].loc[doc]
        assets = float(r0["assets_start"])
        xs = r0[x0c].to_numpy(dtype=float) * assets
        ends = [gen[v].loc[doc][x1c].to_numpy(dtype=float) * assets for v in VERSIONS]
        lines = build_bs_lines(spec, names, [xs, *ends], labels)
        head = [f"# {r0['filerName']}(docID: {doc})", "", "単位: 百万円。終点は 5 年後(t=1)。", ""]
        for v in VERSIONS:
            t = targets[v].loc[doc]
            head.append(
                f"- {LABELS[v]}の対応先: {t['target_name']}(docID {t['target_docID']}、"
                f"RE/資産 {t['target_RE_over_assets']:.2f})"
            )
        cols = ["始点", *(LABELS[v] for v in VERSIONS)]
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
        (args.out / f"{doc}.md").write_text("\n".join(head + table) + "\n", encoding="utf-8")
        index_lines.append(f"- [{r0['filerName']}]({doc}.md)")
    (args.out / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print("\n".join(index_lines))


if __name__ == "__main__":
    main()
