#!/usr/bin/env python
"""ステージ32 D.1: 停止規則の判定(``results/stage32/criteria.md`` の「勝つ」の定義)。

numpy/pandas のみ。``run_stage25.py`` の出力(``metrics/``)から、版 2〜4 の等重みとのペア比較を読み、
主指標 3 つ(純資産の改善 ↑、債務超過の解消率 ↑、|総資産変化| ↓)について、差の中央値の 95%
ブートストラップ区間が 0 を含まず良い向き(有意に優れる)/ 悪い向き(有意に劣る)かを判定する。
区間が [0, 0] や全差 0(NaN)は「差なし」。

    uv run python scripts/evaluate_stage32.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PRIMARY = (
    ("純資産の改善", "大"),
    ("債務超過の解消率(社ごとの 3 seed 平均)", "大"),
    ("|総資産変化|", "小"),
)
SECONDARY = (("sparsity", "小"), ("validity LightGBM", "大"), ("等重み proximity", "小"))
CANDIDATES = ("frequency", "frequency_asym")
LABELS = {"frequency": "版 2 頻度", "persistence": "版 3 持続性(候補外)",
          "frequency_asym": "版 4 頻度 + 非対称"}


def verdict(row: pd.Series, better: str) -> str:
    lo, hi = row["ci_low"], row["ci_high"]
    if pd.isna(lo) or (lo == 0 and hi == 0):
        return "差なし"
    good = lo > 0 if better == "大" else hi < 0
    bad = hi < 0 if better == "大" else lo > 0
    return "有意に優れる" if good else ("有意に劣る" if bad else "有意差なし")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results/stage32"))
    args = ap.parse_args()
    paired = pd.read_csv(args.results / "metrics" / "paired_excl_outliers55.csv")
    summ = pd.read_csv(args.results / "metrics" / "summary_excl_outliers55.csv").set_index(
        "version")
    lines = ["# ステージ32 D.1: 停止規則の判定(55 社、各版 − 等重み)", "",
             "| 版 | 指標 | 差の中央値 | 95% CI | Wilcoxon p | 判定 |",
             "|---|---|---:|---|---:|---|"]
    wins = {}
    for v in ("frequency", "persistence", "frequency_asym"):
        n_good = n_bad = 0
        for metric, better in (*PRIMARY, *SECONDARY):
            r = paired[(paired["version"] == v) & (paired["metric"] == metric)].iloc[0]
            j = verdict(r, better)
            if (metric, better) in PRIMARY:
                n_good += j == "有意に優れる"
                n_bad += j == "有意に劣る"
            med = "0(全社で差 0)" if pd.isna(r["median_diff(version-equal)"]) else (
                f"{r['median_diff(version-equal)']:+.4f}")
            ci = "—" if pd.isna(r["ci_low"]) else f"[{r['ci_low']:+.4f}, {r['ci_high']:+.2g}]"
            p = "—" if pd.isna(r["p"]) else f"{r['p']:.2g}"
            tag = "主" if (metric, better) in PRIMARY else "副"
            shown = metric.replace("|", "\\|")
            lines.append(f"| {LABELS[v]} | {tag}: {shown} | {med} | {ci} | {p} | {j} |")
        ok = int(summ.loc[v, "violations(rows)"]) == 0 and n_good >= 2 and n_bad == 0
        wins[v] = (ok, n_good, n_bad)
    lines += ["", "| 版 | 違反(55 社) | 主指標で有意に優れる | 有意に劣る | 勝つ |",
              "|---|---:|---:|---:|---|"]
    for v, (ok, g, b) in wins.items():
        lines.append(f"| {LABELS[v]} | {int(summ.loc[v, 'violations(rows)'])} | {g} / 3 | {b} "
                     f"| {'○' if ok and v in CANDIDATES else '×'} |")
    won = [v for v in CANDIDATES if wins[v][0]]
    lines += ["", "**結論: " + ("勝った版あり: " + "、".join(LABELS[v] for v in won)
                              if won else "版 2・版 4 とも等重みに勝てない。停止規則により"
                              "等重みを採用し、重みの探索を終了する。") + "**"]
    (args.results / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
