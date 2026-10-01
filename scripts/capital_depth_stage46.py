#!/usr/bin/env python
"""ステージ46 A.1: 実データの減資の深さ(資本金の当期末 / 前期末)の分布。

推定用パネル(ステージ23、同じ書類の前期末 → 当期末、1 年、3 遷移)で、前期末の資本金が正で
当期末に減った組(比 < 1)について、比の分布を class 別に出す。class は始点の年度のラベル
(2021 年度のラベルはないので、class 別は 2 遷移、全体は 3 遷移)。あわせて 2 年
(2022 → 2024 年度末)の class 1・2(特殊業種を除く)の分布も出す。

    uv run python scripts/capital_depth_stage46.py --out results/stage46/capital_depth.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_stage34 import special_reason

_CS = "jppfs_cor_CapitalStock"
QS = (1, 5, 10, 25, 50, 75, 90, 95)


def describe(r: np.ndarray) -> dict[str, float]:
    out = {f"p{q}": float(np.percentile(r, q)) for q in QS} if len(r) else {}
    out.update({"n": float(len(r)), "min": float(r.min()) if len(r) else np.nan,
                "lt001": float(np.mean(r < 0.01)) if len(r) else np.nan,
                "n_lt001": float(np.sum(r < 0.01)),
                "zero": float(np.sum(r == 0)), "lt05": float(np.sum(r < 0.5))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    lp = np.array([lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                              strict=True)])
    x0 = p[f"prev_{_CS}"].fillna(0).to_numpy()
    x1 = p[f"cur_{_CS}"].fillna(0).to_numpy()
    dec = (x0 > 0) & (x1 < x0)
    ratio = np.where(x0 > 0, x1 / np.where(x0 > 0, x0, 1.0), np.nan)
    rows = {}
    for key, m in (("全体(3 遷移)", np.ones(len(p), dtype=bool)), ("class 0", lp == 0),
                   ("class 1", lp == 1), ("class 2", lp == 2), ("class 1+2", np.isin(lp, [1, 2]))):
        rows[key] = describe(ratio[dec & m]) | {"n_pairs": float(((x0 > 0) & m).sum())}
    # 2 年: class 1・2(2022 年度)、特殊業種を除く(ステージ44・45 の実データと同じ母集団)
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    a = p[p["t"] == 2023].set_index("edinetCode")
    b = p[p["t"] == 2024].set_index("edinetCode")
    both = [c for c in a.index.intersection(b.index) if wl.loc[c, 2022] in (1, 2)]
    reason = special_reason(fy, both)
    press = [c for c in both if reason.get(c, "") == ""]
    y0 = a.loc[press, f"prev_{_CS}"].fillna(0).to_numpy()
    y1 = b.loc[press, f"cur_{_CS}"].fillna(0).to_numpy()
    d2 = (y0 > 0) & (y1 < y0)
    rows["class 1・2 の 2 年(172 社)"] = (describe(y1[d2] / y0[d2])
                                          | {"n_pairs": float((y0 > 0).sum())})
    t = pd.DataFrame(rows).T
    t.to_csv(args.out.with_suffix(".csv"))
    lines = ["# ステージ46 A.1: 減資の深さ(資本金の当期末 / 前期末)", "",
             "推定用パネル(1 年)。資本金が前期末に正で、当期末に減った組の比。class は始点の年"
             "度のラベル。", "",
             "| 母集団 | 資本金が正の組 | 減った組 | 最小 | p1 | p5 | p10 | p25 | p50 | p75 | p9"
             "0 | p95 "
             "| 比 < 0.5 | 比 < 0.01 | 完全に 0 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in t.iterrows():
        if r["n"] == 0:
            continue
        lines.append(
            f"| {k} | {int(r['n_pairs'])} | {int(r['n'])} | {r['min']:.4f} | {r['p1']:.4f} "
            f"| {r['p5']:.4f} | {r['p10']:.4f} | {r['p25']:.3f} | {r['p50']:.3f} | {r['p75']:.3f} "
            f"| {r['p90']:.3f} | {r['p95']:.3f} | {int(r['lt05'])} | {int(r['n_lt001'])}"
            f"({r['lt001']:.1%}) | {int(r['zero'])} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
