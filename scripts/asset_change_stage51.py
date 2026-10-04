#!/usr/bin/env python
"""ステージ51 A.1: 実データの総資産変化の分布(2022 → 2024 年度末、5 年に換算)。

集団と規則は ``results/stage51/criteria_v2.md``(集計の前に記録)。5 年換算は複利
(R5 = R2^(5/2))、線形(2 年の変化 × 2.5)は参考。生成(ステージ47 の最終設定、t = 1)の分布も並べる。

出力: ``results/stage51/asset_change.md``、``asset_change.csv``(企業ごと)、``asset_change.json``。

    uv run python scripts/asset_change_stage51.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evaluate_stage34 import special_reason

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import subtotal_members

_PANEL = Path("data/interim/stage23/panel/panel.csv")
_FY = Path("data/interim/stage30/firm_years.csv")
_REC = Path("results/stage30/recovery_firms.csv")
_GEN = Path("data/interim/stage47/gen")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
QS = (5, 10, 25, 50, 75, 90, 95)
N_HEALTHY = 500


def two_year_ratio(codes: list[str]) -> pd.Series:
    """A_2024 / A_2022(両年度の書類がある企業だけ)。"""
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2023].set_index("edinetCode")["prev_total_assets"]
    b = p[p["t"] == 2024].set_index("edinetCode")["cur_total_assets"]
    cs = [c for c in codes if c in a.index and c in b.index]
    r: pd.Series = (b.loc[cs] / a.loc[cs]).astype(float)
    return r[np.isfinite(r) & (r > 0)]


def boot_median_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(2000, len(x)))
    lo, hi = np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage51"))
    args = ap.parse_args()
    fy = pd.read_csv(_FY, dtype={"edinetCode": str})
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    rec = pd.read_csv(_REC, dtype={"edinetCode": str})
    rec = rec[(rec["reason"].fillna("") == "") & rec["in_panel"].astype(bool)]
    groups: dict[str, list[str]] = {
        "再建群(class 2 → 回復)": rec[rec["group"] == "Y"]["edinetCode"].tolist(),
        "非再建群(class 2 → class 2)": rec[rec["group"] == "Z"]["edinetCode"].tolist(),
    }
    c1 = wl[(wl[2022] == 1) & (wl[2024] == 0)].index.tolist()
    reason = special_reason(fy, c1)
    groups["再建群(class 1 → class 0)"] = [c for c in c1 if reason.loc[c] == ""]
    # 健全企業: 2022 年度 class 0、両年度がパネルにある企業を乱数で並べ、除外規則に当たらない先頭 
    # 500 社
    c0 = sorted(wl[wl[2022] == 0].index.tolist())
    c0 = list(two_year_ratio(c0).index)
    order = np.random.default_rng(0).permutation(len(c0))
    picked: list[str] = []
    start = 0
    while len(picked) < N_HEALTHY and start < len(order):
        batch = [c0[i] for i in order[start:start + 200]]
        rs = special_reason(fy, batch)
        picked += [c for c in batch if rs.loc[c] == ""]
        start += 200
    groups["健全企業(class 0、無作為)"] = picked[:N_HEALTHY]
    order_keys = ["再建群(class 2 → 回復)", "再建群(class 1 → class 0)",
                  "非再建群(class 2 → class 2)", "健全企業(class 0、無作為)"]

    rng = np.random.default_rng(0)
    rows, firm_rows = [], []
    summ: dict[str, dict[str, Any]] = {}
    for k in order_keys:
        r2 = two_year_ratio(groups[k])
        comp = r2.to_numpy() ** 2.5 - 1
        lin = (r2.to_numpy() - 1) * 2.5
        lo, hi = boot_median_ci(comp, rng)
        rows.append((k, len(groups[k]), len(r2), comp, lin, lo, hi))
        summ[k] = {"n": len(r2), "median": float(np.median(comp)), "ci": [lo, hi],
                   **{f"p{q}": float(np.percentile(comp, q)) for q in QS},
                   "median_linear": float(np.median(lin)),
                   "share_up": float(np.mean(comp > 0))}
        firm_rows.append(pd.DataFrame({"group": k, "edinetCode": r2.index,
                                       "ratio_2y": r2.to_numpy(),
                                       "d5_compound": comp, "d5_linear": lin}))

    # 生成(ステージ47 の最終設定、t = 1)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    m = subtotal_members(spec, names)["jppfs_cor_Assets"]
    gen: dict[str, np.ndarray] = {}
    for cls in ("c2", "c1"):
        g = pd.concat([pd.read_parquet(f) for f in sorted((_GEN / cls / "final").glob("*seed*.pa"
                                                                                      "rquet"))])
        g = g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS[cls])]
        a0 = g[[f"x0_{n}" for n in names]].to_numpy() @ m
        a1 = g[[f"x1_{n}" for n in names]].to_numpy() @ m
        d = pd.Series((a1 - a0) / a0, index=g["docID"].to_numpy()).groupby(level=0).median()
        gen[cls] = d.to_numpy()
        lo, hi = boot_median_ci(gen[cls], rng)
        summ[f"生成 {cls}"] = {"n": len(d), "median": float(d.median()), "ci": [lo, hi],
                             **{f"p{q}": float(np.percentile(d, q)) for q in QS},
                             "share_up": float(np.mean(d > 0))}

    lines = ["# ステージ51 A.1: 総資産の変化 ΔA / A0 の分布(5 年換算)", "",
             "実データは 2022 → 2024 年度末(2 年)の比 R2 を複利で 5 年に換算(R2^2.5 − 1)。"
             "生成はステージ47 の最終設定の t = 1(5 年)、企業ごとの seed 中央値。"
             "中央値の区間は企業の復元抽出 2,000 回(seed 0)。集団の定義は `criteria_v2.md`。", "",
             "| 集団 | 社数(パネルにある / 定義) | p5 | p10 | p25 | **p50** [95% 区間] | p75 | p"
             "90 | p95 "
             "| 増えた割合 | p50(線形換算) |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for k, n_def, n, comp, lin, lo, hi in rows:
        p = {q: np.percentile(comp, q) for q in QS}
        lines.append(f"| {k} | {n} / {n_def} | {p[5]:+.3f} | {p[10]:+.3f} | {p[25]:+.3f} "
                     f"| **{p[50]:+.3f}** [{lo:+.3f}, {hi:+.3f}] | {p[75]:+.3f} | {p[90]:+.3f} "
                     f"| {p[95]:+.3f} | {np.mean(comp > 0):.0%} | {np.median(lin):+.3f} |")
    for cls, lab in (("c2", "生成 class 2(55 社)"), ("c1", "生成 class 1(131 社)")):
        s = summ[f"生成 {cls}"]
        lines.append(f"| {lab} | {s['n']} | {s['p5']:+.3f} | {s['p10']:+.3f} | {s['p25']:+.3f} "
                     f"| **{s['median']:+.3f}** [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] "
                     f"| {s['p75']:+.3f} | {s['p90']:+.3f} | {s['p95']:+.3f} |"
                     f" {s['share_up']:.0%} | — |")
    # A.2 の判定
    rec_keys = order_keys[:2]
    zero_in = {k: summ[k]["ci"][0] <= 0 <= summ[k]["ci"][1] for k in rec_keys}
    centered = all(zero_in.values())
    lines += ["", "## A.2 判定(`criteria_v2.md` §2 の規則)", ""]
    for k in rec_keys:
        lines.append(f"- {k}: 中央値の 95% 区間が 0 を{'含む' if zero_in[k] else '含まない'}")
    verdict = ("中心が 0 付近: |総資産変化| を主指標に残す" if centered
               else "中心が 0 でない: 指標を変える(案 3)")
    lines += ["", f"**{verdict}**", ""]
    # 案 2(参考): 生成が再建群の p10〜p90 に入る割合
    lines += ["## 参考(案 2): 生成の ΔA / A0 が再建群の p10〜p90 に入る企業の割合", "",
              "| 生成 | 再建群(class 2 → 回復)の p10〜p90 | 再建群(class 1 → class 0)の p10〜p90 |",
              "|---|---|---|"]
    for cls in ("c2", "c1"):
        cells = []
        for k in rec_keys:
            lo_, hi_ = summ[k]["p10"], summ[k]["p90"]
            cells.append(f"[{lo_:+.3f}, {hi_:+.3f}] に"
                         f" {np.mean((gen[cls] >= lo_) & (gen[cls] <= hi_)):.0%}")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    lines.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "asset_change.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.concat(firm_rows).to_csv(args.out / "asset_change.csv", index=False)
    summ["centered"] = {"value": centered}
    (args.out / "asset_change.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
