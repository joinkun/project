#!/usr/bin/env python
"""ステージ56 A: 払込資本(資本金 + 資本剰余金)の 2 年の比の分布と、下限 θ の機械的な決定。

規則は ``results/stage56/criteria.md``(集計の前に記録)。比 = 払込資本(2024 年度末)/ 払込資本
(2022 年度末)、名目額、推定用パネル。θ = 再建群 72 社で減少した企業の比の p1(30 社未満なら
class 1+2 で減少した企業の p1)。

出力: ``results/stage56/paidin.md``・``paidin.csv``・``theta.json``。

    uv run python scripts/paidin_stage56.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from criteria_stage31 import wilson
from evaluate_stage34 import special_reason

_PANEL = Path("data/interim/stage23/panel/panel.csv")
_CS, _CSUR = "jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus"
QS = (5, 10, 25, 50, 75, 90, 95)
MIN_N = 30
THR = 0.001  # ステージ47 の年率の閾値(総資産比)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage56"))
    args = ap.parse_args()
    pan = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = pan[pan["t"] == 2022].set_index("edinetCode")
    b = pan[pan["t"] == 2024].set_index("edinetCode")
    paid_a = a[f"cur_{_CS}"].fillna(0) + a[f"cur_{_CSUR}"].fillna(0)
    paid_b = b[f"cur_{_CS}"].fillna(0) + b[f"cur_{_CSUR}"].fillna(0)
    grp = pd.read_csv("results/stage51/asset_change.csv", dtype={"edinetCode": str})
    groups = {g: grp[grp["group"] == g]["edinetCode"].tolist() for g in grp["group"].unique()}
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    c12 = wl[wl[2022].isin([1, 2])].index.tolist()
    c12 = [c for c in c12 if c in a.index and c in b.index]
    reason = special_reason(fy, c12)
    groups["class 1+2(2022 年度、除外規則の後)"] = [c for c in c12 if reason.loc[c] == ""]
    # 年率の変化(ステージ47 の定義: 各年の差 / 前期総資産 の和 / 2)
    d = pan[pan["t"].isin([2023, 2024])].copy()
    d["dp"] = ((d[f"cur_{_CS}"].fillna(0) + d[f"cur_{_CSUR}"].fillna(0)
                - d[f"prev_{_CS}"].fillna(0) - d[f"prev_{_CSUR}"].fillna(0))
               / d["prev_total_assets"])
    annual = d.groupby("edinetCode")["dp"].sum() / 2

    rows, lines = [], ["# ステージ56 A: 払込資本(資本金 + 資本剰余金)の 2 年の比", "",
                       "比 = 払込資本(2024 年度末)/ 払込資本(2022 年度末)。名目額、**2 年の変化*"
                       "*(5 年には換算しない)。"
                       "始点の払込資本が 0 以下の企業は除く。", "",
                       "## A.1 分布", "",
                       "| 集団 | 社数 | " + " | ".join(f"p{q}" for q in QS) + " |",
                       "|---|---:|" + "---:|" * len(QS)]
    ratios: dict[str, np.ndarray] = {}
    for g, codes in groups.items():
        cs = [c for c in codes if c in a.index and c in b.index]
        pa, pb = paid_a.loc[cs], paid_b.loc[cs]
        ok = pa > 0
        r = (pb[ok] / pa[ok]).to_numpy()
        ratios[g] = r
        lines.append(f"| {g} | {len(r)}(除外 {int((~ok).sum())}) | "
                     + " | ".join(f"{np.percentile(r, q):.3f}" for q in QS) + " |")
        for c, v in zip(np.array(cs)[ok.to_numpy()], r, strict=True):
            rows.append({"group": g, "edinetCode": c, "ratio": float(v),
                         "annual_d_paid": float(annual.get(c, np.nan))})
    lines += ["", "## A.2・A.3 減少と増加", "",
              "「減少」= 比 < 1、「増加」= 比 > 1。年率の定義(ステージ47、総資産比 ±0.001)も併"
              "記する。", "",
              "| 集団 | 減少 | 減少した企業の比(p5 / p25 / p50) | 比 < 0.5 | 比 < 0.1 | 増加 "
              "| 増加した企業の比(p50 / p75) | 年率で増加(> 0.001) [Wilson 95%] | 年率で減少(< −"
              "0.001) [Wilson 95%] |",
              "|---|---|---|---:|---:|---|---|---|---|"]
    summ: dict[str, object] = {}
    for g, r in ratios.items():
        dec, inc = r[r < 1], r[r > 1]
        an = np.array([annual.get(c, np.nan) for c in
                       [x["edinetCode"] for x in rows if x["group"] == g]])
        an = an[np.isfinite(an)]
        k_up, k_dn = int((an > THR).sum()), int((an < -THR).sum())
        wu, wd = wilson(k_up, len(an)), wilson(k_dn, len(an))
        dq = (" / ".join(f"{np.percentile(dec, q):.3f}" for q in (5, 25, 50)) if len(dec) else "—")
        iq = (" / ".join(f"{np.percentile(inc, q):.3f}" for q in (50, 75)) if len(inc) else "—")
        lines.append(f"| {g} | {len(dec)} / {len(r)}({len(dec) / len(r):.0%}) | {dq} "
                     f"| {np.mean(r < 0.5):.1%} | {np.mean(r < 0.1):.1%} "
                     f"| {len(inc)} / {len(r)}({len(inc) / len(r):.0%}) | {iq} "
                     f"| {k_up} / {len(an)}({k_up / len(an):.0%})[{wu[0]:.0%}, {wu[1]:.0%}] "
                     f"| {k_dn} / {len(an)}({k_dn / len(an):.0%})[{wd[0]:.0%}, {wd[1]:.0%}] |")
        summ[g] = {"n": len(r), "dec": len(dec), "inc": len(inc), "up_annual": k_up,
                   "down_annual": k_dn, "n_annual": len(an), "wilson_down": wd, "wilson_up": wu}
    # θ(criteria.md §2)
    rec = np.concatenate([ratios["再建群(class 2 → 回復)"], ratios["再建群(class 1 → class 0)"]])
    rec_dec = rec[rec < 1]
    if len(rec_dec) >= MIN_N:
        theta = float(np.percentile(rec_dec, 1))
        src = f"再建群 72 社で減少した {len(rec_dec)} 社の比の p1"
    else:
        c12d = ratios["class 1+2(2022 年度、除外規則の後)"]
        c12d = c12d[c12d < 1]
        theta = float(np.percentile(c12d, 1))
        src = (f"再建群で減少した企業が {len(rec_dec)} 社(< {MIN_N})なので、class 1+2 で減少した "
               f"{len(c12d)} 社の比の p1(再建群の減少の最小は {rec_dec.min():.4f})")
    lines += ["", "## A.4 θ(`criteria.md` §2 の規則)", "", f"**θ = {theta:.6f}**({src})。", ""]
    (args.out / "paidin.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out / "paidin.csv", index=False)
    (args.out / "theta.json").write_text(json.dumps({"theta": theta, "source": src, "groups": summ},
                                                    ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
