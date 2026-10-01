#!/usr/bin/env python
"""ステージ44 C.4: 分配可能額の代理(会社法 461 条の近似)の妥当性を実データで確かめる。

推定用パネル(ステージ23、同じ書類の前期末 → 当期末、1 年)で、自己株式が増えた組(\\|自己株式\\| の
増加 > 前期総資産 × 0.001)について、前期末の代理 D = max(利益剰余金 + 資本剰余金 + 自己株式, 0) と
増加額を比べる。判定(``results/stage44/criteria.md``): 増加額 > D の組が 5% 以下なら妥当。

あわせて: 利益剰余金が負の企業で自己株式が増えた件数、class 1・2 の企業(2022 年度)で
自己株式を保有する企業の
2 年での増加の割合(C.5 の実データ)、生成の始点で債務超過かつ D > 0 の企業(C.6 の包含の確認)。

    uv run python scripts/treasury_stage44.py --out results/stage44/treasury.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from criteria_stage31 import wilson

_RE, _CSUR, _TS = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalSurplus",
                   "jppfs_cor_TreasuryStock")
MATERIAL, VALID_MAX = 0.001, 0.05


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})
    p = p[(p["prev_total_assets"] > 0) & (p["cur_total_assets"] > 0)].reset_index(drop=True)
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    lab = fy.set_index(["edinetCode", "year"])["label"]
    p["label_prev"] = [lab.get((c, t - 1), np.nan) for c, t in zip(p["edinetCode"], p["t"],
                                                                   strict=True)]

    def col(side: str, n: str) -> np.ndarray:
        out: np.ndarray = p[f"{side}_{n}"].fillna(0.0).to_numpy()
        return out

    ts0, ts1 = col("prev", _TS), col("cur", _TS)
    a0 = p["prev_total_assets"].to_numpy()
    inc = np.abs(ts1) - np.abs(ts0)
    ev = inc > MATERIAL * a0
    d0 = np.maximum(col("prev", _RE) + col("prev", _CSUR) + ts0, 0.0)
    exceed = ev & (inc > d0)
    re_neg = col("prev", _RE) < 0
    press = p["label_prev"].isin([1, 2]).to_numpy()
    lines = ["# ステージ44 C.4: 分配可能額の代理の妥当性(実データ)", "",
             f"推定用パネル {p['edinetCode'].nunique():,} 社・{len(p):,} 組(1 年)。"
             "自己株式の増加 = "
             f"\\|自己株式\\| の増加 > 前期総資産 × {MATERIAL}。"
             "代理 D = max(利益剰余金 + 資本剰余金 + "
             "自己株式, 0)(前期末)。", "",
             "| 母集団 | 自己株式が増えた組 | うち増加額 > D | 割合 [Wilson 95%] "
             "| うち前期の利益剰余金 < 0 "
             "| そのうち増加額 > D |",
             "|---|---:|---:|---|---:|---:|"]
    res: dict[str, float] = {}
    for name, m in (("全体", np.ones(len(p), dtype=bool)), ("前年度 class 0", (p["label_prev"] == 0)
                                                           .to_numpy()),
                    ("前年度 class 1・2", press)):
        n_ev, n_ex = int((ev & m).sum()), int((exceed & m).sum())
        lo, hi = wilson(n_ex, n_ev) if n_ev else (float("nan"), float("nan"))
        lines.append(f"| {name} | {n_ev} | {n_ex} | {n_ex / n_ev if n_ev else float('nan'):.1%} "
                     f"[{lo:.1%}, {hi:.1%}] | {int((ev & m & re_neg).sum())} "
                     f"| {int((exceed & m & re_neg).sum())} |")
        res[name] = n_ex / n_ev if n_ev else float("nan")
    share = res["全体"]
    lines += ["", f"**判定: 増加額 > D の割合 {share:.1%}(全体)→ "
              + ("5% 以下で、近似は妥当**" if share <= VALID_MAX
                 else "5% を超え、近似は不正確。制約を課さない**"),
              ""]
    # 超えた組の内訳
    ex = p[exceed].assign(inc_ratio=inc[exceed] / a0[exceed], d_ratio=d0[exceed] / a0[exceed],
                          re_ratio=col("prev", _RE)[exceed] / a0[exceed])
    lines += ["増加額 > D の組の内訳(総資産比、中央値): 増加額 "
              f"{ex['inc_ratio'].median():.4f}、D {ex['d_ratio'].median():.4f}、前期の利益剰余金 "
              f"{ex['re_ratio'].median():+.4f}。D = 0 の組 "
              f"{int((ex['d_ratio'] == 0).sum())} / {len(ex)}。", ""]
    # 補足(判定には使わない): D が小さく、制約が実際に効きうる組に限った場合
    lines += ["補足(判定には使わない): D が小さい組(制約が効きうる)に限った場合の、"
              "増加額 > D の割合:", "",
              "| D / 前期総資産 | 自己株式が増えた組 | うち増加額 > D |", "|---|---:|---:|"]
    for thr in (0.0, 0.01, 0.05, 0.10):
        m = ev & (d0 <= thr * a0)
        lines.append(f"| ≤ {thr:g} | {int(m.sum())} | {int((m & exceed).sum())} |")
    lines.append("")
    # 補足: D = 0(上限で増加が禁じられる)の組で、実際に自己株式が増えた割合
    dz = d0 <= 0
    lo0, hi0 = wilson(int((dz & ev).sum()), int(dz.sum()))
    lines += [f"補足: D = 0 の組 {int(dz.sum()):,} のうち、自己株式が増えた組 "
              f"{int((dz & ev).sum())}"
              f"({(dz & ev).sum() / dz.sum():.1%}、[{lo0:.1%}, {hi0:.1%}])。D > 0 の組では "
              f"{(ev & ~dz).sum() / (~dz).sum():.1%}。", ""]
    # 利益剰余金が負で増えた件数(461 条に照らせば 0 に近いはず)
    lines += [f"利益剰余金が負(前期末)の組: {int(re_neg.sum()):,}、うち自己株式が増えた組 "
              f"{int((ev & re_neg).sum())}({(ev & re_neg).sum() / re_neg.sum():.1%})。"
              f"利益剰余金が非負の組では {(ev & ~re_neg).sum() / (~re_neg).sum():.1%}。", ""]
    # C.5 の実データ: class 1・2(2022 年度)で自己株式を保有する企業の 2 年の増加
    a = p[p["t"] == 2023].set_index("edinetCode")
    b = p[p["t"] == 2024].set_index("edinetCode")
    both = a.index.intersection(b.index)
    lab22 = np.array([lab.get((c, 2022), np.nan) for c in both])
    t22 = a.loc[both, f"prev_{_TS}"].fillna(0).to_numpy()
    t24 = b.loc[both, f"cur_{_TS}"].fillna(0).to_numpy()
    a22 = a.loc[both, "prev_total_assets"].to_numpy()
    hold = (np.abs(t22) >= 1e-6 * a22) & np.isin(lab22, [1, 2])
    up2 = hold & ((np.abs(t24) - np.abs(t22)) > MATERIAL * a22)
    lo, hi = wilson(int(up2.sum()), int(hold.sum()))
    lines += ["C.5 の実データ: class 1・2(2022 年度)で自己株式を保有する "
              f"{int(hold.sum())} 社のうち、2 年で "
              f"\\|自己株式\\| が総資産比 {MATERIAL} を超えて増えた企業 {int(up2.sum())} 社"
              f"({up2.sum() / hold.sum():.1%}、[{lo:.1%}, {hi:.1%}])。", ""]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
