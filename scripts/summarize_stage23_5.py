#!/usr/bin/env python
"""ステージ23.5 B.3: 分母・手法を揃えた再推定の比較表を作る。

``results/stage23_5/estimates.csv`` から、科目 × 分母 × 手法の λ と、文献値との照合を
``results/stage23_5/comparison.md`` に書く。文献値は、本文または要旨で数値を確認できたもの
だけを載せる(``literature.md`` を参照)。

    uv run python scripts/summarize_stage23_5.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

_P = "jppfs_cor_"
# 科目 -> (文献値, 出典と確認の水準)。数値を確認できたものだけ
LITERATURE: dict[str, tuple[float, str]] = {
    "receivables_trade": (0.70, "Luo (2021) 要旨「approximately 70%」。分母は未確認"),
}
LABELS = {"assets": "総資産", "sales": "売上高", "cogs": "売上原価"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, default=Path("results/stage23_5/estimates.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/stage23_5/comparison.md"))
    args = parser.parse_args()
    e = pd.read_csv(args.estimates)
    sub = e[e["sample"] == "subsample"]
    lines = [
        "# ステージ23.5 B.3: 分母と手法を揃えた再推定",
        "",
        "パネル A のうち、すべての年で売上高・売上原価が取れた社(同じ部分標本)。目標は業種中央値。",
        "括弧内は標準誤差(企業クラスタ)。F は第 1 段階の F、Jp は Hansen 検定の p 値。",
        "",
        "## 符号を分けない",
        "",
        "| 科目 | 分母 | 手法 | λ (se) | F | Jp | 文献値 | 一致するか |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    both = sub[sub["direction"] == "both"]
    for (account, denom), g in both.groupby(["account", "denominator"], sort=False):
        short = account.removeprefix(_P)
        lit = LITERATURE.get(account)
        for method in ("fe", "diff_gmm", "sys_gmm"):
            r = g[g["method"] == method].iloc[0]
            f_stat = "-" if pd.isna(r["first_stage_f"]) else f"{r['first_stage_f']:.0f}"
            j_p = "-" if pd.isna(r["j_pval"]) else f"{r['j_pval']:.3f}"
            if lit is None:
                lit_s, match = "(数値を確認できる文献なし)", "-"
            else:
                inside = not math.isnan(r["lam"]) and r["ci_low"] <= lit[0] <= r["ci_high"]
                lit_s, match = f"{lit[0]:.2f}", "○(95% CI が含む)" if inside else "×"
            lines.append(
                f"| {short} | {LABELS[denom]} | {method} | {r['lam']:.3f} ({r['se']:.3f}) "
                f"| {f_stat} | {j_p} | {lit_s} | {match} |"
            )

    lines += [
        "",
        "## 運転資本 4 科目の順位(差分 GMM、λ の大きい順)",
        "",
    ]
    gmm = both[both["method"] == "diff_gmm"]
    for denom_set, title in (
        ({"assets"}, "すべて総資産比"),
        ({"sales"}, "すべて売上高比"),
    ):
        g = gmm[gmm["denominator"].isin(denom_set)].sort_values("lam", ascending=False)
        order = " > ".join(f"{r.account.removeprefix(_P)}({r.lam:.2f})" for r in g.itertuples())
        lines.append(f"- {title}: {order}")
    lines += [
        "",
        "## 符号別(差分 GMM)",
        "",
        "| 科目 | 分母 | λ+ (se) | n+ | λ− (se) | n− |",
        "|---|---|---|---:|---|---:|",
    ]
    d = sub[sub["method"] == "diff_gmm"]
    for (account, denom), g in d.groupby(["account", "denominator"], sort=False):
        p = g[g["direction"] == "positive"].iloc[0]
        n = g[g["direction"] == "negative"].iloc[0]
        lines.append(
            f"| {account.removeprefix(_P)} | {LABELS[denom]} | {p['lam']:.3f} ({p['se']:.3f}) "
            f"| {int(p['n_obs'])} | {n['lam']:.3f} ({n['se']:.3f}) | {int(n['n_obs'])} |"
        )

    pa = e[e["sample"] == "panel_a"]
    j_rej = int((pa["j_pval"] < 0.05).sum())
    lines += [
        "",
        "## システム GMM(パネル A 全社、総資産比、全 23 科目)",
        "",
        f"- Hansen 検定が 5% で棄却: {j_rej} / {len(pa)}",
        f"- 推定不能・標準誤差が非有限: {int((pa['status'] != 'ok').sum())}",
        "",
        "| 科目 | λ (se) | F | Jp | status |",
        "|---|---|---:|---:|---|",
    ]
    for r in pa.itertuples():
        lines.append(
            f"| {r.account.removeprefix(_P)} | {r.lam:.3f} ({r.se:.3f}) | {r.first_stage_f:.0f} "
            f"| {r.j_pval:.3f} | {r.status} |"
        )
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
