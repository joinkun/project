#!/usr/bin/env python
"""ステージ22 C・D: 調整速度の推定結果を表にまとめる。

``results/stage22/estimates.csv`` から、主表(科目 × 目標 × 手法)、符号別の表(観測数つき)、
重みへの変換と破綻した科目、文献値との照合、パネル A / B の比較を ``results/stage22/report.md``
に書く。

    uv run python scripts/summarize_adjustment_stage22.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

_P = "jppfs_cor_"
_WEAK_F = 10.0  # 第 1 段階の F がこれ未満なら弱操作変数とみなす(Staiger & Stock 1997 の目安)

# (科目, 方向, 文献値の下限, 上限, 出典)。方向は both / positive(増やす)/ negative(減らす)
LITERATURE: tuple[tuple[str, str, float, float, str], ...] = (
    ("short_term_loans", "both", 0.27, 0.28, "岡本(2013)、日本上場企業"),
    (f"{_P}CapitalStock", "positive", 0.33, 0.33, "岡本(2013) 増資"),
    (f"{_P}CapitalStock", "negative", 0.037, 0.037, "岡本(2013) 減資"),
    ("receivables_trade", "both", 0.65, 0.70, "Luo(2021)、Abuhommous(2017)"),
    ("payables_trade", "both", 0.65, 0.70, "Luo(2021)、Abuhommous(2017)"),
)
WORKING_CAPITAL = (f"{_P}CashAndDeposits", "receivables_trade", "inventories", "payables_trade")


def _fmt(lam: float, se: float) -> str:
    return "-" if math.isnan(lam) else f"{lam:.3f} ({se:.3f})"


def _flag(r: pd.Series) -> str:
    marks = []
    if r["status"] != "ok":
        return str(r["status"])
    if not r["in_unit_interval"]:
        marks.append("範囲外")
    if r["method"] == "diff_gmm" and r["first_stage_f"] < _WEAK_F:
        marks.append("弱IV")
    if r["method"] == "diff_gmm" and r["j_pval"] == r["j_pval"] and r["j_pval"] < 0.05:
        marks.append("J棄却")
    return "、".join(marks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, default=Path("results/stage22/estimates.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/stage22/report.md"))
    args = parser.parse_args()
    e = pd.read_csv(args.estimates)
    e["account"] = (
        e["account"].str.removeprefix(_P).where(~e["account"].str.startswith(_P), e["account"])
    )
    e["short"] = e["account"].str.removeprefix(_P)
    e["flag"] = e.apply(_flag, axis=1)

    lines = [
        "# ステージ22: 部分調整モデルによる調整速度 λ の推定",
        "",
        "括弧内は標準誤差(企業クラスタ頑健、GMM は不均一分散頑健)。",
        f"印: 範囲外 = λ ∉ (0, 1]、弱IV = 第 1 段階の F < {_WEAK_F:.0f}、"
        "J棄却 = Hansen の過剰識別検定 p < 0.05。",
        "AR(2) 検定は、水準が 3 期(差分式が 1 本)しかないため計算できない。",
        "",
    ]

    for panel in ("A", "B"):
        a = e[(e["panel"] == panel) & (e["direction"] == "both")]
        lines += [
            f"## 主表(パネル {panel}、符号を分けない)",
            "",
            "| 科目 | 業種中央値・FE | 業種中央値・GMM | 企業特性・FE | 企業特性・GMM |",
            "|---|---|---|---|---|",
        ]
        for acc, g in a.groupby("short", sort=False):
            cells = []
            for target in ("industry", "firm"):
                for method in ("fe", "diff_gmm"):
                    r = g[(g["target"] == target) & (g["method"] == method)].iloc[0]
                    flag = f" {r['flag']}" if r["flag"] else ""
                    cells.append(_fmt(r["lam"], r["se"]) + flag)
            lines.append(f"| {acc} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## 符号別(パネル A)",
        "",
        "positive = 目標より下(増やす方向)、negative = 目標より上(減らす方向)。"
        "n は推定に使った観測数(FE は 2 年とも同じ群の社の観測、GMM は 2 年とも同じ符号の社数)。",
        "",
        "| 科目 | 目標 | 手法 | λ+ (se) | n+ | λ− (se) | n− |",
        "|---|---|---|---|---:|---|---:|",
    ]
    a = e[e["panel"] == "A"]
    for acc, g in a.groupby("short", sort=False):
        for target in ("industry", "firm"):
            for method in ("fe", "diff_gmm"):
                sub = g[(g["target"] == target) & (g["method"] == method)].set_index("direction")
                pos, neg = sub.loc["positive"], sub.loc["negative"]
                lines.append(
                    f"| {acc} | {target} | {method} "
                    f"| {_fmt(pos['lam'], pos['se'])} {pos['flag']} | {int(pos['n_obs'])} "
                    f"| {_fmt(neg['lam'], neg['se'])} {neg['flag']} | {int(neg['n_obs'])} |"
                )
    small = a[(a["direction"] != "both") & (a["n_obs"] < 100)]
    lines += [
        "",
        f"観測数が 100 未満の群: {len(small)} 件"
        + (
            "("
            + "、".join(
                f"{r.short}/{r.target}/{r.method}/{r.direction}={int(r.n_obs)}"
                for r in small.itertuples()
            )
            + ")"
            if len(small)
            else ""
        ),
        "",
    ]

    lines += ["## 重みへの変換 c = (1 − λ) / λ と破綻した科目(パネル A、符号を分けない)", ""]
    a = e[(e["panel"] == "A") & (e["direction"] == "both")]
    lines += [
        "| 科目 | 業種・GMM の c | 企業・GMM の c | 破綻(範囲外・推定不能)の組 |",
        "|---|---:|---:|---|",
    ]
    for acc, g in a.groupby("short", sort=False):
        c = {}
        for target in ("industry", "firm"):
            r = g[(g["target"] == target) & (g["method"] == "diff_gmm")].iloc[0]
            c[target] = "-" if pd.isna(r["cost"]) else f"{r['cost']:.2f}"
        broken = g[(g["status"] != "ok") | ~g["in_unit_interval"]]
        desc = "、".join(
            f"{r.target}/{r.method}(λ={r.lam:.2f})"
            if r.status == "ok"
            else f"{r.target}/{r.method}({r.status})"
            for r in broken.itertuples()
        )
        lines.append(f"| {acc} | {c['industry']} | {c['firm']} | {desc} |")
    n_fe = int(((a["method"] == "fe") & ~a["in_unit_interval"]).sum())
    n_gmm = int(((a["method"] == "diff_gmm") & ~a["in_unit_interval"]).sum())
    lines += [
        "",
        f"λ ∉ (0, 1] の組: FE {n_fe} / {int((a['method'] == 'fe').sum())}、"
        f"GMM {n_gmm} / {int((a['method'] == 'diff_gmm').sum())}(推定不能を含む)",
        "",
    ]

    lines += [
        "## 文献値との照合(パネル A)",
        "",
        "| 科目 | 方向 | 文献値 | 出典 | 業種・FE | 業種・GMM | 企業・FE | 企業・GMM |",
        "|---|---|---|---|---|---|---|---|",
    ]
    a = e[e["panel"] == "A"]
    for acc, direction, lo, hi, src in LITERATURE:
        short = acc.removeprefix(_P)
        g = a[(a["short"] == short) & (a["direction"] == direction)]
        cells = []
        for target in ("industry", "firm"):
            for method in ("fe", "diff_gmm"):
                r = g[(g["target"] == target) & (g["method"] == method)].iloc[0]
                inside = (
                    ""
                    if math.isnan(r["lam"])
                    else (" ✓CI" if r["ci_low"] <= hi and lo <= r["ci_high"] else "")
                )
                cells.append(_fmt(r["lam"], r["se"]) + inside)
        lit = f"{lo}" if lo == hi else f"{lo}〜{hi}"
        lines.append(f"| {short} | {direction} | {lit} | {src} | " + " | ".join(cells) + " |")
    lines += ["", "✓CI = 95% 信頼区間が文献値(の範囲)と重なる。", ""]
    lines += ["運転資本 4 科目の順位(λ の大きい順、符号を分けない、パネル A):", ""]
    b = a[(a["direction"] == "both") & a["account"].isin(WORKING_CAPITAL)]
    for target in ("industry", "firm"):
        for method in ("fe", "diff_gmm"):
            g = b[(b["target"] == target) & (b["method"] == method)].sort_values(
                "lam", ascending=False
            )
            order = " > ".join(f"{r.short}({r.lam:.2f})" for r in g.itertuples())
            lines.append(f"- {target}・{method}: {order}")
    lines += ["", "文献(Seidu 2022): 現金が最速、棚卸資産が最遅。", ""]

    lines += [
        "## パネル A(不一致を含む)と B(除く)の比較(符号を分けない)",
        "",
        "| 目標 | 手法 | |λ_A − λ_B| の中央値 | 最大 | 最大の科目 | 差 > 2×se_A の科目数 |",
        "|---|---|---:|---:|---|---:|",
    ]
    both = e[e["direction"] == "both"].pivot_table(
        index=["target", "method", "short"], columns="panel", values=["lam", "se"]
    )
    for (target, method), g in both.groupby(level=[0, 1]):
        d = (g[("lam", "A")] - g[("lam", "B")]).abs().dropna()
        big = int((d > 2 * g[("se", "A")].reindex(d.index)).sum())
        worst = d.idxmax()[2]
        lines.append(
            f"| {target} | {method} | {d.median():.3f} | {d.max():.3f} | {worst} "
            f"| {big} / {len(d)} |"
        )
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
