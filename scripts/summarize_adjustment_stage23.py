#!/usr/bin/env python
"""ステージ23 C・D: T=4(水準 4 期)の推定結果を、T=3(ステージ22)と並べて判定する。

主たる推定は「業種中央値の目標 × 差分 GMM」。固定効果は参考、企業特性の目標は感度分析。

D.1 の条件(すべて満たす科目だけ costs.yaml の更新候補):
1. λ ∈ (0, 1]
2. 第 1 段階の F ≥ 10
3. Hansen 検定が棄却されない(p ≥ 0.05)
4. 文献値と大きく矛盾しない。定義: 文献値のある科目は 95% 信頼区間が文献値の範囲と重なる。
   Seidu(2022) の順位だけが根拠の現金・棚卸資産は、運転資本 4 科目の中で現金が最速・
   棚卸資産が最遅であること。文献値の無い科目は「該当なし」として通す

    uv run python scripts/summarize_adjustment_stage23.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

_P = "jppfs_cor_"
_WEAK_F = 10.0
_J_ALPHA = 0.05
_MAIN = ("industry", "diff_gmm")

# (科目, 方向, 下限, 上限, 出典)
LITERATURE: tuple[tuple[str, str, float, float, str], ...] = (
    ("short_term_loans", "both", 0.27, 0.28, "岡本(2013)、日本上場企業"),
    ("CapitalStock", "positive", 0.33, 0.33, "岡本(2013) 増資"),
    ("CapitalStock", "negative", 0.037, 0.037, "岡本(2013) 減資"),
    ("receivables_trade", "both", 0.65, 0.70, "Luo(2021)、Abuhommous(2017)"),
    ("payables_trade", "both", 0.65, 0.70, "Luo(2021)、Abuhommous(2017)"),
)
WORKING_CAPITAL = ("CashAndDeposits", "receivables_trade", "payables_trade", "inventories")


def _load(path: Path) -> pd.DataFrame:
    e = pd.read_csv(path)
    e["short"] = e["account"].str.removeprefix(_P)
    return e


def _fmt(lam: float, se: float) -> str:
    return "-" if math.isnan(lam) else f"{lam:.3f} ({se:.3f})"


def _row(e: pd.DataFrame, short: str, target: str, method: str, direction: str) -> pd.Series:
    g = e[
        (e["short"] == short)
        & (e["target"] == target)
        & (e["method"] == method)
        & (e["direction"] == direction)
    ]
    return g.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t4", type=Path, default=Path("results/stage23/estimates.csv"))
    parser.add_argument("--t3", type=Path, default=Path("results/stage22/estimates.csv"))
    parser.add_argument("--panel", default="A")
    parser.add_argument("--out", type=Path, default=Path("results/stage23/report.md"))
    args = parser.parse_args()
    e4 = _load(args.t4)
    e3 = _load(args.t3)
    e4p = e4[e4["panel"] == args.panel]
    e3p = e3[e3["panel"] == args.panel]
    accounts = list(dict.fromkeys(e4p["short"]))
    target, method = _MAIN

    lines = [
        "# ステージ23: 水準 4 期(差分 3 期)での調整速度 λ の再推定",
        "",
        f"パネル {args.panel}。主たる推定は業種中央値の目標 × 差分 GMM(差分式 2 本を縦に積み、"
        "t-2 以前の水準を期ごとの操作変数にする。重み行列と標準誤差は企業クラスタ)。",
        "括弧内は標準誤差。AR(2) 検定は水準 5 期以上が必要で、計算できない。",
        "",
        "## 主表(符号を分けない)",
        "",
        "| 科目 | GMM λ (se) | F | J の自由度 | J の p | FE λ (se) | 企業特性・GMM λ (se) |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for acc in accounts:
        g = _row(e4p, acc, target, method, "both")
        fe = _row(e4p, acc, target, "fe", "both")
        firm = _row(e4p, acc, "firm", method, "both")
        j_df = "-" if pd.isna(g["j_df"]) else str(int(g["j_df"]))
        j_p = "-" if pd.isna(g["j_pval"]) else f"{g['j_pval']:.3f}"
        f_stat = "-" if pd.isna(g["first_stage_f"]) else f"{g['first_stage_f']:.1f}"
        status = "" if g["status"] == "ok" else f" {g['status']}"
        lines.append(
            f"| {acc} | {_fmt(g['lam'], g['se'])}{status} | {f_stat} | {j_df} | {j_p} "
            f"| {_fmt(fe['lam'], fe['se'])} | {_fmt(firm['lam'], firm['se'])} |"
        )
    n_out = {
        m: int(
            ((e4p["method"] == m) & (e4p["direction"] == "both") & ~e4p["in_unit_interval"]).sum()
        )
        for m in ("fe", "diff_gmm")
    }
    n_all = {m: int(((e4p["method"] == m) & (e4p["direction"] == "both")).sum()) for m in n_out}
    lines += [
        "",
        f"λ ∉ (0, 1](推定不能を含む、符号を分けない、目標 2 通り): "
        f"FE {n_out['fe']} / {n_all['fe']}、GMM {n_out['diff_gmm']} / {n_all['diff_gmm']}",
        "",
        "## 符号別(業種中央値の目標)",
        "",
        "n は推定に使った観測数(GMM は差分式の数)。",
        "",
        "| 科目 | GMM λ+ (se) | n+ | GMM λ− (se) | n− | FE λ+ | FE λ− |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for acc in accounts:
        gp, gn = (_row(e4p, acc, target, method, d) for d in ("positive", "negative"))
        fp, fn = (_row(e4p, acc, target, "fe", d) for d in ("positive", "negative"))
        lines.append(
            f"| {acc} | {_fmt(gp['lam'], gp['se'])} | {int(gp['n_obs'])} "
            f"| {_fmt(gn['lam'], gn['se'])} | {int(gn['n_obs'])} "
            f"| {fp['lam']:.3f} | {fn['lam']:.3f} |"
        )

    lines += [
        "",
        "## T=3(ステージ22)と T=4 の比較(業種中央値の目標、符号を分けない)",
        "",
        "| 科目 | GMM λ(T=3) | GMM λ(T=4) | 差 | se(T=4) | 差/se | FE λ(T=3) | FE λ(T=4) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    diffs = []
    for acc in accounts:
        a3, a4 = _row(e3p, acc, target, method, "both"), _row(e4p, acc, target, method, "both")
        f3, f4 = _row(e3p, acc, target, "fe", "both"), _row(e4p, acc, target, "fe", "both")
        d = a4["lam"] - a3["lam"]
        ratio = d / a4["se"] if a4["se"] and not math.isnan(a4["se"]) else math.nan
        diffs.append((acc, d, ratio, a3, a4))
        lines.append(
            f"| {acc} | {a3['lam']:.3f} | {a4['lam']:.3f} | {d:+.3f} | {a4['se']:.3f} "
            f"| {ratio:+.2f} | {f3['lam']:.3f} | {f4['lam']:.3f} |"
        )
    valid = [x for x in diffs if not math.isnan(x[1])]
    out3 = sum(1 for x in valid if not x[3]["in_unit_interval"])
    out4 = sum(1 for x in valid if not x[4]["in_unit_interval"])
    big = sum(1 for x in valid if abs(x[2]) > 2)
    med = pd.Series([abs(x[1]) for x in valid]).median()
    se3 = pd.Series([x[3]["se"] for x in valid]).median()
    se4 = pd.Series([x[4]["se"] for x in valid]).median()
    weak3 = sum(1 for x in valid if x[3]["first_stage_f"] < _WEAK_F)
    weak4 = sum(1 for x in valid if x[4]["first_stage_f"] < _WEAK_F)
    lines += [
        "",
        f"- |λ(T=4) − λ(T=3)| の中央値: {med:.3f}、|差| > 2 se の科目: {big} / {len(valid)}",
        f"- λ ∉ (0, 1] の科目(GMM・業種): T=3 {out3} → T=4 {out4}",
        f"- 標準誤差の中央値: T=3 {se3:.3f} → T=4 {se4:.3f}",
        f"- 弱操作変数(F < {_WEAK_F:.0f})の科目: T=3 {weak3} → T=4 {weak4}",
        "",
        "## 文献値との照合",
        "",
        "| 科目 | 方向 | 文献値 | 出典 | GMM T=3 | GMM T=4 | 95% CI(T=4) | 重なる |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    lit_ok: dict[str, bool] = {}
    for acc, direction, lo, hi, src in LITERATURE:
        a3 = _row(e3p, acc, target, method, direction)
        a4 = _row(e4p, acc, target, method, direction)
        overlap = not math.isnan(a4["lam"]) and a4["ci_low"] <= hi and lo <= a4["ci_high"]
        if direction == "both":
            lit_ok[acc] = overlap
        lit = f"{lo}" if lo == hi else f"{lo}〜{hi}"
        lines.append(
            f"| {acc} | {direction} | {lit} | {src} | {a3['lam']:.3f} | {a4['lam']:.3f} "
            f"| [{a4['ci_low']:.2f}, {a4['ci_high']:.2f}] | {'○' if overlap else '×'} |"
        )
    wc = {acc: _row(e4p, acc, target, method, "both")["lam"] for acc in WORKING_CAPITAL}
    wc3 = {acc: _row(e3p, acc, target, method, "both")["lam"] for acc in WORKING_CAPITAL}
    order4 = sorted(wc, key=lambda k: -wc[k])
    order3 = sorted(wc3, key=lambda k: -wc3[k])
    cash_fastest = order4[0] == "CashAndDeposits"
    inv_slowest = order4[-1] == "inventories"
    lines += [
        "",
        "運転資本 4 科目の λ の順位(大きいほど速い)。"
        "文献(Seidu 2022): 現金が最速、棚卸資産が最遅。",
        "",
        "- T=3: " + " > ".join(f"{k}({wc3[k]:.2f})" for k in order3),
        "- T=4: " + " > ".join(f"{k}({wc[k]:.2f})" for k in order4),
        f"- T=4 で現金が最速: {'○' if cash_fastest else '×'}、棚卸資産が最遅: "
        f"{'○' if inv_slowest else '×'}",
        "",
    ]
    lit_ok["CashAndDeposits"] = cash_fastest
    lit_ok["inventories"] = inv_slowest

    lines += [
        "## D.1 costs.yaml の更新候補(業種中央値 × GMM、符号を分けない)",
        "",
        "| 科目 | λ | c = (1−λ)/λ | ① λ∈(0,1] | ② F≥10 | ③ J 非棄却 | ④ 文献と整合 | 候補 |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    n_pass = 0
    for acc in accounts:
        g = _row(e4p, acc, target, method, "both")
        c1 = bool(g["in_unit_interval"])
        c2 = not pd.isna(g["first_stage_f"]) and g["first_stage_f"] >= _WEAK_F
        c3 = pd.isna(g["j_pval"]) or g["j_pval"] >= _J_ALPHA
        c4 = lit_ok.get(acc)
        c4_ok = True if c4 is None else c4
        passed = g["status"] == "ok" and c1 and c2 and c3 and c4_ok
        n_pass += passed
        cost = "-" if pd.isna(g["cost"]) else f"{g['cost']:.2f}"
        lines.append(
            f"| {acc} | {g['lam']:.3f} | {cost} | {'○' if c1 else '×'} | {'○' if c2 else '×'} "
            f"| {'○' if c3 else '×'} | {'該当なし' if c4 is None else ('○' if c4 else '×')} "
            f"| {'○' if passed else ''} |"
        )
    lines += ["", f"4 条件をすべて満たす科目: {n_pass} / {len(accounts)}", ""]

    # パネル A / B
    both = e4[(e4["direction"] == "both") & (e4["target"] == target)].pivot_table(
        index=["method", "short"], columns="panel", values=["lam", "se"]
    )
    lines += [
        "## パネル A(不一致を含む)と B(除く)の比較(業種中央値の目標、符号を分けない)",
        "",
        "| 手法 | |λ_A − λ_B| の中央値 | 最大 | 最大の科目 | 差 > 2 se_A |",
        "|---|---:|---:|---|---:|",
    ]
    for m, g in both.groupby(level=0):
        d = (g[("lam", "A")] - g[("lam", "B")]).abs().dropna()
        big_ab = int((d > 2 * g[("se", "A")].reindex(d.index)).sum())
        lines.append(
            f"| {m} | {d.median():.3f} | {d.max():.3f} | {d.idxmax()[1]} | {big_ab} / {len(d)} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
