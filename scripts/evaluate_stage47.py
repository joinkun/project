#!/usr/bin/env python
"""ステージ47: 最終設定(資本金の下限 α = 0.00199)の確定。

前後比較、最終数値の表(正本)、実データとの比較。入力は
``metrics_stage39.py --gen-dir data/interim/stage47/gen`` の出力。``final``・``r1`` がステージ47 の
最終設定と R1、``s45``・``s45_r1`` はステージ45(α = 0)。

出力(``results/stage47/``): ``comparison.md``(B)、``final_table.md``(中間報告の正本:
指標と実データとの比較)、``summary.json``。

    uv run python scripts/evaluate_stage47.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage39 import boot, comp_level, corr, fmt_ci
from criteria_stage31 import wilson
from evaluate_stage34 import special_reason
from evaluate_stage43 import boot_diff, boot_median
from evaluate_stage45 import annual
from recovery_stage30 import changes
from run_stage25 import CONTINUOUS, summarize

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names

_M = Path("data/interim/stage47/metrics")
_GEN = Path("data/interim/stage47/gen")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2", "c1": "class 1"}
PAIRS = (("s45", "final", "最終設定(R3)"), ("s45_r1", "r1", "R1(感度分析)"))
_RE, _CS, _CSUR, _TS = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalStock",
                        "jppfs_cor_CapitalSurplus", "jppfs_cor_TreasuryStock")
GEN_YEARS, THR, HOLD, TOL, MATERIAL = 5.0, 0.001, 1e-6, 1e-4, 0.001


def load_gen(cls: str, s: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in
                   sorted((_GEN / cls / s).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS[cls])].reset_index(drop=True)


def firm_flags(flag: pd.Series, doc: pd.Series) -> np.ndarray:
    """3 seed の 2/3 以上で真の企業を 1、それ以外を 0 とした企業ごとの配列。"""
    f = pd.Series(np.asarray(flag), index=np.asarray(doc)).groupby(level=0).mean()
    out: np.ndarray = (f >= 2 / 3).astype(float).to_numpy()
    return out


def gen_capital(g: pd.DataFrame) -> dict[str, np.ndarray]:
    x0, x1 = g[f"x0_{_CS}"], g[f"x1_{_CS}"]
    held = x0 > HOLD
    doc = g.loc[held, "docID"]

    def below(r: float) -> pd.Series:
        out: pd.Series = (x1 < r * x0 - np.maximum(r * x0 * TOL, 1e-6))[held]
        return out

    t0, t1 = g[f"x0_{_TS}"].abs(), g[f"x1_{_TS}"].abs()
    th = t0 >= HOLD
    return {"half": firm_flags(below(0.5), doc), "lt1": firm_flags(below(0.01), doc),
            "zero": firm_flags((x1.abs() <= 1e-6)[held], doc),
            "ts": firm_flags(((t1 - t0) > MATERIAL)[th], g.loc[th, "docID"])}


def real_capital(codes: list[str]) -> dict[str, np.ndarray]:
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2023].set_index("edinetCode")
    b = p[p["t"] == 2024].set_index("edinetCode")
    cs = [c for c in codes if c in a.index and c in b.index]
    y0 = a.loc[cs, f"prev_{_CS}"].fillna(0).to_numpy()
    y1 = b.loc[cs, f"cur_{_CS}"].fillna(0).to_numpy()
    h = y0 > 0
    t0 = a.loc[cs, f"prev_{_TS}"].fillna(0).abs().to_numpy()
    t1 = b.loc[cs, f"cur_{_TS}"].fillna(0).abs().to_numpy()
    a0 = a.loc[cs, "prev_total_assets"].to_numpy()
    th = t0 >= HOLD * a0
    return {"half": (y1[h] < 0.5 * y0[h]).astype(float),
            "lt1": (y1[h] < 0.01 * y0[h]).astype(float),
            "zero": (y1[h] == 0).astype(float),
            "ts": ((t1 - t0) > MATERIAL * a0)[th].astype(float)}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage47"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    labels = {m: lab.replace("|", "\\|") for m, lab, _ in CONTINUOUS}
    summary: dict[str, object] = {}
    b_lines = ["# ステージ47 B: 全指標の前後比較(ステージ45 α = 0 → ステージ47 α = 0.00199)", "",
               "差は企業単位のペア(seed の中央値)の差の中央値 [95% ブートストラップ区間]。順位"
               "相関の差は企業の同時復元抽出。", ""]
    ctx: dict[str, dict[str, object]] = {}
    for cls in ("c2", "c1"):
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[~rows["docID"].isin(OUTLIERS[cls])]
        comp = comp_level(rows)
        viol = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        viol = viol[~viol["docID"].isin(OUTLIERS[cls])]
        dels = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        dels = dels[(dels["space"] == "native") & ~dels["docID"].isin(OUTLIERS[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]
        wide = {s: dels[dels["setting"] == s].pivot(index="docID", columns="account",
                                                    values="delta")[list(names)]
                for s in ("s45", "final", "s45_r1", "r1")}
        ctx[cls] = {"comp": comp, "viol": viol, "ref": ref, "wide": wide}
        for before, after, title in PAIRS:
            summ, pr = summarize(comp, (), (before, after), before)
            s_ = summ.set_index("version")
            pr = pr.set_index("metric")
            n = int(s_.loc[after, "n"])
            vb = int(viol[viol["setting"] == before]["rows"].sum())
            va = int(viol[viol["setting"] == after]["rows"].sum())
            cb, ca = corr(wide[before], ref), corr(wide[after], ref)
            _, _, dlo, dhi = boot(wide[after], ref, wide[before].loc[wide[after].index], ref, rng)
            b_lines += [f"## {title}、{CLS[cls]}({n} 社)", "",
                        "| 指標 | ステージ45(α = 0) | ステージ47(α = 0.00199) | 差 [95% CI] |",
                        "|---|---:|---:|---|",
                        f"| 15 項目の違反(行) | {vb} / {3 * n} | **{va} / {3 * n}** | — |"]
            for m in ("net_asset_improvement", "abs_asset_change", "sparsity",
                      "validity_lightgbm", "proximity_l2"):
                q = pr.loc[labels[m].replace("\\|", "|")]
                dg = 0 if m == "sparsity" else 3
                ci = fmt_ci(q["median_diff(version-equal)"], q["ci_low"], q["ci_high"])
                b_lines.append(f"| {labels[m]} | {s_.loc[before, m]:.{dg}f} "
                               f"| {s_.loc[after, m]:.{dg}f} | {ci} |")
                if m == "net_asset_improvement":
                    q2 = pr.loc["債務超過の解消率(社ごとの 3 seed 平均)"]
                    ci2 = fmt_ci(q2["median_diff(version-equal)"], q2["ci_low"], q2["ci_high"])
                    b_lines.append(
                        f"| 債務超過の解消(2/3 seed 以上) "
                        f"| {int(s_.loc[before, 'resolved_firms(>=2/3 seeds)'])}/"
                        f"{int(s_.loc[before, 'insolvent_start'])} "
                        f"| {int(s_.loc[after, 'resolved_firms(>=2/3 seeds)'])}/"
                        f"{int(s_.loc[after, 'insolvent_start'])} | 解消率 {ci2} |")
            b_lines += [f"| 符号付き順位相関 | {cb:+.3f} | {ca:+.3f} |"
                        f" {fmt_ci(ca - cb, dlo, dhi, 3)} |",
                        ""]
            summary[f"{cls}_{after}"] = {"violations": va, "signed_corr": ca,
                                         **{m: float(s_.loc[after, m]) for m in (
                                             "net_asset_improvement", "abs_asset_change",
                                             "sparsity", "validity_lightgbm", "proximity_l2")}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.md").write_text("\n".join(b_lines) + "\n", encoding="utf-8")

    # ---------------- final_table.md(正本) ----------------
    t_lines = ["# 最終設定の数値(中間報告 第 4 版の正本、ステージ47)", "",
               "最終設定: 等重み最近傍 + W1 + PCFM(K = 13、5-fold、50 epoch)+ R3 + 上限 γ = 0.75"
               "7 + 第 2 層の制約"
               "(小計の非負、純資産の非悪化、残余科目の下限、控除後純額の非負、β = 0.3、**資本"
               "金の下限 α = 0.00199**"
               "(資本金だけ)、**自己株式の上限**(分配可能額の代理))+ 不活性制約(規則 M、4 科目)+"
               " 目標 baseline。"
               "seed 0/1/2。評価は外れ値を除く class 2 55 社・class 1 131 社。", "",
               "## 1. 指標", "",
               "値は企業ごとの seed 中央値の中央値。[ ] は 95% 区間(企業の復元抽出 2,000 回、see"
               "d 0。解消は Wilson、"
               "順位相関は 1,000 回)。", "",
               "| 指標 | class 2(n = 55) | class 1(n = 131) | R1 class 2(感度分析) | R1 class 1 |",
               "|---|---|---|---|---|"]
    cells: dict[str, dict[str, str]] = {}
    for cls in ("c2", "c1"):
        for s in ("final", "r1"):
            c = ctx[cls]
            comp = c["comp"]
            assert isinstance(comp, pd.DataFrame)
            g = comp[comp["version"] == s].set_index("docID")
            viol = c["viol"]
            assert isinstance(viol, pd.DataFrame)
            n = len(g)
            n_viol = int(viol[viol["setting"] == s]["rows"].sum())
            d: dict[str, str] = {"15 項目の違反(行)": f"{n_viol}"
                                                    f" / {3 * n}"}
            for m, lab, dg in (("net_asset_improvement", "純資産の改善(総資産比、5 年)", 3),
                               ("abs_asset_change", "\\|総資産変化\\|", 3),
                               ("sparsity", "sparsity(総資産比 1% 超で動いた科目数)", 0),
                               ("validity_lightgbm", "validity LightGBM", 3),
                               ("proximity_l2", "proximity(等重み L2)", 3)):
                x = g[m].to_numpy()
                lo, hi = boot_median(x, rng)
                d[lab] = f"{np.median(x):.{dg}f} [{lo:.{dg}f}, {hi:.{dg}f}]"
                if m == "net_asset_improvement":
                    ins = g[g["insolvent_start"]]
                    k = int((ins["insolvency_resolved"] >= 2 / 3).sum())
                    wl_, wh_ = wilson(k, len(ins))
                    d["債務超過の解消(3 seed の 2/3 以上)"] = (
                        f"{k} / {len(ins)}({k / len(ins):.0%})[{wl_:.0%}, {wh_:.0%}]")
            w = c["wide"]
            ref = c["ref"]
            assert isinstance(w, dict) and isinstance(ref, pd.Series)
            cc = corr(w[s], ref)
            clo, chi, _, _ = boot(w[s], ref, None, None, rng)
            d["符号付き順位相関(実データ)"] = f"{cc:+.3f} [{clo:+.3f}, {chi:+.3f}]"
            cells[f"{cls}_{s}"] = d
    for lab in cells["c2_final"]:
        t_lines.append(f"| {lab} | {cells['c2_final'][lab]} | {cells['c1_final'][lab]} "
                       f"| {cells['c2_r1'][lab]} | {cells['c1_r1'][lab]} |")
    t_lines += ["", "符号付き順位相関の実データ: class 2 は再建群 15 社の 2 年、class 1 は class"
                    " 1 → class 0 の"
                "1 年の遷移 68 組。R1 = 利益剰余金を固定した感度分析"
                "(他の制約は最終設定と同じ)。", ""]

    # 実データとの比較
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wl = fy.pivot(index="edinetCode", columns="year", values="label")
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rec2 = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool)
              & (rf["group"] == "Y")].set_index("edinetCode")
    c1 = wl[wl[2022] == 1].copy()
    c1["reason"] = special_reason(fy, c1.index.tolist())
    c1 = c1.join(changes(_PANEL.parent, c1.index.tolist(), names), how="left")
    c1 = c1[(c1["reason"] == "") & c1["n_years"].notna()]
    rec1 = c1[wl.loc[c1.index, 2024] == 0]
    rec = {"c2": rec2, "c1": rec1}
    rec_lab = {"c2": f"class 2 → 注記なし({len(rec2)} 社)", "c1": f"class 1 → class 0"
                                                             f"({len(rec1)} 社)"}
    t_lines += ["## 2. 実データ(実際に再建した企業)との比較", "",
                "生成は t = 1(5 年)、実データは 2022 → 2024 年度末(2 年)。純資産・利益剰余金・"
                "払込資本は年率(生成 / 5、"
                "実データ / 2)、閾値 0.001。資本金・自己株式は期間を通した比較(生成 5 年、実デ"
                "ータ 2 年)。生成の企業は "
                "3 seed の 2/3 以上で数える。差の区間は、生成と実データを独立に復元抽出した 2,00"
                "0 回。", ""]
    for cls in ("c2", "c1"):
        g = load_gen(cls, "final")
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[(rows["version"] == "final") & ~rows["docID"].isin(OUTLIERS[cls])]
        gna = (rows.groupby("docID")["net_asset_improvement"].median() / GEN_YEARS).to_numpy()
        gre = annual(g, _RE).to_numpy()
        gpaid = (annual(g, _CS) + annual(g, _CSUR)).to_numpy()
        gcap = gen_capital(g)
        r = rec[cls]
        rcap = real_capital(list(r.index))
        t_lines += [f"### {CLS[cls]} 対 {rec_lab[cls]}", "",
                    "| 項目 | 生成 | 実データ(再建群) | 差(生成 − 実データ)[95% CI] |",
                    "|---|---:|---:|---|"]
        items = [("純資産の改善(年率、中央値)", gna, r["d_na"].dropna().to_numpy(), "median"),
                 ("利益剰余金が増えた企業の割合", gre, r["d_re"].dropna().to_numpy(), "share"),
                 ("払込資本を増やした企業の割合", gpaid, r["d_paid"].dropna().to_numpy(), "share"),
                 ("資本金を半減させた企業の割合", gcap["half"], rcap["half"], "share"),
                 ("資本金を始点の 1% 未満まで減らした割合", gcap["lt1"], rcap["lt1"], "share"),
                 ("資本金を 0 にした割合", gcap["zero"], rcap["zero"], "share"),
                 ("自己株式が増えた企業の割合(保有企業のみ)", gcap["ts"], rcap["ts"], "share")]
        for lab, a, b, stat in items:
            dd, lo, hi = boot_diff(a, b, stat, rng)
            if stat == "median":
                t_lines.append(f"| {lab} | {np.median(a):+.4f} | {np.median(b):+.4f} "
                               f"| {dd:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
            else:
                ka, kb = int((a > THR).sum()), int((b > THR).sum())
                t_lines.append(f"| {lab} | {ka / len(a):.0%}({ka}/{len(a)}) "
                               f"| {kb / len(b):.0%}({kb}/{len(b)}) | {dd:+.0%} [{lo:+.0%},"
                               f" {hi:+.0%}] |")
        t_lines.append("")
    t_lines += ["参考(苦境企業全体、class 1・2 の 2022 年度 171 社の 2 年): 資本金を半減 18.1%、"
                "1% 未満 2.3%、"
                "0 にした 0%、自己株式が増えた(保有 142 社)6.3%(ステージ44〜46)。", ""]
    (args.out / "final_table.md").write_text("\n".join(t_lines) + "\n", encoding="utf-8")
    summary["final_cells"] = cells
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n".join(b_lines + [""] + t_lines))


if __name__ == "__main__":
    main()
