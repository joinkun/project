#!/usr/bin/env python
"""ステージ43: R1 の再計算(不活性制約あり)、class 1 の向きの診断、最終数値の確定。

新しい探索は行わない。入力は ``metrics_stage39.py --gen-dir data/interim/stage43/gen`` の出力
(``data/interim/stage43/metrics``)と生成の parquet。設定: ``final``(R3 + 不活性制約、最終設定)、
``r1_inactive``(R1 + 不活性制約)、``r1``(R1、不活性制約なし。class 2 はステージ31 と同一)。

出力(``results/stage43/``): ``r1.md``(A)、``direction.md``(B)、``final_table.md``(C.1・C.2)、
``summary.json``。

    uv run python scripts/evaluate_stage43.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from compare_stage39 import boot, comp_level, corr, fmt_ci
from criteria_stage31 import wilson
from evaluate_stage34 import special_reason
from persistence_tables_stage33 import spearman
from recovery_stage30 import changes
from run_stage25 import summarize

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import INACTIVE_ACCOUNTS

_M = Path("data/interim/stage43/metrics")
_GEN = Path("data/interim/stage43/gen")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
OUTLIERS = {"c2": ("S100TSS9", "S100TWFQ"), "c1": ("S100VGDJ",)}
CLS = {"c2": "class 2(55 社)", "c1": "class 1(131 社)"}
SETS = {"1": "final", "2": "r1_inactive", "3": "r1"}
SET_LABEL = {"1": "R3 + 不活性制約(最終設定)", "2": "R1 + 不活性制約(感度分析)",
             "3": "R1(不活性制約なし、参考)"}
_RE, _CS, _CSUR = ("jppfs_cor_RetainedEarnings", "jppfs_cor_CapitalStock",
                   "jppfs_cor_CapitalSurplus")
GEN_YEARS, THR, ZERO, NONZERO, SIGN0 = 5.0, 0.001, 1e-6, 1.001e-3, 1e-4
_B = 2000


def load_gen(cls: str, setting: str) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in
                   sorted((_GEN / cls / setting).glob("*__seed*.parquet"))])
    return g[(g["method"] == "PCFM") & ~g["docID"].isin(OUTLIERS[cls])].reset_index(drop=True)


def firm_annual(g: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    d = pd.DataFrame({n: g[f"x1_{n}"].to_numpy() - g[f"x0_{n}"].to_numpy() for n in names})
    d["docID"] = g["docID"].to_numpy()
    return d.groupby("docID").median() / GEN_YEARS


def real_annual(codes: list[str], names: tuple[str, ...]) -> pd.DataFrame:
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    pp = p[p["edinetCode"].isin(codes) & p["t"].isin([2023, 2024])]
    d = pd.DataFrame({n: (pp[f"cur_{n}"].fillna(0) - pp[f"prev_{n}"].fillna(0))
                      / pp["prev_total_assets"] for n in names})
    d["edinetCode"] = pp["edinetCode"].to_numpy()
    return d.groupby("edinetCode").sum() / 2.0


def sgn(v: float) -> int:
    return 0 if abs(v) < SIGN0 else int(np.sign(v))


def boot_median(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(_B, len(x)))
    lo, hi = np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def boot_diff(a: np.ndarray, b: np.ndarray, stat: str,
              rng: np.random.Generator) -> tuple[float, float, float]:
    """独立な 2 群(生成と実際)の差 a − b(中央値の差か、閾値超の割合の差)と 95% 区間。"""
    def f(x: np.ndarray) -> float:
        return float(np.median(x)) if stat == "median" else float(np.mean(x > THR))
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ia = rng.integers(0, len(a), size=(_B, len(a)))
    ib = rng.integers(0, len(b), size=(_B, len(b)))
    d = np.array([f(a[i]) - f(b[j]) for i, j in zip(ia, ib, strict=True)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f(a) - f(b), float(lo), float(hi)


def agreement(gen: pd.Series, real: pd.Series) -> tuple[int, int, int]:
    """(一致した科目, 両方が動いて一致した科目, どちらかが動いた科目)。"""
    both = [(sgn(gen[a]), sgn(real[a])) for a in gen.index]
    agree = sum(1 for x, y in both if x == y)
    moving = [(x, y) for x, y in both if x != 0 or y != 0]
    return agree, sum(1 for x, y in moving if x == y), len(moving)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/stage43"))
    ap.add_argument("--gen-dir", type=Path, default=None, help="生成の場所(ステージ45 で追加)")
    ap.add_argument("--metrics-dir", type=Path, default=None, help="指標の場所(ステージ45 で追加)")
    args = ap.parse_args()
    global _M, _GEN
    _M = args.metrics_dir or _M
    _GEN = args.gen_dir or _GEN
    rng = np.random.default_rng(0)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    ja: dict[str, str] = yaml.safe_load(
        Path("conf/accounts/labels_ja.yaml").read_text(encoding="utf-8"))["accounts"]
    summary: dict[str, object] = {}
    args.out.mkdir(parents=True, exist_ok=True)

    # 実データ: 再建群
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    rf = rf[(rf["reason"].fillna("") == "") & rf["in_panel"].astype(bool)]
    rec2 = rf[rf["group"] == "Y"].set_index("edinetCode")
    fy = pd.read_csv("data/interim/stage30/firm_years.csv", dtype={"edinetCode": str})
    wide = fy.pivot(index="edinetCode", columns="year", values="label")
    c1 = wide[wide[2022] == 1].copy()
    c1["reason"] = special_reason(fy, c1.index.tolist())
    c1 = c1.join(changes(_PANEL.parent, c1.index.tolist(), names), how="left")
    c1all = c1[(c1["reason"] == "") & c1["n_years"].notna()]
    rec1 = c1all[wide.loc[c1all.index, 2024] == 0]
    real_acc = {"c2": real_annual(list(rec2.index), names),
                "c1_all": real_annual(list(c1all.index), names),
                "c1": real_annual(list(rec1.index), names)}

    # ---------------- A. R1 ----------------
    a_lines = ["# ステージ43 A: R1 の再計算(不活性制約あり)", "",
               "1 − 2 が利益剰余金の扱い(R3 対 R1)による効果、2 − 3 が R1 での不活性制約の効果。"
               "指標は企業ごとの seed 中央値の中央値。外れ値を除く。", ""]
    judg = {}
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
        summ, _ = summarize(comp, (), tuple(SETS.values()), "final")
        s = summ.set_index("version")
        w = {k: dels[dels["setting"] == v].pivot(index="docID", columns="account",
                                                  values="delta")[list(names)]
             for k, v in SETS.items()}
        a_lines += [f"## A.3 {CLS[cls]}", "",
                    "| 設定 | 15 項目の違反(行) | 内訳 | 純資産の改善 | 債務超過の解消 "
                    "| \\|総資産変化\\| "
                    "| sparsity | validity LGBM | 符号付き順位相関 [95%] |",
                    "|---|---:|---|---:|---|---:|---:|---:|---|"]
        for k, v in SETS.items():
            vv = viol[viol["setting"] == v].groupby("check")["rows"].sum()
            c = corr(w[k], ref)
            lo, hi, _, _ = boot(w[k], ref, None, None, rng)
            r = s.loc[v]
            a_lines.append(
                f"| {k} {SET_LABEL[k]} | {int(vv.sum())} | "
                + ("、".join(f"{x} {int(n)}" for x, n in vv.items()) or "なし")
                + f" | {r['net_asset_improvement']:.3f} "
                f"| {int(r['resolved_firms(>=2/3 seeds)'])} / {int(r['insolvent_start'])} "
                f"| {r['abs_asset_change']:.3f} | {r['sparsity']:.0f} "
                f"| {r['validity_lightgbm']:.3f} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        a_lines += ["", "差(差の中央値 [95% CI]、順位相関は差 [95%]):", "",
                    "| 比較 | 純資産の改善 | 解消率 | \\|総資産変化\\| | sparsity | validity LGBM "
                    "| 順位相関 |", "|---|---|---|---|---|---|---|"]
        for a, b, name in (("1", "2", "R3 − R1(どちらも不活性制約あり)"),
                           ("2", "3", "R1: 不活性制約あり − なし"),
                           ("1", "3", "最終設定 − ステージ31 の R1(2 つの変更が混ざる)")):
            _su, pr = summarize(comp, (), (SETS[b], SETS[a]), SETS[b])
            pr = pr.set_index("metric")

            def cell(m: str, pr: pd.DataFrame = pr) -> str:
                q = pr.loc[m]
                return fmt_ci(q["median_diff(version-equal)"], q["ci_low"], q["ci_high"])

            ca, cb = corr(w[a], ref), corr(w[b].loc[w[a].index], ref)
            _, _, dlo, dhi = boot(w[a], ref, w[b].loc[w[a].index], ref, rng)
            a_lines.append(f"| {name} | {cell('純資産の改善')} "
                           f"| {cell('債務超過の解消率(社ごとの 3 seed 平均)')} "
                           f"| {cell('|総資産変化|')} | {cell('sparsity')} "
                           f"| {cell('validity LightGBM')} | {fmt_ci(ca - cb, dlo, dhi, 3)} |")
            if (a, b) == ("1", "2"):
                q = pr.loc["純資産の改善"]
                judg[cls] = float(q["median_diff(version-equal)"])
        # A.4 R1 での出現
        g3, g2 = load_gen(cls, "r1"), load_gen(cls, "r1_inactive")
        a_lines += ["", f"### A.4 R1 での「0 → 非0」({CLS[cls]}、5 年、seed × 企業の行)", "",
                    "| 科目 | R1(なし) | R1 + 不活性制約 |", "|---|---:|---:|"]
        for n in names:
            cells = []
            for g in (g3, g2):
                z = np.abs(g[f"x0_{n}"].to_numpy()) < ZERO
                e = z & (np.abs(g[f"x1_{n}"].to_numpy()) > NONZERO)
                cells.append((int(e.sum()), int(z.sum())))
            if cells[0][1] == 0 or (cells[0][0] == 0 and cells[1][0] == 0):
                continue
            mark = "(対象)" if n in INACTIVE_ACCOUNTS else ""
            a_lines.append(f"| {ja[n]}{mark} | " + " | ".join(
                f"{k / m:.0%}({k}/{m})" for k, m in cells) + " |")
        a_lines.append("(どちらの設定でも出現しない科目は省いた)")
        a_lines.append("")
        summary[f"r1_{cls}"] = {v: {m: float(s.loc[v, m]) for m in
                                    ("net_asset_improvement", "abs_asset_change", "sparsity",
                                     "validity_lightgbm")} for v in SETS.values()}
    (args.out / "r1.md").write_text("\n".join(a_lines) + "\n", encoding="utf-8")

    # ---------------- B. 向きの診断 ----------------
    gen = {cls: firm_annual(load_gen(cls, "final"), names) for cls in ("c2", "c1")}
    gm = {k: v.median() for k, v in gen.items()}
    rm = {k: v.median() for k, v in real_acc.items()}
    ga = {k: v.abs().median() for k, v in gen.items()}
    ra = {k: v.abs().median() for k, v in real_acc.items()}
    sym = {1: "+", -1: "−", 0: "0"}
    b_lines = ["# ステージ43 B: class 1 での向きの不一致", "",
               "生成は最終設定の t = 1 を 5 で割った年率、実際は 2022 → 2024 年度末(2 年)を"
               " 2 で割った年率。"
               f"企業ごとの値の中央値の符号(\\|x\\| < {SIGN0:g} は 0)。"
               "大きさは \\|Δx\\|/A の中央値(年率)。",
               f"実際: class 2 は再建群 {len(rec2)} 社、class 1 は 2022 年度に class 1 の "
               f"{len(c1all)} 社"
               f"(全体)と、そのうち 2024 年度に class 0 へ回復した {len(rec1)} 社"
               "(2022 年度 class 1 → 2024 年度 class 0 は 63 社で、パネルにあるのが 57 社)。", "",
               "## B.2 科目別の向き", "",
               "| 科目 | c2 生成 | c2 実際(再建) | 一致 | 大きさ 生成 / 実際 | c1 生成 "
               "| c1 実際(全体) "
               "| 一致 | c1 実際(回復) | 一致 | 大きさ 生成 / 実際(全体) | class 1 だけ逆 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    only_c1: list[str] = []
    for n in names:
        s2g, s2r = sgn(gm["c2"][n]), sgn(rm["c2"][n])
        s1g, s1a, s1r = sgn(gm["c1"][n]), sgn(rm["c1_all"][n]), sgn(rm["c1"][n])
        flag = s2g == s2r and s2g != 0 and s1g != s1a
        if flag:
            only_c1.append(n)
        b_lines.append(
            f"| {ja[n]} | {sym[s2g]} | {sym[s2r]} | {'○' if s2g == s2r else '×'} "
            f"| {ga['c2'][n]:.4f} / {ra['c2'][n]:.4f} | {sym[s1g]} | {sym[s1a]} "
            f"| {'○' if s1g == s1a else '×'} | {sym[s1r]} | {'○' if s1g == s1r else '×'} "
            f"| {ga['c1'][n]:.4f} / {ra['c1_all'][n]:.4f} | {'**○**' if flag else ''} |")
    ag = {k: agreement(gm[c], rm[k]) for k, c in (("c2", "c2"), ("c1_all", "c1"), ("c1", "c1"))}
    b_lines += ["", "一致した科目(全 32 / 両方が動いて一致 / どちらかが動いた科目): "
                f"class 2 {ag['c2'][0]} / {ag['c2'][1]} / {ag['c2'][2]}、"
                f"class 1(全体) {ag['c1_all'][0]} / {ag['c1_all'][1]} / {ag['c1_all'][2]}、"
                f"class 1(回復) {ag['c1'][0]} / {ag['c1'][1]} / {ag['c1'][2]}。", "",
                "class 1 だけ逆になる科目(class 2 では生成と再建群の向きが一致し、"
                "class 1 では生成と"
                "実際(全体)の向きが違う): " + ("、".join(ja[n] for n in only_c1) or "なし"), ""]
    # B.3
    diff_real = [n for n in names if sgn(rm["c2"][n]) * sgn(rm["c1_all"][n]) == -1]
    diff_real_rec = [n for n in names if sgn(rm["c2"][n]) * sgn(rm["c1"][n]) == -1]

    def rc(x: pd.Series, y: pd.Series) -> float:
        return spearman(x.to_numpy(), y.loc[x.index].to_numpy())

    c1_start = load_gen("c1", "final").groupby("docID").first()
    c2_start = load_gen("c2", "final").groupby("docID").first()
    rows_f = {cls: pd.read_parquet(_M / f"rows_{cls}.parquet") for cls in ("c2", "c1")}
    prox = {cls: r[(r["version"] == "final") & ~r["docID"].isin(OUTLIERS[cls])]
            .groupby("docID")["proximity_l2"].median().median() for cls, r in rows_f.items()}

    def eq(g: pd.DataFrame) -> float:
        from fsgen.accounting.plausibility import subtotal_members
        m = subtotal_members(spec, names)
        x0 = g[[f"x0_{n}" for n in names]].to_numpy()
        return float(np.median(x0 @ m["jppfs_cor_NetAssets"] / (x0 @ m["jppfs_cor_Assets"])))

    tot_g = {k: float(v.abs().sum(axis=1).median()) for k, v in gen.items()}
    tot_r = {k: float(v.abs().sum(axis=1).median()) for k, v in real_acc.items()}
    b_lines += [
        "## B.3 原因の検討", "",
        "### 実データの向きは class で違うか", "",
        "実際の向きが class 2 の再建群と class 1 全体で逆の科目: "
        + ("、".join(f"{ja[n]}(c2 {sym[sgn(rm['c2'][n])]}、c1 {sym[sgn(rm['c1_all'][n])]})"
                     for n in diff_real) or "なし") + "。",
        "class 2 の再建群と class 1 の回復群で逆の科目: "
        + ("、".join(ja[n] for n in diff_real_rec) or "なし") + "。", "",
        "### 符号付きの中央値(32 科目)の順位相関", "",
        "| 組 | 順位相関 |", "|---|---:|",
        f"| 生成 class 1 対 生成 class 2 | {rc(gm['c1'], gm['c2']):+.3f} |",
        f"| 実際 class 1(全体)対 実際 class 2(再建) | {rc(rm['c1_all'], rm['c2']):+.3f} |",
        f"| 実際 class 1(回復)対 実際 class 2(再建) | {rc(rm['c1'], rm['c2']):+.3f} |",
        f"| 生成 class 1 対 実際 class 2(再建) | {rc(gm['c1'], rm['c2']):+.3f} |",
        f"| 生成 class 1 対 実際 class 1(全体) | {rc(gm['c1'], rm['c1_all']):+.3f} |",
        f"| 生成 class 1 対 実際 class 1(回復) | {rc(gm['c1'], rm['c1']):+.3f} |",
        f"| 生成 class 2 対 実際 class 2(再建) | {rc(gm['c2'], rm['c2']):+.3f} |", "",
        "### 動く余地", "",
        "| 項目 | class 2 | class 1 |", "|---|---:|---:|",
        f"| 始点の自己資本比率(中央値) | {eq(c2_start.reset_index().assign(method='PCFM')):.3f} "
        f"| {eq(c1_start.reset_index().assign(method='PCFM')):.3f} |",
        f"| 生成の proximity(5 年、中央値) | {prox['c2']:.3f} | {prox['c1']:.3f} |",
        f"| 生成の Σ\\|Δx\\|/A(年率、企業の中央値) | {tot_g['c2']:.4f} | {tot_g['c1']:.4f} |",
        f"| 実際の Σ\\|Δx\\|/A(年率、企業の中央値) | {tot_r['c2']:.4f}(再建) "
        f"| {tot_r['c1_all']:.4f}(全体)・{tot_r['c1']:.4f}(回復) |", ""]
    (args.out / "direction.md").write_text("\n".join(b_lines) + "\n", encoding="utf-8")
    summary["direction"] = {"only_class1_reversed": only_c1, "agreement": ag,
                            "real_opposite_c2_c1all": diff_real}

    # ---------------- C.1・C.2 最終数値 ----------------
    c_lines = ["# ステージ43 C: 最終数値(中間報告 第 4 版)", "",
               "最終設定: 等重み最近傍 + W1 + PCFM + R3 + 上限 γ = 0.757 + 第 2 層の制約 "
               "+ α = 0.5 + β = 0.3 "
               "+ 不活性制約(規則 M、4 科目)+ 目標 baseline。seed 0/1/2。",
               "値は企業ごとの seed 中央値の中央値、[ ] は 95% 区間"
               "(企業の復元抽出 2,000 回、seed 0。"
               "解消は Wilson、順位相関は 1,000 回)。", "",
               "## C.1 最終設定の指標", "",
               "| 指標 | class 2(n = 55) | class 1(n = 131) |", "|---|---|---|"]
    final_cells: dict[str, dict[str, str]] = {}
    for cls in ("c2", "c1"):
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[(rows["version"] == "final") & ~rows["docID"].isin(OUTLIERS[cls])]
        comp = comp_level(rows).set_index("docID")
        viol = pd.read_csv(_M / f"viol_{cls}.csv", dtype={"docID": str})
        viol = viol[(viol["setting"] == "final") & ~viol["docID"].isin(OUTLIERS[cls])]
        dels = pd.read_parquet(_M / f"deltas_{cls}.parquet")
        dels = dels[(dels["space"] == "native") & (dels["setting"] == "final")
                    & ~dels["docID"].isin(OUTLIERS[cls])]
        refs = pd.read_parquet(_M / "refs.parquet")
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index(
            "account")["median"]
        wv = dels.pivot(index="docID", columns="account", values="delta")[list(names)]
        c = corr(wv, ref)
        clo, chi, _, _ = boot(wv, ref, None, None, rng)
        ins = comp[comp["insolvent_start"]]
        n_res = int((ins["insolvency_resolved"] >= 2 / 3).sum())
        wl, wh = wilson(n_res, len(ins))
        d: dict[str, str] = {"15 項目の違反(行)": f"0 / {3 * len(comp)}" if viol["rows"].sum() == 0
                             else f"{int(viol['rows'].sum())} / {3 * len(comp)}"}
        for m, lab, dg in (("net_asset_improvement", "純資産の改善(総資産比、5 年)", 3),
                           ("abs_asset_change", "\\|総資産変化\\|", 3),
                           ("sparsity", "sparsity(総資産比 1% 超で動いた科目数)", 0),
                           ("validity_lightgbm", "validity LightGBM", 3),
                           ("validity_random_forest", "validity RF", 3),
                           ("validity_logreg", "validity LogReg", 3),
                           ("proximity_l2", "proximity(等重み L2)", 3),
                           ("debt_reduction", "有利子負債の圧縮", 3)):
            x = comp[m].to_numpy()
            lo, hi = boot_median(x, rng)
            d[lab] = f"{np.median(x):.{dg}f} [{lo:.{dg}f}, {hi:.{dg}f}]"
            if m == "net_asset_improvement":
                d["債務超過の解消(3 seed の 2/3 以上)"] = (
                    f"{n_res} / {len(ins)}({n_res / len(ins):.0%}) [{wl:.0%}, {wh:.0%}]")
        d["符号付き順位相関(実データ)"] = f"{c:+.3f} [{clo:+.3f}, {chi:+.3f}]"
        final_cells[cls] = d
    for lab in final_cells["c2"]:
        c_lines.append(f"| {lab} | {final_cells['c2'][lab]} | {final_cells['c1'][lab]} |")
    c_lines += ["", "符号付き順位相関の実データ: class 2 は再建群 15 社の 2 年、"
                "class 1 は class 1 → class 0 の"
                "1 年の遷移 68 組(ステージ38 と同じ)。", "",
                "## C.2 実データ(実際に再建した企業)との比較(年率)", "",
                "生成は t = 1(5 年)/ 5、実際は 2022 → 2024 年度末(2 年)/ 2。閾値 0.001(年率)。"
                "差の区間は、生成と実際を独立に復元抽出した 2,000 回。向きは 32 科目の中央値の符号"
                f"(\\|x\\| < {SIGN0:g} は 0)。", ""]
    for cls, rec, lab in (("c2", rec2, f"class 2 → 注記なし({len(rec2)} 社)"),
                          ("c1", rec1, f"class 1 → class 0({len(rec1)} 社)")):
        g = load_gen(cls, "final")
        ga_ = firm_annual(g, names)
        rows = pd.read_parquet(_M / f"rows_{cls}.parquet")
        rows = rows[(rows["version"] == "final") & ~rows["docID"].isin(OUTLIERS[cls])]
        gna = rows.groupby("docID")["net_asset_improvement"].median() / GEN_YEARS
        gpaid = ga_[_CS] + ga_[_CSUR]
        items = [("純資産の改善(中央値)", gna.to_numpy(), rec["d_na"].to_numpy(), "median"),
                 ("利益剰余金が増えた社の割合", ga_[_RE].to_numpy(), rec["d_re"].to_numpy(),
                  "share"),
                 ("払込資本が増えた社の割合", gpaid.to_numpy(), rec["d_paid"].to_numpy(), "share")]
        c_lines += [f"### {CLS[cls]} 対 実際の再建群: {lab}", "",
                    "| 項目 | 生成 | 実際の再建群 | 差(生成 − 実際)[95%] |", "|---|---:|---:|---|"]
        for name, a, b, stat in items:
            dd, lo, hi = boot_diff(a, b, stat, rng)
            if stat == "median":
                c_lines.append(f"| {name} | {np.median(a):+.4f} | {np.median(b):+.4f} "
                               f"| {dd:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
            else:
                c_lines.append(f"| {name} | {np.mean(a > THR):.0%}"
                               f"({int((a > THR).sum())}/{len(a)}) "
                               f"| {np.mean(b > THR):.0%}({int((b > THR).sum())}/{len(b)}) "
                               f"| {dd:+.0%} [{lo:+.0%}, {hi:+.0%}] |")
        agr = agreement(ga_.median(), real_acc[cls].median())
        c_lines += [f"| 向きが一致した科目 | {agr[0]} / 32"
                    f"(両方が動いた科目では {agr[1]} / {agr[2]}) "
                    "| — | — |", ""]
    (args.out / "final_table.md").write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    summary["final_cells"] = final_cells
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                      default=str), encoding="utf-8")
    print("\n".join(a_lines + [""] + b_lines + [""] + c_lines))


if __name__ == "__main__":
    main()
