#!/usr/bin/env python
"""ステージ82: 実験 1〜6 の表と ``report.md``(lightgbm・sklearn のプロセス)。

規則は ``results/stage82/criteria.md``。事前に ``stage82_unified_revision.py`` の
``costs,ot,qp,metrics,evaluate,full`` を実行する。数値は小数第 3 位、割合は % で小数第 1 位。

出力: ``results/stage82/report.md`` と、各表の CSV(``exp*_*.csv``)。

    uv run python scripts/stage82_final.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd
from stage76_feasibility_map import moves, panel_years
from stage82_common import COSTS, DATA, EXP1, EXP2, EXP3, FOLDS, OUT, Setting
from tables_stage69 import INTERIM, Src, metrics, setup_excl
from tilde_stage51 import load_tilde

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.accounting.plausibility import (
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.models.unit_costs import emergence_rate, inactive_mask, individual_theta, psi

_RE = "jppfs_cor_RetainedEarnings"
REP_DOC = "S100UUP0"  # 実験 3 の代表企業(ANAP、class 2、BS 表の 5 社の 1 社)


def f3(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.3f}"


def pct(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{100 * v:.1f}%"


def sym(s: list[int]) -> str:
    return " ".join({1: "+", -1: "−", 0: "0"}[int(v)] for v in s)


def exp_table(key: str, sets: dict[str, Setting]) -> tuple[list[str], pd.DataFrame]:
    """実験 1〜3 の主指標の表(D1 の fold 平均と追加の指標)。"""
    sm = json.loads((OUT / key / "summary.json").read_text(encoding="utf-8"))
    ex = pd.read_csv(OUT / key / "extras.csv").set_index(["setting", "cls"])
    rows = []
    for name in sets:
        s = sm["summary"][name]
        lf = [v for v in sm["res"][name]["L"].values() if v is not None and np.isfinite(v)]
        r: dict[str, Any] = {"setting": name,
                             "lam_folds": ",".join("—" if v is None else f"{v:g}"
                                                   for v in sm["res"][name]["star"].values()),
                             "L_mean": float(np.mean(lf)) if len(lf) == len(FOLDS) else np.nan}
        for c in ("c2", "c1"):
            a = s[c]
            if not a.get("ok"):
                continue
            r |= {f"V_{c}": a["viol_max"], f"dNA_{c}": a["nai"], f"dNA69_{c}": a["d69"]["nai"][0],
                  f"sign69_{c}": sym(a["sign69"]), f"Rsolv_{c}": a["res"], f"rho_{c}": a["rho"],
                  f"ess_{c}": a["ess"], f"paid_dec_{c}": a["paid_dec"],
                  f"paid_inc_{c}": a["paid_inc"], f"n_changed_{c}": a["n_changed"],
                  f"infl_{c}": a["infl"], f"zero_{c}": a["e_zero"],
                  f"signBase_{c}": sym(a["signR1"]) if "signR1" in a else "基準",
                  f"Pdist_{c}": ex.loc[(name, c), "p_distress"],
                  f"Rvoid_{c}": ex.loc[(name, c), "r_void"],
                  f"emerge_any_{c}": ex.loc[(name, c), "emerge_any"],
                  f"emerge_mat_{c}": ex.loc[(name, c), "emerge_mat"]}
        rows.append(r)
    df = pd.DataFrame(rows)
    lines = ["| 設定 | λ̃*_f | V c2 / c1 | ΔNA c2 / c1 | 対 69 の差 c2 / c1 | 記号(対 69) c2 / c1 "
             "| 記号(対 基準) c2 / c1 | R_solv c2 / c1 | ρ_rank c2 / c1 | P_distress c2 / c1 "
             "| R_void c2 / c1 | 発生率(≥10⁻³) c2 / c1 | ESS c2 / c1 | L_f |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|"]
    for _, r in df.iterrows():
        def two(k: str, fmt: Any = f3, r: Any = r) -> str:
            return f"{fmt(r.get(f'{k}_c2', np.nan))} / {fmt(r.get(f'{k}_c1', np.nan))}"
        lines.append(f"| {r['setting']} | {r['lam_folds']} | {r.get('V_c2', '—')} / "
                     f"{r.get('V_c1', '—')} | {two('dNA')} | {two('dNA69')} "
                     f"| {r.get('sign69_c2', '—')} / {r.get('sign69_c1', '—')} "
                     f"| {r.get('signBase_c2', '—')} / {r.get('signBase_c1', '—')} "
                     f"| {two('Rsolv')} | {two('rho')} | {two('Pdist', pct)} | {two('Rvoid', pct)} "
                     f"| {two('emerge_mat', pct)} | {two('ess')} | {f3(r['L_mean'])} |")
    df.to_csv(OUT / f"{key}_main.csv", index=False)
    return lines, df


def ratios(v: np.ndarray, m: dict[str, np.ndarray], idx: dict[str, int]
           ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """5 つの財務比率(分母の規則は criteria §7)。除外は NaN にし、件数を返す。"""
    a = v @ m["jppfs_cor_Assets"]
    na, ca, cl = v @ m["jppfs_cor_NetAssets"], v @ m["jppfs_cor_CurrentAssets"], \
        v @ m["jppfs_cor_CurrentLiabilities"]
    nca, ncl = v @ m["jppfs_cor_NoncurrentAssets"], v @ m["jppfs_cor_NoncurrentLiabilities"]
    debt = sum(v[:, idx[n]] for n in ("short_term_loans", "long_term_loans", "bonds_current",
                                      "bonds_noncurrent"))
    wc = v[:, idx["receivables_trade"]] + v[:, idx["inventories"]] - v[:, idx["payables_trade"]]
    cur_ok = cl >= 1e-3 * a
    fix_den = na + ncl
    fix_ok = fix_den > 1e-3 * a
    out = {"自己資本比率": na / a,
           "流動比率": np.where(cur_ok, ca / np.where(cur_ok, cl, 1.0), np.nan),
           "固定長期適合率": np.where(fix_ok, nca / np.where(fix_ok, fix_den, 1.0), np.nan),
           "ネット有利子負債/総資産": (debt - v[:, idx["jppfs_cor_CashAndDeposits"]]) / a,
           "運転資本/総資産": wc / a}
    return out, {"流動比率": int((~cur_ok).sum()), "固定長期適合率": int((~fix_ok).sum())}


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    idx = {n: i for i, n in enumerate(names)}
    setup_excl(names)
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    best = EXP2[sel["best"]]
    lam = float(sel["lambda_tilde"])
    tj = json.loads((DATA / "theta.json").read_text(encoding="utf-8"))
    rep: list[str] = ["# ステージ82 報告: 数理モデルの改訂と指導教員の指摘の検証", "",
                      "規則は `criteria.md`(コミット 41cf02a、計算の前に記録)。"
                      "数値は小数第 3 位、割合は % で小数第 1 位。",
                      "D1 = ステージ75 と同じ 5 fold の交差適合"
                      "(値は fold の平均、V は最大、ESS は中央値)。",
                      "記号 = 純資産の改善・解消率・順位相関(+ 優れる、− 劣る、0 差なし)。", "",
                      "<!-- 要約(手で書く) -->", ""]
    # 実験 1
    meta = json.loads((OUT / "exp1_theta_meta.json").read_text(encoding="utf-8"))
    corr = pd.read_csv(OUT / "exp1_theta_corr.csv", index_col=0)
    rep += ["## 実験 1: 基準単価 θ の 5 方式", "",
            f"θ の順位相関(57 社の全体、評価差額金を除く {meta['n_directions']} 方向、"
            "`exp1_theta_corr.csv`):",
            "", "| | " + " | ".join(corr.columns) + " |", "|---|" + "---:|" * len(corr.columns)]
    for k_, row in corr.iterrows():
        rep.append(f"| {k_} | " + " | ".join(f3(v) for v in row) + " |")
    rep += ["", "上限 10³ に当たる方向: " + "、".join(f"{k} {v}" for k, v in meta["capped"].items())
            + "。KDE0 で R1 に戻した方向(fold・全体): "
            + "、".join(f"{k} {v}" for k, v in meta["kde0_fallback"].items()) + "。", "",
            "D1(基準 = R1、`exp1/comparison_report.md`):", ""]
    lines, _ = exp_table("exp1", EXP1)
    rep += lines + ["", f"R1 か KDE0 か(criteria §6): KDE0 が R1 より優れる条件を"
                    f"{'満たす' if sel['kde0_over_r1'] else '満たさない'}。", ""]
    # 実験 2
    lines, _ = exp_table("exp2", EXP2)
    rep += ["## 実験 2: 個体別の単価 × 不活性科目の扱い(R1)", "", *lines, "",
            "主モデルの選択(criteria §6、`selection.json`):", "",
            "| 設定 | 条件 1 | 条件 2 | 条件 4 | 発生率(≥10⁻³、両 class の平均) |",
            "|---|---|---|---|---:|"]
    for n in EXP2:
        c = sel["conditions"][n]
        rep.append(f"| {n} | {'○' if c['c1'] else '×'} | {'○' if c['c2'] else '×'} "
                   f"| {'○' if c['c4'] else '×'} | {pct(sel['emerge_mat_mean'][n])} |")
    rep += ["", f"**選ばれた主モデル: {sel['best']}**(単一の生成の λ̃ = {lam:g})。", ""]
    # 実験 3
    lines, _ = exp_table("exp3", EXP3)
    diag = []
    for name, st in EXP3.items():
        for cls in ("c2", "c1"):
            js = [json.loads((OUT / "ot" / f"{cls}_{g}_{st.cost}_{st.ot}.json").read_text(
                encoding="utf-8")) for g in FOLDS]
            diag.append({"setting": name, "cls": cls,
                         "nz_mean": float(np.mean([j["nz_mean"] for j in js])),
                         "max_load": float(np.mean([j["max_load"] for j in js])),
                         "unique": float(np.mean([j["unique_targets"] for j in js])),
                         "ess": float(np.mean([j["ess_med"] for j in js])),
                         "gamma": js[0].get("gamma", np.nan)})
    dg = pd.DataFrame(diag)
    dg.to_csv(OUT / "exp3_ot_diag.csv", index=False)
    rep += ["## 実験 3: 輸送の方式のアブレーション(R1)", "",
            "輸送の診断(5 fold の平均。F1 に使う重みでの値):", "",
            "| 方式 | 非ゼロの候補数 c2 / c1 | 最大の列の負荷 c2 / c1 "
            "| 選ばれた健全企業の数 c2 / c1 "
            "| ESS c2 / c1 |", "|---|---|---|---|---|"]
    for name in EXP3:
        a, b = (dg[(dg["setting"] == name) & (dg["cls"] == c)].iloc[0] for c in ("c2", "c1"))
        rep.append(f"| {name} | {f3(a['nz_mean'])} / {f3(b['nz_mean'])} "
                   f"| {f3(a['max_load'])} / {f3(b['max_load'])} "
                   f"| {a['unique']:.1f} / {b['unique']:.1f} | {f3(a['ess'])} / {f3(b['ess'])} |")
    rep += ["", "D1(基準 = proposed_quad_unbalanced):", "", *lines, ""]
    wrows = []
    for name, st in EXP3.items():
        z = np.load(DATA / f"plan_c2_full_{st.cost}_{st.ot}.npz", allow_pickle=True)
        i = [str(v) for v in z["source_docs"]].index(REP_DOC)
        wrows.append({"setting": name,
                      **{f"w{k + 1}": float(z["top10_p"][i][k]) for k in range(10)}})
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(OUT / "exp3_rep_weights.csv", index=False)
    rep += [f"代表企業(ANAP、{REP_DOC}、class 2、57 社の全体の計画)の上位 10 候補の重み"
            "(行で正規化した p_ij、Sinkhorn はカットオフ前):", "",
            "| 方式 | " + " | ".join(f"{k}" for k in range(1, 11)) + " |", "|---|" + "---:|" * 10]
    for _, r in wdf.iterrows():
        rep.append(f"| {r['setting']} | " + " | ".join(f3(r[f'w{k}']) for k in range(1, 11)) + " |")
    rep.append("")
    # 単一の生成(主モデル)
    def src(cost: str, tier: str) -> Src:
        tag = Setting("full", cost, "prop", best.qp, tier).tag
        return Src(DATA / "full" / "metrics_full", DATA / "full" / "gen_full", f"{tag}__l{lam:g}",
                   f"{lam:g}")

    def arrays(s_: Src, cls: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        e51._GEN = s_.gen
        g = e51.load_gen(cls, s_.setting)
        g = g[g["seed"] == 0]
        return (g[[f"x0_{n}" for n in names]].to_numpy(float),
                g[[f"x1_{n}" for n in names]].to_numpy(float), [str(d) for d in g["docID"]])
    # 実験 4
    trows = []
    for tier, tl in (("t1", "tier1"), ("t12", "tier1+2"), ("t123", "tier1+2+3")):
        for cls in ("c2", "c1"):
            m = metrics(src(best.cost, tier), cls, names, members)
            trows.append({"tier": tl, "cls": cls, "V": m["viol"], "nai": m["nai"], "res": m["res"],
                          **{f"item_{k}": v for k, v in m["items"].items()}})
    td = pd.DataFrame(trows).fillna(0)
    td.to_csv(OUT / "exp4_tiers.csv", index=False)
    items = sorted([c for c in td.columns if c.startswith("item_")],
                   key=lambda c: int(c.split("_")[1]))
    labels = {k.split("_")[0]: k for k in (*CHECK_NAMES, *EXTRA_CHECK_NAMES)}
    rep += ["## 実験 4: 制約の 3 階層(主モデルの単一の生成)", "",
            "| 階層 | class | 違反の社数 V | ΔNA | R_solv | 項目別の件数(行 = 企業 × seed) |",
            "|---|---|---:|---:|---:|---|"]
    for _, r in td.iterrows():
        it = "、".join(f"{labels.get(c.split('_')[1], c)} {int(r[c])}" for c in items if r[c] > 0)
        rep.append(f"| {r['tier']} | {r['cls']} | {int(r['V'])} | {f3(r['nai'])} | {f3(r['res'])} "
                   f"| {it or 'なし'} |")
    rep.append("")
    # 実験 5
    gap = pd.read_csv(OUT / "exp5_survival_gap.csv")
    rep += ["## 実験 5: 生存バイアスと再建に要する期間の Tier", "",
            f"回復群 72 社と滞留群 {len(tj['pools']['stag'])} 社の R1 の θ の比"
            "(|log 比| の大きい順に 10 方向、`exp5_survival_gap.csv`):", "",
            "| 科目 | 向き | θ 回復群 | θ 滞留群 | 滞留 / 回復 | n 回復 / 滞留 |",
            "|---|---|---:|---:|---:|---|"]
    for _, r in gap.head(10).iterrows():
        rep.append(f"| {r['label']} | {r['dir']} | {f3(r['theta_rec72'])} | {f3(r['theta_stag'])} "
                   f"| {f3(r['ratio_stag_over_rec'])} | {r['n_rec']} / {r['n_stag']} |")
    cmp_rows = []
    for lab_, cost in (("回復群 57 社(主モデル)", best.cost),
                       ("苦境全体(回復 + 滞留)", f"{best.cost}_all")):
        for cls in ("c2", "c1"):
            s_ = src(cost, "t123")
            m = metrics(s_, cls, names, members)
            x0, y, _d = arrays(s_, cls)
            cmp_rows.append({"pool": lab_, "cls": cls, "V": m["viol"], "nai": m["nai"],
                             "res": m["res"], "rho": m["rho"], "paid_dec": m["paid_dec"],
                             "n_changed": m["n_changed"],
                             "emerge_mat": emergence_rate(x0, y, names, 1e-3)})
    cd = pd.DataFrame(cmp_rows)
    cd.to_csv(OUT / "exp5_alldist.csv", index=False)
    rep += ["", "単価を推定するプールの比較(単一の生成、同じ λ̃。class 1 の順位相関は標準の参照で、"
            "コストのプールと重なる):", "",
            "| プール | class | V | ΔNA | R_solv | ρ_rank | 払込資本↓ | 変更科目数 "
            "| 発生率(≥10⁻³) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in cd.iterrows():
        rep.append(f"| {r['pool']} | {r['cls']} | {r['V']} | {f3(r['nai'])} | {f3(r['res'])} "
                   f"| {f3(r['rho'])} | {pct(r['paid_dec'])} | {r['n_changed']:.0f} "
                   f"| {pct(r['emerge_mat'])} |")
    cs = COSTS[best.cost]
    sp = cs.mode
    th = np.array(tj["configs"]["full"][sp]["theta"], dtype=float)
    sigma = np.array(tj["sigma"], dtype=float)
    re_i = idx[_RE]

    def psi_i(x0: np.ndarray, d: np.ndarray) -> float:
        t = individual_theta(th, x0, sigma, indiv_scale=cs.indiv,
                             inactive=inactive_mask(x0, names) if cs.penalty else None)
        t[:, re_i] = 0.0
        return float(psi(t, d))
    comp, _ = panel_years(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    d72, c72 = moves(comp, 2022, 2024, pairs["edinetCode"].tolist())
    x72 = comp[2022].loc[c72].to_numpy()
    rc = np.array([psi_i(x72[k], d72[k]) for k in range(len(c72))])
    q = {p: float(np.quantile(rc, p)) for p in (0.5, 0.75, 0.9, 0.95)}
    tier_rows, mid_rows = [], []
    for cls in ("c2", "c1"):
        x0, y, _d = arrays(src(best.cost, "t123"), cls)
        g = np.array([psi_i(x0[i], y[i] - x0[i]) for i in range(len(x0))])
        ta, tb = g <= q[0.75], (g > q[0.75]) & (g <= q[0.95])
        tc = g > q[0.95]
        tier_rows.append({"cls": cls, "n": len(g), "A": int(ta.sum()), "B": int(tb.sum()),
                          "C": int(tc.sum()), "gen_q50": float(np.median(g))})
        if tc.any():
            xm = 0.5 * x0[tc] + 0.5 * y[tc]
            f8 = evaluate_checks(x0[tc], xm, spec, names)
            f7 = evaluate_extra_checks(xm, x0[tc], spec, names)
            fl = np.column_stack([f8, f7])
            fe = np.column_stack([evaluate_checks(x0[tc], y[tc], spec, names),
                                  evaluate_extra_checks(y[tc], x0[tc], spec, names)])
            mid_rows.append({"cls": cls, "n_C": int(tc.sum()), "V_mid": int(fl.any(axis=1).sum()),
                             "V_end": int(fe.any(axis=1).sum()),
                             "items_mid": "、".join(
                                 n for n, b in zip((*CHECK_NAMES, *EXTRA_CHECK_NAMES),
                                                   fl.any(axis=0), strict=True) if b) or "なし"})
    tdf = pd.DataFrame(tier_rows)
    tdf.to_csv(OUT / "exp5_tiers.csv", index=False)
    rep += ["", "再建群 72 社(2022 → 2024)の移動コスト ψ_k(主モデルの θ)の分位点: "
            + "、".join(f"q{p:.2f} = {f3(v)}" for p, v in q.items()) + "。", "",
            "| class | 社数 | Tier A(≤ q0.75)| Tier B(q0.75〜q0.95)| Tier C(> q0.95)"
            "| 生成の ψ の中央値 |", "|---|---:|---|---|---|---:|"]
    for _, r in tdf.iterrows():
        rep.append(f"| {r['cls']} | {r['n']} | {r['A']}({pct(r['A'] / r['n'])}) "
                   f"| {r['B']}({pct(r['B'] / r['n'])}) | {r['C']}({pct(r['C'] / r['n'])}) "
                   f"| {f3(r['gen_q50'])} |")
    rep += ["", "Tier の名前(短期 2〜3 年・中期 5 年・長期 10 年超)は解釈のラベルで、"
            "実際の年数ではない(criteria §0)。", "",
            "Tier C の中間点 y⁽⁰·⁵⁾ = 0.5 x + 0.5 ŷ の 15 項目のチェック:", "",
            "| class | Tier C の社数 | 中間点で違反した社数 | 終点で違反した社数 "
            "| 中間点の違反項目 |",
            "|---|---:|---:|---:|---|"]
    for r in mid_rows:
        rep.append(f"| {r['cls']} | {r['n_C']} | {r['V_mid']} | {r['V_end']} | {r['items_mid']} |")
    pd.DataFrame(mid_rows).to_csv(OUT / "exp5_midpoint.csv", index=False)
    rep.append("")
    # 実験 6
    x, lab, _docs, _ = load_tilde(names)
    xh = x[lab == 0]
    hr, _ = ratios(xh, members, idx)
    old = Src(INTERIM / "stage80/metrics_full", INTERIM / "stage80/gen_full", "A1_sp__l0.1", "0.1")
    rrows = []
    rep += ["## 実験 6: 5 つの財務比率", "",
            "健全レンジ = 健全企業(class 0、3,768 社)の [Q1, Q3](と [P10, P90])。"
            "適合率 = 範囲に入る割合"
            "(除外を除く)。除外: 流動比率は流動負債 < 10⁻³ × 総資産、固定長期適合率は分母 ≤ 10⁻³ × "
            "総資産。旧モデル = ステージ80 の A1・SP-F1(λ̃ = 0.1)。", "",
            "| 指標 | class | 対象 | 中央値 | IQR | [Q1, Q3] 適合率 | [P10, P90] 適合率 | 除外 |",
            "|---|---|---|---:|---|---:|---:|---:|"]
    for k in hr:
        h = hr[k][np.isfinite(hr[k])]
        q1, q3, p10, p90 = np.quantile(h, [0.25, 0.75, 0.1, 0.9])
        rep.append(f"| {k} | — | 健全企業 | {f3(np.median(h))} | [{f3(q1)}, {f3(q3)}] | — | — | "
                   f"{int((~np.isfinite(hr[k])).sum())} |")
        for cls in ("c2", "c1"):
            x0, ym, _d = arrays(src(best.cost, "t123"), cls)
            yo = arrays(old, cls)[1]
            for lab_, v in (("始点", x0), (f"主モデル({sel['best']})", ym),
                            ("旧モデル(A1・SP-F1)", yo)):
                rv, exc = ratios(v, members, idx)
                vv = rv[k][np.isfinite(rv[k])]
                in_iqr = float(((vv >= q1) & (vv <= q3)).mean()) if len(vv) else np.nan
                in_8 = float(((vv >= p10) & (vv <= p90)).mean()) if len(vv) else np.nan
                rrows.append({"ratio": k, "cls": cls, "who": lab_, "median": float(np.median(vv)),
                              "q1": float(np.quantile(vv, 0.25)),
                              "q3": float(np.quantile(vv, 0.75)),
                              "fit_iqr": in_iqr, "fit_p10_90": in_8, "excluded": exc.get(k, 0)})
                rep.append(f"| {k} | {cls} | {lab_} | {f3(np.median(vv))} "
                           f"| [{f3(np.quantile(vv, 0.25))}, {f3(np.quantile(vv, 0.75))}] "
                           f"| {pct(in_iqr)} | {pct(in_8)} | {exc.get(k, 0)} |")
    pd.DataFrame(rrows).to_csv(OUT / "exp6_ratios.csv", index=False)
    rep.append("")
    (OUT / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
