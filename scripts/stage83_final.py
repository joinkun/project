#!/usr/bin/env python
"""ステージ83: パートA・B の表、実験 4〜6、図 1・2、報告(lightgbm・sklearn のプロセス)。

規則は ``results/stage83/criteria_stage83.md``。事前に ``run_d1.sh``・``stage83_evaluate.py``・
バックテスト(``stage83_backtest_*``)を実行する。数値は小数第 3 位、割合は % で小数第 1 位。

出力: ``results/stage83/report.md``・``paper_tables.md``、各表の CSV、
``results/stage83/figures/bs_manifold_mapping_kde0_stage1.{png,pdf}``・
``bs_backtest_overlay_kde0.{png,pdf}``(300 dpi)。

    uv run python scripts/stage83_final.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import matplotlib as mpl

mpl.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stage76_feasibility_map import (  # noqa: E402
    moves,
    panel_years,
)
from stage82_common import COSTS, Setting  # noqa: E402
from stage82_final import f3, pct, ratios, sym  # noqa: E402
from stage83_common import DATA, FOLDS, NAMES, OUT  # noqa: E402
from stage83_figures import draw_figures  # noqa: E402
from tables_stage69 import INTERIM, Src, metrics, setup_excl  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.models.unit_costs import (  # noqa: E402
    emergence_rate,
    inactive_mask,
    individual_theta,
    psi,
)

_RE = "jppfs_cor_RetainedEarnings"
MAIN = NAMES["KDE0p"]
REP_DOC = "S100UUP0"
OT_ORDER = (("proposed_quad_unbalanced", "prop"), ("pure_lp", "pure_lp"), ("topk_hard_5", "top5"),
            ("topk_hard_8", "top8"), ("sinkhorn", "sinkhorn"), ("quad_row_only", "row"),
            ("quad_col_only", "col"))


def d1_table(key: str, order: list[str]) -> tuple[list[str], pd.DataFrame]:
    """D1 の主指標の表(ステージ82 と同じ列)。"""
    sm = json.loads((OUT / key / "summary.json").read_text(encoding="utf-8"))
    ex = pd.read_csv(OUT / key / "extras.csv").set_index(["setting", "cls"])
    rows = []
    for name in order:
        s = sm["summary"][name]
        lf = [v for v in sm["res"][name]["L"].values() if v is not None and np.isfinite(v)]
        r: dict[str, Any] = {"setting": name, "lam": ",".join(
            "—" if v is None else f"{v:g}" for v in sm["res"][name]["star"].values()),
            "L": float(np.mean(lf)) if len(lf) == len(FOLDS) else np.nan}
        for c in ("c2", "c1"):
            a = s[c]
            if not a.get("ok"):
                continue
            r |= {f"V_{c}": a["viol_max"], f"dNA_{c}": a["nai"], f"d69_{c}": a["d69"]["nai"][0],
                  f"s69_{c}": sym(a["sign69"]), f"sB_{c}": sym(a["signR1"]) if "signR1" in a
                  else "基準", f"res_{c}": a["res"], f"rho_{c}": a["rho"], f"ess_{c}": a["ess"],
                  f"pd_{c}": ex.loc[(name, c), "p_distress"],
                  f"rv_{c}": ex.loc[(name, c), "r_void"],
                  f"em_{c}": ex.loc[(name, c), "emerge_mat"], f"zero_{c}": a["e_zero"],
                  f"paid_{c}": a["paid_dec"], f"nch_{c}": a["n_changed"]}
        rows.append(r)
    df = pd.DataFrame(rows)
    lines = ["| 設定 | λ̃*_f | V | ΔNA | 対 69 の差 | 記号(対 69) | 記号(対 基準) | R_solv | ρ_rank "
             "| P_distress | R_void | 発生率(≥10⁻³) | ESS | L_f |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|"]
    for _, r in df.iterrows():
        def two(k: str, fmt: Any = f3, r: Any = r) -> str:
            return f"{fmt(r.get(f'{k}_c2', np.nan))} / {fmt(r.get(f'{k}_c1', np.nan))}"
        lines.append(f"| {r['setting']} | {r['lam']} | {r.get('V_c2', '—')} / {r.get('V_c1', '—')} "
                     f"| {two('dNA')} | {two('d69')} | {r.get('s69_c2', '—')} / "
                     f"{r.get('s69_c1', '—')} | {r.get('sB_c2', '—')} / {r.get('sB_c1', '—')} "
                     f"| {two('res')} | {two('rho')} | {two('pd', pct)} | {two('rv', pct)} "
                     f"| {two('em', pct)} | {two('ess')} | {f3(r['L'])} |")
    df.to_csv(OUT / f"{key}_main.csv", index=False)
    return lines, df


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    idx = {n: i for i, n in enumerate(names)}
    setup_excl(names)
    gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
    lam = float(gate["lambda_tilde"])
    tj = json.loads((DATA / "theta.json").read_text(encoding="utf-8"))
    sigma = np.array(tj["sigma"], dtype=float)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rep = ["# ステージ83 報告: 主モデル KDE0 + stage1_penalty の確定、バックテスト、移動の地図", "",
           "規則は `criteria_stage83.md`(コミット 83125f5、計算の前に記録)。数値は小数第 3 位、"
           "割合は % で小数第 1 位。", "D1 = ステージ75 と同じ 5 fold の交差適合"
           "(値は fold の平均、V は最大、ESS は中央値)。記号 = 純資産の改善・解消率・順位相関"
           "(+ 優れる、− 劣る、0 差なし)。対 基準の記号は「その行 − 基準の行」。", "",
           "<!-- 要約(手で書く) -->", ""]
    # パートA
    la, _ = d1_table("partA", [MAIN, NAMES["KDE0"], NAMES["R1p"]])
    fb = pd.read_csv(OUT / "kde0_fallback.csv")
    corr = pd.read_csv(Path("results/stage82/exp1_theta_corr.csv"), index_col=0)
    e1 = json.loads(Path("results/stage82/exp1/summary.json").read_text(encoding="utf-8"))
    e1x = pd.read_csv(Path("results/stage82/exp1/extras.csv")).set_index(["setting", "cls"])
    rep += ["## パートA: D1(基準 = 主モデル)", "", *la, "",
            f"**単一の生成の λ̃ = {lam:g}**(主モデルの λ̃*_f の中央値)。", "",
            "ゲート(criteria §5、`gate.json`): "
            + "、".join(f"{k} {'○' if v else '×'}" for k, v in gate.items()
                       if k.startswith("gate")) + "。", "",
            "### KDE0 の代替の一覧(`kde0_fallback.csv`)", "",
            "| プール | 科目 | 向き | 動いた社数 | 理由 |", "|---|---|---|---:|---|"]
    for _, r in fb[fb["pool"].isin([*FOLDS, "full"])].iterrows():
        rep.append(f"| {r['pool']} | {r['label']} | {r['dir']} | {r['n_movers']} | {r['reason']} |")
    rep += ["", "### 頑健性: KDE0・R1・A1", "",
            "θ の順位相関(57 社の全体、62 方向、ステージ82): "
            f"KDE0–R1 {f3(corr.loc['KDE0', 'R1'])}、KDE0–A1 {f3(corr.loc['KDE0', 'A1'])}、"
            f"R1–A1 {f3(corr.loc['R1', 'A1'])}。", "",
            "| 設定(D1) | ΔNA c2 / c1 | ρ_rank c2 / c1 | 発生率 c2 / c1 | 記号(対 69)c2 / c1 |",
            "|---|---|---|---|---|"]
    sa = json.loads((OUT / "partA" / "summary.json").read_text(encoding="utf-8"))
    pa = pd.read_csv(OUT / "partA" / "extras.csv").set_index(["setting", "cls"])
    for lab_, ss, xx, key in (("R1(base、ステージ82)", e1, e1x, "R1"),
                              ("KDE0(base、ステージ82)", e1, e1x, "KDE0"),
                              ("A1(base、ステージ82)", e1, e1x, "A1"),
                              ("R1 + stage1_penalty", sa, pa, NAMES["R1p"]),
                              ("KDE0 + stage1_penalty(主)", sa, pa, MAIN)):
        s = ss["summary"][key]
        rep.append(f"| {lab_} | {f3(s['c2']['nai'])} / {f3(s['c1']['nai'])} "
                   f"| {f3(s['c2']['rho'])} / {f3(s['c1']['rho'])} "
                   f"| {pct(xx.loc[(key, 'c2'), 'emerge_mat'])} / "
                   f"{pct(xx.loc[(key, 'c1'), 'emerge_mat'])} "
                   f"| {sym(s['c2']['sign69'])} / {sym(s['c1']['sign69'])} |")
    rep.append("")
    # パートB 実験 3
    lb, _ = d1_table("partB", [n for n, _k in OT_ORDER])
    diag = []
    for name, k in OT_ORDER:
        for cls in ("c2", "c1"):
            js = [json.loads((OUT / "ot" / f"{cls}_{g}_KDE0p_{k}.json").read_text(encoding="utf-8"))
                  for g in FOLDS]
            diag.append({"setting": name, "cls": cls,
                         **{m: float(np.mean([j[m] for j in js]))
                            for m in ("nz_mean", "max_load", "unique_targets", "ess_med")}})
    dg = pd.DataFrame(diag)
    dg.to_csv(OUT / "exp3_ot_diag.csv", index=False)
    rep += ["## パートB-1: 輸送の方式のアブレーション(KDE0 + stage1_penalty)", "",
            "| 方式 | 非ゼロの候補数 c2 / c1 | 最大の列の負荷 c2 / c1 "
            "| 選ばれた健全企業の数 c2 / c1 "
            "| ESS c2 / c1 |", "|---|---|---|---|---|"]
    for name, _k in OT_ORDER:
        a, b = (dg[(dg["setting"] == name) & (dg["cls"] == c)].iloc[0] for c in ("c2", "c1"))
        rep.append(f"| {name} | {f3(a['nz_mean'])} / {f3(b['nz_mean'])} | {f3(a['max_load'])} / "
                   f"{f3(b['max_load'])} | {a['unique_targets']:.1f} / {b['unique_targets']:.1f} "
                   f"| {f3(a['ess_med'])} / {f3(b['ess_med'])} |")
    rep += ["", "D1(基準 = proposed_quad_unbalanced):", "", *lb, "",
            f"代表企業(ANAP、{REP_DOC}、class 2、57 社の全体の計画)の上位 10 候補の重み:", "",
            "| 方式 | " + " | ".join(str(k) for k in range(1, 11)) + " |", "|---|" + "---:|" * 10]
    wrows = []
    for name, k in OT_ORDER:
        z = np.load(DATA / f"plan_c2_full_KDE0p_{k}.npz", allow_pickle=True)
        i = [str(v) for v in z["source_docs"]].index(REP_DOC)
        w = [float(z["top10_p"][i][j]) for j in range(10)]
        wrows.append({"setting": name, **{f"w{j + 1}": w[j] for j in range(10)}})
        rep.append(f"| {name} | " + " | ".join(f3(v) for v in w) + " |")
    pd.DataFrame(wrows).to_csv(OUT / "exp3_rep_weights.csv", index=False)
    rep.append("")

    def src(cost: str, tier: str) -> Src:
        tag = Setting("full", cost, "prop", "base", tier).tag
        return Src(DATA / "full" / "metrics_full", DATA / "full" / "gen_full", f"{tag}__l{lam:g}",
                   f"{lam:g}")

    def arrays(s_: Src, cls: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        e51._GEN = s_.gen
        g = e51.load_gen(cls, s_.setting)
        g = g[g["seed"] == 0]
        return (g[[f"x0_{n}" for n in names]].to_numpy(float),
                g[[f"x1_{n}" for n in names]].to_numpy(float), [str(d) for d in g["docID"]])
    # 実験 4
    labels = {k.split("_")[0]: k for k in (*CHECK_NAMES, *EXTRA_CHECK_NAMES)}
    trows = []
    rep += ["## パートB-2: 制約の 3 階層(主モデルの単一の生成)", "",
            "| 階層 | class | V | ΔNA | R_solv | 項目別の件数(企業 × seed) |",
            "|---|---|---:|---:|---:|---|"]
    for tier, tl in (("t1", "tier1"), ("t12", "tier1+2"), ("t123", "tier1+2+3")):
        for cls in ("c2", "c1"):
            m = metrics(src("KDE0p", tier), cls, names, members)
            it = "、".join(f"{labels.get(k, k)} {v}" for k, v in sorted(
                m["items"].items(), key=lambda kv: int(kv[0])))
            trows.append({"tier": tl, "cls": cls, "V": m["viol"], "nai": m["nai"], "res": m["res"],
                          "items": it})
            rep.append(f"| {tl} | {cls} | {m['viol']} | {f3(m['nai'])} | {f3(m['res'])} "
                       f"| {it or 'なし'} |")
    pd.DataFrame(trows).to_csv(OUT / "exp4_tiers.csv", index=False)
    rep.append("")
    # 実験 5
    gap = pd.read_csv(OUT / "exp5_survival_gap.csv")
    rep += ["## パートB-3: 生存バイアスと Tier", "",
            f"回復群 72 社と滞留群 {len(tj['pools']['stag'])} 社の KDE0 の θ の比"
            "(上位 10 方向):", "",
            "| 科目 | 向き | θ 回復群 | θ 滞留群 | 滞留 / 回復 | n 回復 / 滞留 |",
            "|---|---|---:|---:|---:|---|"]
    for _, r in gap.head(10).iterrows():
        rep.append(f"| {r['label']} | {r['dir']} | {f3(r['theta_rec72'])} | {f3(r['theta_stag'])} "
                   f"| {f3(r['ratio_stag_over_rec'])} | {r['n_rec']} / {r['n_stag']} |")
    rep += ["", "| プール | class | V | ΔNA | R_solv | ρ_rank(標準の参照) | 払込資本↓ | 変更科目数 "
            "| 発生率(≥10⁻³) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    crows = []
    for lab_, cost in (("回復群 57 社(主モデル)", "KDE0p"), ("苦境全体(回復 + 滞留)", "KDE0p_all")):
        for cls in ("c2", "c1"):
            m = metrics(src(cost, "t123"), cls, names, members)
            x0, y, _d = arrays(src(cost, "t123"), cls)
            em = emergence_rate(x0, y, names, 1e-3)
            crows.append({"pool": lab_, "cls": cls, **{k: m[k] for k in
                          ("viol", "nai", "res", "rho", "paid_dec", "n_changed")}, "emerge": em})
            rep.append(f"| {lab_} | {cls} | {m['viol']} | {f3(m['nai'])} | {f3(m['res'])} "
                       f"| {f3(m['rho'])} | {pct(m['paid_dec'])} | {m['n_changed']:.0f} "
                       f"| {pct(em)} |")
    pd.DataFrame(crows).to_csv(OUT / "exp5_alldist.csv", index=False)
    cs = COSTS["KDE0p"]
    th = np.array(tj["configs"]["full"]["KDE0"]["theta"], dtype=float)

    def psi_i(x0: np.ndarray, d: np.ndarray) -> float:
        t = individual_theta(th, x0, sigma, indiv_scale=cs.indiv,
                             inactive=inactive_mask(x0, names) if cs.penalty else None)
        t[:, idx[_RE]] = 0.0
        return float(psi(t, d))
    comp, _ = panel_years(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    d72, c72 = moves(comp, 2022, 2024, pairs["edinetCode"].tolist())
    x72 = comp[2022].loc[c72].to_numpy()
    rc = np.array([psi_i(x72[k], d72[k]) for k in range(len(c72))])
    q = {p: float(np.quantile(rc, p)) for p in (0.5, 0.75, 0.9, 0.95)}
    rep += ["", "再建群 72 社の ψ_k(KDE0p の θ)の分位点: "
            + "、".join(f"q{p:.2f} = {f3(v)}" for p, v in q.items()) + "。", "",
            "| class | 社数 | Tier A | Tier B | Tier C | 生成の ψ の中央値 |",
            "|---|---:|---|---|---|---:|"]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    filer = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["filerName"]
    tier_rows: list[dict[str, Any]] = []
    tc_list: list[dict[str, Any]] = []
    mids: dict[str, list[np.ndarray]] = {}
    for cls in ("c2", "c1"):
        x0, y, dd = arrays(src("KDE0p", "t123"), cls)
        g = np.array([psi_i(x0[i], y[i] - x0[i]) for i in range(len(x0))])
        ta, tb, tc = g <= q[0.75], (g > q[0.75]) & (g <= q[0.95]), g > q[0.95]
        n = len(g)
        tier_rows.append({"cls": cls, "n": n, "A": int(ta.sum()), "B": int(tb.sum()),
                          "C": int(tc.sum()), "gen_q50": float(np.median(g))})
        rep.append(f"| {cls} | {n} | {int(ta.sum())}({pct(ta.mean())}) | {int(tb.sum())}"
                   f"({pct(tb.mean())}) | {int(tc.sum())}({pct(tc.mean())}) | {f3(np.median(g))} |")
        for i in map(int, np.where(tc)[0]):
            xm = 0.5 * x0[i] + 0.5 * y[i]
            fl = np.concatenate([evaluate_checks(x0[i:i + 1], xm[None], spec, names)[0],
                                 evaluate_extra_checks(xm[None], x0[i:i + 1], spec, names)[0]])
            row: dict[str, Any] = {"cls": cls, "docID": dd[i], "code": code_of.get(dd[i], ""),
                                   "name": str(filer.get(dd[i], "")), "psi": float(g[i]),
                                   "viol_mid": int(fl.sum())}
            for lab_, v in (("x0", x0[i]), ("mid", xm), ("y", y[i])):
                a_ = float(v @ members["jppfs_cor_Assets"])
                row[f"{lab_}_equity"] = float(v @ members["jppfs_cor_NetAssets"]) / a_
                row[f"{lab_}_cash"] = float(v[idx["jppfs_cor_CashAndDeposits"]]) / a_
                debt = sum(v[idx[k]] for k in ("short_term_loans", "long_term_loans",
                                               "bonds_current", "bonds_noncurrent"))
                row[f"{lab_}_debt"] = float(debt) / a_
            tc_list.append(row)
            mids.setdefault(cls, []).append(xm)
    pd.DataFrame(tier_rows).to_csv(OUT / "exp5_tiers.csv", index=False)
    tcd = pd.DataFrame(tc_list)
    tcd.to_csv(OUT / "exp5_tierC.csv", index=False)
    rep += ["", "Tier の名前(短期・中期・長期)は解釈のラベルで、実際の年数ではない。", "",
            "Tier C の企業と中間点 y⁽⁰·⁵⁾ = 0.5 x + 0.5 ŷ"
            "(自己資本比率・現預金・有利子負債の総資産比、"
            "始点 → 中間 → 終点):", "",
            "| class | 企業コード | 企業名 | ψ | 中間点の違反(15 項目) | 自己資本比率 | 現預金 "
            "| 有利子負債 |",
            "|---|---|---|---:|---:|---|---|---|"]
    for _, r in tcd.iterrows():
        rep.append(f"| {r['cls']} | {r['code']} | {r['name']} | {f3(r['psi'])} | {r['viol_mid']} | "
                   + " | ".join(f"{f3(r[f'x0_{k}'])} → {f3(r[f'mid_{k}'])} → {f3(r[f'y_{k}'])}"
                                for k in ("equity", "cash", "debt")) + " |")
    rep.append("")
    # 実験 6
    x, lab, docs, _ = load_tilde(names)
    xh = x[lab == 0]
    hr, hex_ = ratios(xh, members, idx)
    old = Src(INTERIM / "stage80/metrics_full", INTERIM / "stage80/gen_full", "A1_sp__l0.1", "0.1")
    rep += ["## パートB-4: 5 つの財務比率", "",
            "健全レンジ = 健全企業の [Q1, Q3] と [P10, P90]。"
            "除外: 流動比率は流動負債 < 10⁻³ × 総資産、"
            "固定長期適合率は分母 ≤ 10⁻³ × 総資産。旧モデル = ステージ80 の A1・SP-F1。", "",
            "| 指標 | class | 対象 | 中央値 | IQR | [Q1, Q3] 適合率 | [P10, P90] 適合率 | 除外 |",
            "|---|---|---|---:|---|---:|---:|---:|"]
    rrows = []
    for k in hr:
        h = hr[k][np.isfinite(hr[k])]
        q1, q3, p10, p90 = np.quantile(h, [0.25, 0.75, 0.1, 0.9])
        rep.append(f"| {k} | — | 健全企業 | {f3(np.median(h))} | [{f3(q1)}, {f3(q3)}] | — | — "
                   f"| {hex_.get(k, 0)} |")
        for cls in ("c2", "c1"):
            x0, ym, _d = arrays(src("KDE0p", "t123"), cls)
            yo = arrays(old, cls)[1]
            for lab_, v in (("始点", x0), ("主モデル(KDE0 + stage1_penalty)", ym),
                            ("旧モデル(A1・SP-F1)", yo)):
                rv, exc = ratios(v, members, idx)
                vv = rv[k][np.isfinite(rv[k])]
                fi = float(((vv >= q1) & (vv <= q3)).mean())
                f8 = float(((vv >= p10) & (vv <= p90)).mean())
                rrows.append({"ratio": k, "cls": cls, "who": lab_, "median": float(np.median(vv)),
                              "q1": float(np.quantile(vv, 0.25)),
                              "q3": float(np.quantile(vv, 0.75)),
                              "fit_iqr": fi, "fit_p10_90": f8, "excluded": exc.get(k, 0)})
                rep.append(f"| {k} | {cls} | {lab_} | {f3(np.median(vv))} "
                           f"| [{f3(np.quantile(vv, 0.25))}, {f3(np.quantile(vv, 0.75))}] "
                           f"| {pct(fi)} | {pct(f8)} | {exc.get(k, 0)} |")
    pd.DataFrame(rrows).to_csv(OUT / "exp6_ratios.csv", index=False)
    rep.append("")
    # パートC
    bs = pd.read_csv(OUT / "backtest_summary.csv")
    br = pd.read_csv(OUT / "backtest_ratios.csv")
    rep += ["## パートC: 実在の再建企業への out-of-fold のバックテスト", "",
            "中央値 [Q1, Q3](正の割合)。"
            "主 = KDE0 + stage1_penalty、λ̃ はほかの fold の中央値(リークなし)。"
            "「λ̃*_f」は指示どおりの自 fold の λ̃(集計値の水準でリークあり)。", "",
            "| class | モデル | 社数 | DSA(ε=10⁻²)| DSA(5×10⁻³)| DSA(2×10⁻²)| Cos | ρ_k "
            "| ERR^L1 | ERR^ψ | P_distress x₀ → x₁ / ŷ |",
            "|---|---|---:|---|---|---|---|---|---|---|---|"]
    lab_m = {"KDE0p|main": "主モデル", "KDE0p|own": "主モデル(λ̃*_f)", "KDE0|main": "KDE0 base",
             "KDE0|own": "KDE0 base(λ̃*_f)", "1-NN": "1-NN の健全企業",
             "健全企業の中央値": "健全企業の中央値"}

    def cell(r: pd.Series, m: str, signed: bool = False) -> str:
        s_ = f"{f3(r[f'{m}_med'])} [{f3(r[f'{m}_q1'])}, {f3(r[f'{m}_q3'])}]"
        return s_ + (f"({pct(r[f'{m}_pos'])})" if signed else "")
    for cls in ("c2", "c1"):
        for mk, ml in lab_m.items():
            sel = bs[(bs["cls"] == cls) & (bs["model"] == mk)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            rep.append(f"| {cls} | {ml} | {r['n']} | {cell(r, 'dsa_0.01')} "
                       f"| {cell(r, 'dsa_0.005')} "
                       f"| {cell(r, 'dsa_0.02')} | {cell(r, 'cos', True)} | {cell(r, 'rho', True)} "
                       f"| {cell(r, 'err_l1', True)} | {cell(r, 'err_psi', True)} "
                       f"| {pct(r['p_x0_med'])} → {pct(r['p_x1_med'])} / {pct(r['p_y_med'])} |")
    rep += ["", "5 つの財務比率(主モデル): 中央値 x₀ → x₁(実績)/ ŷ(生成)、"
            "改善幅の相関、MAE(ŷ と x₁)", "",
            "| class | 指標 | x₀ | x₁ | ŷ | 改善幅の相関 | MAE |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in br[br["model"] == "KDE0p|main"].iterrows():
        rep.append(f"| {r['cls']} | {r['ratio']} | {f3(r['x0_med'])} | {f3(r['x1_med'])} "
                   f"| {f3(r['y_med'])} | {f3(r['corr_change'])} | {f3(r['mae'])} |")
    rep += ["", "事例は `backtest_cases.md`。", ""]
    # 図(パートD、描画は stage83_figures.draw_figures)
    f1, f2, share = draw_figures(lam, bs)
    rep += ["## パートD: 図", "",
            f"- 図 1: `{f1}.png`・`.pdf`(Ψ_KDE0 の古典的 MDS の上位 2 成分の寄与 {share:.0%}。"
            "距離は罰則なしの KDE0 の θ を対称化したもの。"
            "下段の横軸は分類器のロジットなので、左右の勾配は"
            "作り方から生じる)。★ = class 2 の Tier C の中間点。",
            "左列 (a)(c) は主な点が読める標準の範囲に固定し、Tier C の全体は別図 "
            "`results/stage83/figures/bs_tier_c_overview_kde0.png`・`.pdf` に示す。",
            f"- 図 2: `{f2}.png`・`.pdf`(緑 = 実績、紫 = out-of-fold の生成。同じ始点から)。", ""]
    (OUT / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
