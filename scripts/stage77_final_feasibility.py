#!/usr/bin/env python
"""ステージ77: 採用した R1・F1(案 a: 57 社の R1 コスト、λ̃ = 0.1)とステージ69 の設定の比較(lightgbm・
sklearn のプロセス)。

ステージ76 の苦境の確率・実行可能性・近傍の実在密度・地図を、2 つの設定で並べて計算する。あわせて、
最終の比較表(主指標)と class 2 の 5 社の BS 表を出す。

- ステージ69 の設定: ``data/interim/stage67/all`` の ``s1__l1``。
- R1・F1(案 a): ``data/interim/stage75/gen_full`` の ``R1_f1__l0.1``。
- 移動の尺度: ステージ69 の Ψ(ステージ60 の c±、ステージ76 と同じ)と、R1 の Ψ(57 社の全体)。
- 地図: 図 1 はステージ76 と同じ古典的 MDS。図 2 は、横軸を LogReg(苦境 vs 健全、全データで 1 回
  学習)のロジット、縦軸を MDS 1 からロジットの成分を除いた残差にした平面。
  横軸が分類器そのものなので、左右の勾配は作り方から生じる(図の説明に明記する)。

出力: ``results/stage77/``(``final_feasibility.md``・``final_tables.md``・``final.json``・
``bs_manifold_mapping_r1f1.png``・``.pdf``・``bs_r1f1.md``)。

    uv run python scripts/stage77_final_feasibility.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from compare_stage39 import corr  # noqa: E402
from evaluate_stage65 import bs_table  # noqa: E402
from stage71_evaluate import endpoints  # noqa: E402
from stage72_evaluate import ZERO, ci  # noqa: E402
from stage74_evaluate import inflation, start_points  # noqa: E402
from stage75_evaluate import t2_ref, versus_ref  # noqa: E402
from stage76_feasibility_map import (  # noqa: E402
    CLFS,
    N_FOLD,
    N_LAND,
    REAL_GROUP,
    SEED_CLF,
    TARGETS,
    boot_median_diff,
    make_clf,
    mds_embed,
    mds_fit,
    moves,
    panel_years,
    psi,
)
from tables_stage69 import INTERIM, S67, Src, ctx, metrics, real_row, setup_excl  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import subtotal_members  # noqa: E402
from fsgen.evaluation.classifiers import strict_sklearn_convergence  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402
from fsgen.viz.balance_sheet import load_labels  # noqa: E402

OUT = Path("results/stage77")
S75F = (INTERIM / "stage75/metrics_full", INTERIM / "stage75/gen_full")
SRC = {"s69": Src(*S67, "s1__l1", "1"), "r1f1": Src(*S75F, "R1_f1__l0.1", "0.1")}
LAB = {"s69": "ステージ69 の設定(λ̃ = 1)", "r1f1": "R1・F1(案 a、λ̃ = 0.1)"}
_RE = "jppfs_cor_RetainedEarnings"
K_SMOOTH = 30


def load_gen(src: Src, cls: str, names: tuple[str, ...]) -> pd.DataFrame:
    e51._GEN = src.gen
    g = e51.load_gen(cls, src.setting)
    out: pd.DataFrame = g[g["seed"] == 0].set_index("docID")
    return out


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    x, lab, docs, _ = load_tilde(names)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes = np.array([code_of.get(d, d) for d in docs])
    # 分類器(ステージ76 と同じ: 企業単位の 5 分割、乱数 42)
    uc = np.random.default_rng(SEED_CLF).permutation(np.unique(codes))
    fold_of_code = {c: i for i, part in enumerate(np.array_split(uc, N_FOLD)) for c in part}
    fold = np.array([fold_of_code[c] for c in codes])
    ymask = {"distress": np.ones(len(lab), bool), "gc": lab != 1}
    yvec = {"distress": lab >= 1, "gc": lab == 2}
    models: dict[tuple[str, str, int], Any] = {}
    oof = {(t, k): np.full(len(lab), np.nan) for t in TARGETS for k in CLFS}
    prior: dict[str, float] = {}
    for t in TARGETS:
        m_t = ymask[t]
        prior[t] = float(yvec[t][m_t].sum() / (~yvec[t][m_t]).sum())
        for k in CLFS:
            for f in range(N_FOLD):
                tr = m_t & (fold != f)
                clf = make_clf(k)
                with strict_sklearn_convergence():
                    clf.fit(x[tr], yvec[t][tr].astype(int))
                models[(t, k, f)] = clf
                te = fold == f
                oof[(t, k)][te] = clf.predict_proba(x[te])[:, 1]
    # 図 2 の横軸: 全データで 1 回学習した LogReg(苦境 vs 健全)のロジット(記述用)
    axis_clf = make_clf("logreg")
    with strict_sklearn_convergence():
        axis_clf.fit(x, yvec["distress"].astype(int))

    def logit_axis(pts: np.ndarray) -> np.ndarray:
        out: np.ndarray = axis_clf.decision_function(pts)
        return out

    def predict(t: str, k: str, pts: np.ndarray, folds: np.ndarray) -> np.ndarray:
        out = np.empty(len(pts))
        for f in range(N_FOLD):
            s = folds == f
            if s.any():
                out[s] = models[(t, k, f)].predict_proba(pts[s])[:, 1]
        return out

    def corrected(p: np.ndarray, t: str) -> np.ndarray:
        o = p / np.clip(1.0 - p, 1e-12, None) * prior[t]
        out: np.ndarray = o / (1.0 + o)
        return out

    comp, _ = panel_years(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    wj = json.loads((INTERIM / "stage75/weights.json").read_text(encoding="utf-8"))
    cr1 = (np.array(wj["configs"]["full"]["R1"]["plus"], float),
           np.array(wj["configs"]["full"]["R1"]["minus"], float))
    for c_ in cr1:
        c_[names.index(_RE)] = 0.0
    healthy = lab == 0
    xh, ch = x[healthy], codes[healthy]
    land_i = np.random.default_rng(0).choice(len(xh), size=N_LAND, replace=False)
    q_i = np.random.default_rng(1).choice(len(xh), size=2000, replace=False)
    res: dict[str, Any] = {"prior_odds": prior}
    fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
    for ci_, cls in enumerate(("c2", "c1")):
        meta = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(encoding="utf-8"))
        c60 = (np.array(meta["c_plus"], float), np.array(meta["c_minus"], float))
        rc_ = pairs[pairs["group"] == REAL_GROUP[cls]]["edinetCode"].tolist()
        # 実在の推移の Ψ(2 つの尺度)
        refs: dict[str, dict[str, np.ndarray]] = {}
        for mk, (cp, cm) in (("psi60", c60), ("psiR1", cr1)):
            refs[mk] = {
                "all_h1": np.concatenate([psi(moves(comp, a, b)[0], cp, cm) for a, b in
                                          ((2021, 2022), (2022, 2023), (2023, 2024))]),
                "all_h3": psi(moves(comp, 2021, 2024)[0], cp, cm),
                "rec72": psi(moves(comp, 2022, 2024, pairs["edinetCode"].tolist())[0], cp, cm)}
        # 健全企業どうしの最近傍の Ψ60(ステージ76 と同じ半径)
        nn = np.empty(len(q_i))
        for s in range(0, len(q_i), 100):
            qq = q_i[s:s + 100]
            dd = psi(xh[None, :, :] - xh[qq][:, None, :], *c60)
            dd[ch[None, :] == ch[qq][:, None]] = np.inf
            nn[s:s + 100] = dd.min(axis=1)
        r90 = float(np.quantile(nn, 0.9))

        def count(pts: np.ndarray, own: np.ndarray, c60: tuple[np.ndarray, np.ndarray] = c60,
                  r90: float = r90) -> np.ndarray:
            dd = psi(xh[None, :, :] - pts[:, None, :], *c60)
            dd[ch[None, :] == own[:, None]] = np.inf
            return np.array([len(set(ch[row <= r90])) for row in dd])
        hq = q_i[:300]
        healthy_cnt = count(xh[hq], ch[hq])
        rcls: dict[str, Any] = {"r90": r90, "healthy_zero": float((healthy_cnt == 0).mean()),
                                "healthy_med": float(np.median(healthy_cnt))}
        ends: dict[str, np.ndarray] = {}
        x0 = np.empty(0)
        gfold = np.empty(0, int)
        gcode = np.empty(0, str)
        order = list(load_gen(SRC["s69"], cls, names).index)
        for key, src in SRC.items():
            g = load_gen(src, cls, names).loc[order]
            x0 = g[[f"x0_{n}" for n in names]].to_numpy(float)
            y1 = g[[f"x1_{n}" for n in names]].to_numpy(float)
            ends[key] = y1
            gcode = np.array([code_of.get(str(d), str(d)) for d in g.index])
            gfold = np.array([fold_of_code[c] for c in gcode])
            r: dict[str, Any] = {"n": len(g)}
            pdr = []
            for t in TARGETS:
                for k in CLFS:
                    p0, p1 = predict(t, k, x0, gfold), predict(t, k, y1, gfold)
                    q0, q1 = corrected(p0, t), corrected(p1, t)
                    hz = oof[(t, k)][healthy]
                    pdr.append({"target": t, "clf": k,
                                "score": (float(np.median(p0)), float(np.median(p1))),
                                "prob_med": (float(np.median(q0)), float(np.median(q1))),
                                "prob_mean": (float(q0.mean()), float(q1.mean())),
                                "d_score": boot_median_diff(p1, p0),
                                "share_down": float((p1 < p0).mean()),
                                "below_healthy": float((p1 <= np.median(hz)).mean())})
            r["pd"] = pdr
            feas: dict[str, Any] = {}
            for mk, (cp, cm) in (("psi60", c60), ("psiR1", cr1)):
                cg = psi(y1 - x0, cp, cm)
                feas[mk] = {"gen_q50": float(np.median(cg)), "gen_q90": float(np.quantile(cg, 0.9)),
                            "gen_max": float(cg.max())}
                for rk, v in refs[mk].items():
                    v = np.sort(v)
                    pct = np.searchsorted(v, cg, side="right") / len(v)
                    feas[mk][rk] = {"q50": float(np.median(v)), "q95": float(np.quantile(v, 0.95)),
                                    "q99": float(np.quantile(v, 0.99)),
                                    "pct_med": float(np.median(pct)),
                                    "in95": float((cg <= np.quantile(v, 0.95)).mean()),
                                    "in99": float((cg <= np.quantile(v, 0.99)).mean())}
            r["feas"] = feas
            ce = count(y1, gcode)
            r["density"] = {"end_med": float(np.median(ce)), "end_zero": float((ce == 0).mean())}
            if key == "s69":
                cs = count(x0, gcode)
                rcls["start_zero"] = float((cs == 0).mean())
                rcls["start_med"] = float(np.median(cs))
            # 主指標
            m = metrics(src, cls, names, members)
            ye = endpoints(src, cls, names)
            r["m"] = m
            r["infl"] = inflation(ye, start_points(src, cls, names))
            r["e_zero"] = float((np.abs(ye) < ZERO).sum(axis=1).mean())
            rcls[key] = r
        # R1・F1 の統合と候補の AUC、基準とのペア比較
        z = np.load(INTERIM / f"stage75/plan_{cls}_full.npz", allow_pickle=True)
        idx, pt = np.asarray(z["R1__idx"]), np.asarray(z["R1__pt"])
        rcls["r1f1"]["multi"] = float(np.median(1.0 / (pt ** 2).sum(axis=1)))
        rcls["r1f1"]["auc"] = float(
            identification_auc_repeated(xt[idx[pt > 0]], xt, 3)["lightgbm"].mean())
        base_ref = ctx(SRC["s69"], cls, names).ref
        rcls["vs_default_ref"] = versus_ref(SRC["r1f1"], SRC["s69"], cls, names, base_ref)
        if cls == "c1":
            p57 = set(pairs[pairs["group"] == "class 1 → class 0"]["edinetCode"])
            p15 = set(pairs[pairs["group"] == "class 2 → 回復"]["edinetCode"])
            r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")
            t2c = set(r38[(r38["label_prev"] == 1) & (r38["label_cur"] == 0)][
                "edinetCode"].astype(str))
            ref7, _ = t2_ref(t2c - p57 - p15, names)
            rcls["vs_ref7"] = versus_ref(SRC["r1f1"], SRC["s69"], cls, names, ref7)
            rcls["rho_ref7"] = {k: float(corr(ctx(s, cls, names).wide[s.setting], ref7))
                                for k, s in SRC.items()}
        rcls["real"] = real_row(cls, names, members)
        res[cls] = rcls
        # 図 1: ステージ76 と同じ MDS
        wsym = 0.5 * (c60[0] + c60[1])
        land = xh[land_i]
        proj, mean_d2, share = mds_fit(land, wsym)
        rcls["mds_share"] = share
        e_all = mds_embed(x, land, wsym, proj, mean_d2)
        e0 = mds_embed(x0, land, wsym, proj, mean_d2)
        e1 = mds_embed(ends["r1f1"], land, wsym, proj, mean_d2)
        _, rc = moves(comp, 2022, 2024, rc_)
        xa, xb = comp[2022].loc[rc].to_numpy(), comp[2024].loc[rc].to_numpy()
        ra, rb = mds_embed(xa, land, wsym, proj, mean_d2), mds_embed(xb, land, wsym, proj, mean_d2)
        score = oof[("distress", "lightgbm")]
        # 図 2: 横 = LogReg のロジット、縦 = MDS 1 からロジットの成分を除いた残差
        la = logit_axis(x)
        beta = np.polyfit(la, e_all[:, 0], 1)

        def plane(pts: np.ndarray, emb: np.ndarray, beta: np.ndarray = beta) -> np.ndarray:
            lx = logit_axis(pts)
            return np.column_stack([lx, emb[:, 0] - np.polyval(beta, lx)])
        p_all = np.column_stack([la, e_all[:, 0] - np.polyval(beta, la)])
        p0, p1 = plane(x0, e0), plane(ends["r1f1"], e1)
        pa, pb = plane(xa, ra), plane(xb, rb)
        rcls["plane_logit"] = {"start_med": float(np.median(p0[:, 0])),
                               "end_med": float(np.median(p1[:, 0])),
                               "healthy_med": float(np.median(la[healthy])),
                               "end_ge_healthy_med": float((p1[:, 0] <= np.median(la[healthy]))
                                                           .mean())}
        for row, (ea, s0, s1, qa, qb, title) in enumerate((
                (e_all, e0, e1, ra, rb, f"Classical MDS (Ψ_c{cls[1]}, 2-D share {share:.0%})"),
                (p_all, p0, p1, pa, pb, "Distress logit (LogReg) × MDS 1 residual"))):
            ax = axes[row, ci_]
            pts = np.vstack([s0, s1, qa, qb, ea[healthy]])
            lo_, hi_ = np.quantile(pts, 0.01, axis=0), np.quantile(pts, 0.99, axis=0)
            pad = 0.08 * (hi_ - lo_)
            lo_, hi_ = lo_ - pad, hi_ + pad
            gx, gy = np.meshgrid(np.linspace(lo_[0], hi_[0], 120),
                                 np.linspace(lo_[1], hi_[1], 120))
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            sc = (hi_ - lo_)
            zz, dmin = np.empty(len(grid)), np.empty(len(grid))
            for s in range(0, len(grid), 600):
                dd = np.linalg.norm((grid[s:s + 600, None, :] - ea[None, :, :]) / sc, axis=2)
                kk = np.argpartition(dd, K_SMOOTH, axis=1)[:, :K_SMOOTH]
                zz[s:s + 600] = score[kk].mean(axis=1)
                dmin[s:s + 600] = dd.min(axis=1)
            zz[dmin > np.quantile(dmin, 0.85)] = np.nan
            cf = ax.contourf(gx, gy, zz.reshape(gx.shape), levels=np.linspace(0, 1, 11),
                             cmap="RdYlGn_r", alpha=0.35)
            ax.scatter(*ea[healthy].T, s=3, c="#4a6b8a", alpha=0.25, linewidths=0,
                       label="Healthy firms (class 0)")
            ax.quiver(*qa.T, *(qb - qa).T, angles="xy", scale_units="xy", scale=1.0,
                      color="#1b7837", width=0.0035, alpha=0.9, zorder=2)
            ax.quiver(*s0.T, *(s1 - s0).T, angles="xy", scale_units="xy", scale=1.0,
                      color="#762a83", width=0.0022, alpha=0.6, zorder=2)
            ax.scatter(*s0.T, s=14, c="#d6604d", edgecolors="white", linewidths=0.4, zorder=3,
                       label=f"Distressed start x_i (class {cls[1]}, n={len(s0)})")
            ax.scatter(*s1.T, s=14, c="#762a83", edgecolors="white", linewidths=0.4, zorder=3,
                       label="R1-F1 endpoint y_i* (λ̃ = 0.1)")
            ax.plot([], [], color="#1b7837", lw=1.3,
                    label=f"Real recovery 2022→2024 (n={len(rc)})")
            if row == 1:
                ax.axvline(float(np.median(la[healthy])), color="#1b7837", lw=0.9, ls=":",
                           label="Median logit of healthy firms")
                ax.set_xlabel("LogReg distress logit (higher = riskier)")
                ax.set_ylabel("MDS 1 residual (orthogonal to the logit)")
            else:
                ax.set_xlabel("MDS 1")
                ax.set_ylabel("MDS 2")
            ax.set_xlim(lo_[0], hi_[0])
            ax.set_ylim(lo_[1], hi_[1])
            ax.set_title(f"({'abcd'[2 * row + ci_]}) Class {cls[1]}: {title}", fontsize=10,
                         loc="left")
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
            ax.grid(True, color="#dddddd", linewidth=0.5)
        print(f"[{cls}] 完了", flush=True)
    fig.suptitle("R1-F1 (λ̃ = 0.1): moves on the movement-cost map; contours = distress score "
                 "(LightGBM, out-of-fold, 30-NN mean in 2-D)", fontsize=11)
    fig.colorbar(cf, ax=axes, shrink=0.6, pad=0.01).set_label("Distress score")
    fig.savefig(OUT / "bs_manifold_mapping_r1f1.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "bs_manifold_mapping_r1f1.pdf", bbox_inches="tight")
    # BS 表(class 2、ステージ65〜76 と同じ 5 社)
    g69, gr = load_gen(SRC["s69"], "c2", names), load_gen(SRC["r1f1"], "c2", names)
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g69[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    dd_ = np.array(g69.index)
    r_ = np.random.default_rng(0)
    ins = dd_[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in dd_ if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    bs = ["# ステージ77: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
          "企業はステージ65〜76 と同じ 5 社。R1・F1 は案 a(57 社の R1 コスト、λ̃ = 0.1、"
          "`data/interim/stage75/gen_full/c2/R1_f1__l0.1`)。", ""]
    for d in picks:
        r0 = g69.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0),
                ("ステージ69", r0[x1c].to_numpy(float) * a0),
                ("R1・F1 λ̃=0.1", gr.loc[d][x1c].to_numpy(float) * a0)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        bs += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
    (OUT / "bs_r1f1.md").write_text("\n".join(bs) + "\n", encoding="utf-8")
    # 表
    s75 = json.loads((Path("results/stage75/summary.json")).read_text(encoding="utf-8"))
    d1 = s75["d1_summary"]["R1_f1"]
    lines = ["# ステージ77: 最終の比較表(主指標)", "",
             "- D1 = ステージ75 の 5 fold の交差適合"
             "(値は fold の平均、膨張・統合・AUC・ゼロは中央値)。"
             "class 1 の順位相関は fold ごとの参照(17〜19 社)。",
             "- 案 a = 57 社の全体の R1 コストに λ̃ = 0.1(fold の λ̃*_f の中央値)の単一の生成。"
             "**class 1 の順位相関は、標準の参照(T2 の 68 組)が 57 社と 56 社重なる(リーク)ので、"
             "重ならない 7 社の参照も並べる。**", "",
             "| class | 設定 | 違反 | 純資産の改善 | 解消率 | 順位相関 | 払込資本↓ / ↑ "
             "| 変更科目数 "
             "| 膨張 | 終点のゼロ | 統合 | 候補の AUC |",
             "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        rc = res[cls]
        rl = rc["real"]
        for key in ("s69", "r1f1"):
            v = rc[key]
            m = v["m"]
            rho = f"{m['rho']:+.3f}"
            if cls == "c1":
                rho += f"(7 社 {rc['rho_ref7'][key]:+.3f})"
            extra = (f"| {v['multi']:.2f} | {v['auc']:.3f} |" if key == "r1f1" else "| 1 | — |")
            lines.append(f"| {cls} | {LAB[key]} | {m['viol']} | {m['nai']:.3f} | {m['res']:.2f} "
                         f"| {rho} | {m['paid_dec']:.0%} / {m['paid_inc']:.0%} "
                         f"| {m['n_changed']:.0f} | {v['infl']:.1%} | {v['e_zero']:.2f} " + extra)
        dm = d1[cls]
        lines.append(f"| {cls} | R1・F1(D1 の交差適合) | {dm['viol_max']} | {dm['nai']:.3f} "
                     f"| {dm['res']:.2f} | {dm['rho']:+.3f}(fold の参照) "
                     f"| {dm['paid_dec']:.0%} / {dm['paid_inc']:.0%} | {dm['n_changed']:.0f} "
                     f"| {dm['infl']:.1%} | {dm['e_zero']:.2f} | {dm['multi']:.2f} "
                     f"| {dm['auc']:.3f} |")
        lines.append(f"| {cls} | 実データ(再建群 {rl['n']} 社) | — | — | — | — "
                     f"| {rl['paid_dec']:.0%} / {rl['paid_inc']:.0%} | {rl['n_changed']:.0f} "
                     f"| — | 6.70(健全企業) | — | — |")
    lines += ["", "**R1・F1(案 a)− ステージ69 の設定**(企業単位のペア比較、2,000 回):", "",
              "| class | 純資産の改善 [95%] | 順位相関(標準の参照)[95%] | 順位相関(7 社)[95%] |",
              "|---|---|---|---|"]
    for cls in ("c2", "c1"):
        rc = res[cls]
        r7 = ci(rc["vs_ref7"]["rho"]) if cls == "c1" else "—"
        note = "(リークあり)" if cls == "c1" else ""
        lines.append(f"| {cls} | {ci(rc['vs_default_ref']['nai'])} "
                     f"| {ci(rc['vs_default_ref']['rho'])}{note} | {r7} |")
    lines += ["", "**D1 の交差適合での差**(ステージ75、fold 平均、括弧は 95% 区間が 0 より上 / 下の"
              " fold 数):", ""]
    for cls in ("c2", "c1"):
        dm = d1[cls]
        lines.append(f"- {cls}: 純資産の改善 {dm['d_nai'][0]:+.3f}({dm['d_nai'][1]}+ / "
                     f"{dm['d_nai'][2]}−)、順位相関 {dm['d_rho'][0]:+.3f}({dm['d_rho'][1]}+ / "
                     f"{dm['d_rho'][2]}−)")
    (OUT / "final_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 実行可能性の表
    fl = ["# ステージ77: 苦境の確率・実行可能性・近傍の実在密度(ステージ69 の設定 と R1・F1)", "",
          "分類器・尺度・半径はステージ76 と同じ。確率は学習データの有病率に戻したもの"
          "(実際の倒産確率ではない)。", ""]
    for cls in ("c2", "c1"):
        rc = res[cls]
        fl += [f"## class {cls[1]}(始点 {rc['s69']['n']} 社)", "", "### 苦境の確率", "",
               "| 目的変数 | 分類器 | 設定 | 確率の中央値(始点 → 終点) | 確率の平均 "
               "| スコアの差 中央値 [95%] | 下がった割合 | 終点 ≤ 健全の中央値 |",
               "|---|---|---|---|---|---|---:|---:|"]
        for i_ in range(len(rc["s69"]["pd"])):
            for key in ("s69", "r1f1"):
                p = rc[key]["pd"][i_]
                fl.append(f"| {TARGETS[p['target']]} | {p['clf']} | {LAB[key]} "
                          f"| {p['prob_med'][0]:.2%} → {p['prob_med'][1]:.2%} "
                          f"| {p['prob_mean'][0]:.2%} → {p['prob_mean'][1]:.2%} "
                          f"| {ci(p['d_score'])} | {p['share_down']:.0%} "
                          f"| {p['below_healthy']:.0%} |")
        fl += ["", "### 年次推移に基づく実行可能性", "",
               "| 尺度 | 設定 | 生成の Ψ 中央値 / 90% 点 / 最大 "
               "| 全上場企業 1 年: パーセンタイル中央値・"
               "95% 点以内・99% 点以内 | 全上場企業 3 年: 同 | 再建群 72 社(2 年): 同 |",
               "|---|---|---|---|---|---|"]
        for mk, ml in (("psi60", "Ψ(ステージ69)"), ("psiR1", "Ψ(R1)")):
            for key in ("s69", "r1f1"):
                f = rc[key]["feas"][mk]
                cells = " | ".join(f"{f[rk]['pct_med']:.0%}・{f[rk]['in95']:.0%}・"
                                   f"{f[rk]['in99']:.0%}" for rk in ("all_h1", "all_h3", "rec72"))
                fl.append(f"| {ml} | {LAB[key]} | {f['gen_q50']:.3g} / {f['gen_q90']:.3g} / "
                          f"{f['gen_max']:.3g} | {cells} |")
        f60 = rc["s69"]["feas"]["psi60"]
        fl += ["", f"実在の Ψ(ステージ69): 1 年の中央値 {f60['all_h1']['q50']:.3g}・95% 点 "
               f"{f60['all_h1']['q95']:.3g}・99% 点 {f60['all_h1']['q99']:.3g}。", "",
               "### 近傍の実在密度(Ψ(ステージ69)(z − y) ≤ r90 の健全企業、自社を除く)", "",
               f"r90 = {rc['r90']:.3g}。", "",
               "| 対象 | 中央値 | 0 社の割合 |", "|---|---:|---:|",
               f"| 始点 | {rc['start_med']:.0f} | {rc['start_zero']:.0%} |"]
        for key in ("s69", "r1f1"):
            dn = rc[key]["density"]
            fl.append(f"| 終点: {LAB[key]} | {dn['end_med']:.0f} | {dn['end_zero']:.0%} |")
        pl = rc["plane_logit"]
        fl += [f"| 健全企業(300 社の標本) | {rc['healthy_med']:.0f} | {rc['healthy_zero']:.0%} |",
               "", f"図 2 の横軸(LogReg のロジット、全データで学習): 始点の中央値 "
               f"{pl['start_med']:+.2f} → R1・F1 の終点 {pl['end_med']:+.2f}(健全企業の中央値 "
               f"{pl['healthy_med']:+.2f}、終点がそれ以下 {pl['end_ge_healthy_med']:.0%})。"
               f"MDS の 2 次元の寄与 {rc['mds_share']:.0%}。", ""]
    (OUT / "final_feasibility.md").write_text("\n".join(fl) + "\n", encoding="utf-8")
    (OUT / "final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                    encoding="utf-8")
    print("\n".join(lines + [""] + fl))


if __name__ == "__main__":
    main()
