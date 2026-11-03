#!/usr/bin/env python
"""ステージ76: デフォルト確率の低下、年次推移に基づく実行可能性、距離空間の地図(lightgbm・sklearn の
プロセス)。

生成は最終設定(ステージ69 の設定 = ステージ67 の ``s1__l1``、seed 0)。距離は最終設定の移動コスト
Ψ_cls(Δ) = Σ_a c_a⁺ (Δ_a)₊ + c_a⁻ (−Δ_a)₊(ステージ60 の ``meta_{cls}.json``、利益剰余金・
評価差額金・非支配株主持分は 0)。

1. 苦境の確率: 苦境(class 1・2)vs 健全(class 0)と、GC(class 2)vs 健全の 2 種類を、LogReg と
   LightGBM(``fsgen.evaluation.classifiers`` と同じ設定)で、企業単位の 5 分割(乱数 42)で学習する。
   始点・終点は、その企業を学習に含まない fold のモデルで予測する。クラス重み均衡のスコアと、学習
   データの有病率で補正した確率の両方を出す。
2. 実行可能性: パネル(2021〜2024 年度、全上場企業)の 1・2・3 年の推移と、再建群 72 社
   (2022 → 2024)の Ψ の経験分布の中で、生成の Ψ(x_i, y_i*) が何パーセンタイルか。
3. 近傍の実在密度: 終点から Ψ(z − y) ≤ r の健全企業の数(自社を除く)。
4. 地図: 対称化した距離 δ(a, b) = Σ_a (c_a⁺ + c_a⁻)/2 · |a − b| の古典的 MDS(健全企業 1,500 社を
   ランドマーク、乱数 0)に、苦境スコアの等高線(2 次元での k 近傍平均)と移動の矢印を重ねる。

出力: ``results/stage76/feasibility_report.md``・``feasibility.json``・
``bs_manifold_mapping.png``・``.pdf``。

    uv run python scripts/stage76_feasibility_map.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import lightgbm as lgb
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evaluate_stage61 import GEN_Y  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tables_stage69 import INTERIM, setup_excl  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.classifiers import strict_sklearn_convergence  # noqa: E402

OUT = Path("results/stage76")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
SEED_CLF, N_FOLD, N_LAND, K_SMOOTH, B = 42, 5, 1500, 30, 2000
REAL_GROUP = {"c2": "class 2 → 回復", "c1": "class 1 → class 0"}
TARGETS = {"distress": "苦境(class 1・2)vs 健全", "gc": "GC(class 2)vs 健全"}
CLFS = ("logreg", "lightgbm")


def psi(d: np.ndarray, cp: np.ndarray, cm: np.ndarray) -> np.ndarray:
    """Ψ(Δ) を最後の軸で計算する。"""
    out: np.ndarray = (np.maximum(d, 0.0) * cp + np.maximum(-d, 0.0) * cm).sum(axis=-1)
    return out


def make_clf(kind: str) -> Any:
    if kind == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=200, num_leaves=15, random_state=SEED_CLF,
                                  n_jobs=1, class_weight="balanced", verbosity=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=SEED_CLF))


def panel_years(names: tuple[str, ...]) -> tuple[dict[int, pd.DataFrame], dict[int, pd.Series]]:
    """年度 → 企業 × 科目の構成(総資産比)と、年度 → 企業のラベル(書類があれば)。"""
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str, "docID": str})
    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str}).drop_duplicates(
        "docID").set_index("docID")["label"]
    comp: dict[int, pd.DataFrame] = {}
    labels: dict[int, pd.Series] = {}

    def frame(rows: pd.DataFrame, pre: str) -> pd.DataFrame:
        ta = rows[f"{pre}_total_assets"].to_numpy(float)
        v = rows[[f"{pre}_{n}" for n in names]].fillna(0.0).to_numpy(float)
        ok = np.isfinite(ta) & (ta > 0)
        return pd.DataFrame(v[ok] / ta[ok, None], index=rows["edinetCode"].to_numpy()[ok],
                            columns=list(names))

    for t in (2022, 2023, 2024):
        rows = p[p["t"] == t]
        comp[t] = frame(rows, "cur")
        labels[t] = pd.Series(pd.to_numeric(lab.reindex(rows["docID"]).to_numpy(),
                                            errors="coerce"), index=rows["edinetCode"].to_numpy())
    comp[2021] = frame(p[p["t"] == 2022], "prev")
    return comp, labels


def moves(comp: dict[int, pd.DataFrame], a: int, b: int, codes: list[str] | None = None
          ) -> tuple[np.ndarray, list[str]]:
    idx = comp[a].index.intersection(comp[b].index)
    if codes is not None:
        idx = idx.intersection(pd.Index(codes))
    d: np.ndarray = comp[b].loc[idx].to_numpy() - comp[a].loc[idx].to_numpy()
    return d, list(idx)


def boot_median_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    d = a - b
    bs = [float(np.median(d[rng.integers(0, len(d), len(d))])) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(np.median(d)), float(lo), float(hi)


def mds_fit(land: np.ndarray, wsym: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ランドマークの古典的 MDS。(射影行列, δ² の列平均, 上位 2 成分の寄与)。"""
    d2 = np.stack([(np.abs(land - p) * wsym).sum(axis=1) for p in land]) ** 2
    n = len(land)
    j = np.eye(n) - 1.0 / n
    bm = -0.5 * j @ d2 @ j
    ev, evec = np.linalg.eigh(bm)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    if ev[1] <= 0:
        raise ValueError("MDS の上位 2 固有値が正でない")
    share = float(ev[:2].sum() / ev[ev > 0].sum())
    proj: np.ndarray = evec[:, :2] / np.sqrt(ev[:2])
    return proj, d2.mean(axis=0), share


def mds_embed(pts: np.ndarray, land: np.ndarray, wsym: np.ndarray, proj: np.ndarray,
              mean_d2: np.ndarray) -> np.ndarray:
    """Gower の公式でランドマークの MDS に点を埋め込む。"""
    out = np.empty((len(pts), 2))
    for s in range(0, len(pts), 200):
        p = pts[s:s + 200]
        d2 = ((np.abs(p[:, None, :] - land[None, :, :]) * wsym).sum(axis=2)) ** 2
        out[s:s + 200] = 0.5 * (mean_d2[None, :] - d2) @ proj
    return out


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    setup_excl(names)
    x, lab, docs, _ = load_tilde(names)
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes = np.array([code_of.get(d, d) for d in docs])
    # 企業単位の 5 分割(乱数 42)
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

    def predict(t: str, k: str, pts: np.ndarray, folds: np.ndarray) -> np.ndarray:
        out = np.empty(len(pts))
        for f in range(N_FOLD):
            s = folds == f
            if s.any():
                out[s] = models[(t, k, f)].predict_proba(pts[s])[:, 1]
        return out

    def corrected(p: np.ndarray, t: str) -> np.ndarray:
        """クラス重み均衡のスコアを学習データの有病率に戻す(オッズ × n₁/n₀)。"""
        o = p / np.clip(1.0 - p, 1e-12, None) * prior[t]
        out: np.ndarray = o / (1.0 + o)
        return out

    from sklearn.metrics import roc_auc_score
    auc = {f"{t}|{k}": float(roc_auc_score(yvec[t][ymask[t]], oof[(t, k)][ymask[t]]))
           for t in TARGETS for k in CLFS}
    comp, ylabels = panel_years(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")[
        ["edinetCode", "t", "label_prev", "label_cur"]].astype({"edinetCode": str})
    res: dict[str, Any] = {"auc_oof": auc, "prior_odds": prior}
    healthy = lab == 0
    xh, ch = x[healthy], codes[healthy]
    rng0 = np.random.default_rng(0)
    land_i = rng0.choice(len(xh), size=N_LAND, replace=False)
    q_i = np.random.default_rng(1).choice(len(xh), size=2000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    lines = ["# ステージ76: デフォルト確率の低下・年次推移に基づく実行可能性・距離空間の地図", "",
             "生成は最終設定(ステージ69 の設定、λ̃ = 1、seed 0)。距離は最終設定の移動コスト Ψ_cls"
             "(ステージ60 の c±。利益剰余金・評価差額金・非支配株主持分は 0)。"
             f"生成の終点は {GEN_Y:g} 年後とみなしている(`evaluate_stage61.GEN_Y`)。", ""]
    for ci_, cls in enumerate(("c2", "c1")):
        meta = json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(encoding="utf-8"))
        cp, cm = np.array(meta["c_plus"], float), np.array(meta["c_minus"], float)
        e51._GEN = INTERIM / "stage67/all"
        g = e51.load_gen(cls, "s1__l1")
        g = g[g["seed"] == 0].reset_index(drop=True)
        x0 = g[[f"x0_{n}" for n in names]].to_numpy(float)
        y1 = g[[f"x1_{n}" for n in names]].to_numpy(float)
        gcode = np.array([code_of.get(str(d), str(d)) for d in g["docID"]])
        gfold = np.array([fold_of_code[c] for c in gcode])
        n_g = len(g)
        r: dict[str, Any] = {"n": n_g}
        # 1. 苦境の確率
        pd_rows = []
        for t in TARGETS:
            for k in CLFS:
                p0, p1 = predict(t, k, x0, gfold), predict(t, k, y1, gfold)
                q0, q1 = corrected(p0, t), corrected(p1, t)
                hz = oof[(t, k)][healthy]
                pd_rows.append({
                    "target": t, "clf": k,
                    "score_start_med": float(np.median(p0)), "score_end_med": float(np.median(p1)),
                    "score_start_mean": float(p0.mean()), "score_end_mean": float(p1.mean()),
                    "prob_start_med": float(np.median(q0)), "prob_end_med": float(np.median(q1)),
                    "prob_start_mean": float(q0.mean()), "prob_end_mean": float(q1.mean()),
                    "d_score": boot_median_diff(p1, p0), "share_down": float((p1 < p0).mean()),
                    "end_below_healthy_med": float((p1 <= np.median(hz)).mean()),
                    "healthy_med": float(np.median(hz)),
                    "healthy_prob_med": float(np.median(corrected(hz, t)))})
        r["pd"] = pd_rows
        # 2. 実行可能性
        c_gen = psi(y1 - x0, cp, cm)
        ref: dict[str, np.ndarray] = {}
        for h, pairs_y in ((1, ((2021, 2022), (2022, 2023), (2023, 2024))),
                           (2, ((2021, 2023), (2022, 2024))), (3, ((2021, 2024),))):
            ref[f"all_h{h}"] = np.concatenate([psi(moves(comp, a, b)[0], cp, cm)
                                               for a, b in pairs_y])
        rec_codes = pairs[pairs["group"] == REAL_GROUP[cls]]["edinetCode"].tolist()
        all_rec = pairs["edinetCode"].tolist()
        d_rec, _ = moves(comp, 2022, 2024, rec_codes)
        ref["rec_group"] = psi(d_rec, cp, cm)
        d72, _ = moves(comp, 2022, 2024, all_rec)
        ref["rec72"] = psi(d72, cp, cm)
        feas: dict[str, dict[str, float]] = {}
        for key, v in ref.items():
            v = np.sort(v)
            pct = np.searchsorted(v, c_gen, side="right") / len(v)
            feas[key] = {"n": int(len(v)), "q50": float(np.median(v)),
                         "q95": float(np.quantile(v, 0.95)), "q99": float(np.quantile(v, 0.99)),
                         "pct_med": float(np.median(pct)), "pct_p90": float(np.quantile(pct, 0.9)),
                         "in95": float((c_gen <= np.quantile(v, 0.95)).mean()),
                         "in99": float((c_gen <= np.quantile(v, 0.99)).mean()),
                         "surv_med": float(np.median(1.0 - pct))}
        r["feas"] = feas
        r["c_gen"] = {"q50": float(np.median(c_gen)), "q90": float(np.quantile(c_gen, 0.9)),
                      "max": float(c_gen.max())}
        # コストと回復の対応(実データ: 始点で苦境の企業の 1・2 年の推移)
        rows_cr = []
        lt_ = {t: r38[r38["t"] == t].set_index("edinetCode") for t in (2023, 2024)}
        lab_y = {2022: lt_[2023]["label_prev"], 2023: lt_[2023]["label_cur"],
                 2024: lt_[2024]["label_cur"]}
        for a, b in ((2022, 2023), (2023, 2024), (2022, 2024)):
            d, cc = moves(comp, a, b)
            la, lb = lab_y[a].reindex(cc).to_numpy(), lab_y[b].reindex(cc).to_numpy()
            ok = (la >= 1) & np.isfinite(lb)
            rows_cr.append(pd.DataFrame({"h": b - a, "cost": psi(d[ok], cp, cm),
                                         "recovered": (lb[ok] == 0).astype(float)}))
        cr = pd.concat(rows_cr, ignore_index=True)
        cr_tab = []
        for h in (1, 2):
            s = cr[cr["h"] == h]
            if len(s) < 10:
                continue
            qb = pd.qcut(s["cost"], 4, duplicates="drop")
            for iv, gg in s.groupby(qb, observed=True):
                cr_tab.append({"h": h, "bin": f"{iv.left:.3g}〜{iv.right:.3g}", "n": len(gg),
                               "recovered": float(gg["recovered"].mean())})
        r["cost_recovery"] = cr_tab
        r["n_cost_recovery"] = {int(h): int((cr["h"] == h).sum()) for h in (1, 2)}
        # 方向: 生成の Δ と、同じ群の実際の回復の平均 Δ の余弦
        mean_rec = d_rec.mean(axis=0)
        dg = y1 - x0

        def cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            out: np.ndarray = (u @ v) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v) + 1e-15)
            return out
        loo = np.array([cos(d_rec[i], (mean_rec * len(d_rec) - d_rec[i]) / (len(d_rec) - 1))
                        for i in range(len(d_rec))])
        r["direction"] = {"cos_gen_med": float(np.median(cos(dg, mean_rec))),
                          "cos_gen_pos": float((cos(dg, mean_rec) > 0).mean()),
                          "cos_real_loo_med": float(np.median(loo)),
                          "cos_real_loo_pos": float((loo > 0).mean()),
                          "n_rec": int(len(d_rec))}
        # 3. 近傍の実在密度(Ψ(z − y)、自社の書類を除く)
        nn = np.empty(len(q_i))
        for s in range(0, len(q_i), 100):
            qq = q_i[s:s + 100]
            dd = psi(xh[None, :, :] - xh[qq][:, None, :], cp, cm)
            dd[ch[None, :] == ch[qq][:, None]] = np.inf
            nn[s:s + 100] = dd.min(axis=1)
        radii = {"r50": float(np.median(nn)), "r90": float(np.quantile(nn, 0.9))}

        def count(pts: np.ndarray, own: np.ndarray, rad: float, cp: np.ndarray = cp,
                  cm: np.ndarray = cm) -> np.ndarray:
            dd = psi(xh[None, :, :] - pts[:, None, :], cp, cm)
            dd[ch[None, :] == own[:, None]] = np.inf
            out = np.array([len(set(ch[row <= rad])) for row in dd])
            return out
        dens: dict[str, Any] = {"radii": radii}
        hq = xh[q_i[:300]]
        for rk, rad in radii.items():
            ce, cs = count(y1, gcode, rad), count(x0, gcode, rad)
            ch_ = count(hq, ch[q_i[:300]], rad)
            dens[rk] = {"end_med": float(np.median(ce)), "end_zero": float((ce == 0).mean()),
                        "start_med": float(np.median(cs)), "start_zero": float((cs == 0).mean()),
                        "healthy_med": float(np.median(ch_)),
                        "healthy_zero": float((ch_ == 0).mean())}
        r["density"] = dens
        # 4. 地図
        wsym = 0.5 * (cp + cm)
        land = xh[land_i]
        proj, mean_d2, share = mds_fit(land, wsym)
        r["mds_share"] = share
        e_all = mds_embed(x, land, wsym, proj, mean_d2)
        e0, e1 = mds_embed(x0, land, wsym, proj, mean_d2), mds_embed(y1, land, wsym, proj, mean_d2)
        _, rc = moves(comp, 2022, 2024, rec_codes)
        ra = mds_embed(comp[2022].loc[rc].to_numpy(), land, wsym, proj, mean_d2)
        rb = mds_embed(comp[2024].loc[rc].to_numpy(), land, wsym, proj, mean_d2)
        ax = axes[ci_]
        pts = np.vstack([e0, e1, ra, rb, e_all[healthy]])
        lo_, hi_ = np.quantile(pts, 0.01, axis=0), np.quantile(pts, 0.99, axis=0)
        pad = 0.08 * (hi_ - lo_)
        lo_, hi_ = lo_ - pad, hi_ + pad
        gx, gy = np.meshgrid(np.linspace(lo_[0], hi_[0], 120), np.linspace(lo_[1], hi_[1], 120))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        score = oof[("distress", "lightgbm")]
        zz = np.empty(len(grid))
        dmin = np.empty(len(grid))
        for s in range(0, len(grid), 600):
            dd = np.linalg.norm(grid[s:s + 600, None, :] - e_all[None, :, :], axis=2)
            kk = np.argpartition(dd, K_SMOOTH, axis=1)[:, :K_SMOOTH]
            zz[s:s + 600] = score[kk].mean(axis=1)
            dmin[s:s + 600] = dd.min(axis=1)
        far = dmin > np.quantile(dmin, 0.85)
        zz[far] = np.nan
        cf = ax.contourf(gx, gy, zz.reshape(gx.shape), levels=np.linspace(0, 1, 11),
                         cmap="RdYlGn_r", alpha=0.35)
        ax.contour(gx, gy, zz.reshape(gx.shape), levels=[0.5], colors="#7a1f1f",
                   linewidths=1.0, linestyles="--")
        ax.scatter(*e_all[healthy].T, s=3, c="#4a6b8a", alpha=0.25, linewidths=0,
                   label="Healthy firms (class 0)")
        # quiver は軸の外を切り取る(annotate の矢印は切り取られない)
        ax.quiver(*ra.T, *(rb - ra).T, angles="xy", scale_units="xy", scale=1.0,
                  color="#1b7837", width=0.0035, alpha=0.9, zorder=2)
        ax.quiver(*e0.T, *(e1 - e0).T, angles="xy", scale_units="xy", scale=1.0,
                  color="#762a83", width=0.0022, alpha=0.6, zorder=2)
        ax.scatter(*e0.T, s=14, c="#d6604d", edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"Distressed start x_i (class {cls[1]}, n={n_g})")
        ax.scatter(*e1.T, s=14, c="#762a83", edgecolors="white", linewidths=0.4, zorder=3,
                   label="Generated endpoint y_i*")
        ax.plot([], [], color="#1b7837", lw=1.3,
                label=f"Real recovery 2022→2024 ({REAL_GROUP[cls].replace('回復', 'recovered')}, "
                      f"n={len(rc)})")
        ax.set_xlim(lo_[0], hi_[0])
        ax.set_ylim(lo_[1], hi_[1])
        ax.set_title(f"({'ab'[ci_]}) Class {cls[1]} sources, Ψ_c{cls[1]} metric "
                     f"(2-D share {share:.0%})", fontsize=10, loc="left")
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        res[cls] = r
        print(f"[{cls}] 完了(MDS の 2 次元の寄与 {share:.0%})", flush=True)
    fig.suptitle("Classical MDS of the symmetrised movement cost (1,500 healthy landmarks); "
                 "contours = distress score", fontsize=11, x=0.45)
    cb = fig.colorbar(cf, ax=axes, shrink=0.75, pad=0.01)
    cb.set_label("Distress score (LightGBM, balanced, out-of-fold; k-NN mean in 2-D)")
    fig.savefig(OUT / "bs_manifold_mapping.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "bs_manifold_mapping.pdf", bbox_inches="tight")
    # 表
    lines += ["## 0. 分類器(企業単位の 5 分割、乱数 42、out-of-fold の AUC)", "",
              "| 目的変数 | LogReg | LightGBM | 学習データの n₁/n₀ |", "|---|---:|---:|---:|"]
    for t, tl in TARGETS.items():
        lines.append(f"| {tl} | {auc[f'{t}|logreg']:.3f} | {auc[f'{t}|lightgbm']:.3f} "
                     f"| {prior[t]:.4f} |")
    lines += ["", "- スコア = クラス重み均衡の分類器の出力。確率 = 学習データの有病率に戻したもの"
              "(オッズ × n₁/n₀)。**どちらも「この構成の企業が苦境の書類である確率」で、"
              "実際の倒産確率ではない。**", ""]
    for cls in ("c2", "c1"):
        r = res[cls]
        lines += [f"## class {cls[1]}(始点 {r['n']} 社)", "",
                  "### 1. 苦境の確率の変化(始点 → 終点)", "",
                  "| 目的変数 | 分類器 | スコア 中央値 | スコア 平均 | 確率 中央値 | 確率 平均 "
                  "| スコアの差 中央値 [95%] | 下がった割合 | 終点 ≤ 健全企業の中央値 |",
                  "|---|---|---|---|---|---|---|---:|---:|"]
        for p in r["pd"]:
            d = p["d_score"]
            lines.append(
                f"| {TARGETS[p['target']]} | {p['clf']} "
                f"| {p['score_start_med']:.3f} → {p['score_end_med']:.3f} "
                f"| {p['score_start_mean']:.3f} → {p['score_end_mean']:.3f} "
                f"| {p['prob_start_med']:.2%} → {p['prob_end_med']:.2%} "
                f"| {p['prob_start_mean']:.2%} → {p['prob_end_mean']:.2%} "
                f"| {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {p['share_down']:.0%} "
                f"| {p['end_below_healthy_med']:.0%}(健全 {p['healthy_med']:.3f}) |")
        f = r["feas"]
        lines += ["", "### 2. 年次推移に基づく実行可能性", "",
                  f"生成の Ψ: 中央値 {r['c_gen']['q50']:.3g}、90% 点 {r['c_gen']['q90']:.3g}、"
                  f"最大 {r['c_gen']['max']:.3g}。", "",
                  "| 実在の推移の分布 | 件数 | 中央値 | 95% 点 | 生成のパーセンタイル 中央値 "
                  "/ 90% 点 | 95% 点以内 | 99% 点以内 | 1 − F̂(c) の中央値 |",
                  "|---|---:|---:|---:|---|---:|---:|---:|"]
        lab_f = {"all_h1": "全上場企業 1 年", "all_h2": "全上場企業 2 年",
                 "all_h3": "全上場企業 3 年",
                 "rec_group": f"再建群({REAL_GROUP[cls]})2022 → 2024",
                 "rec72": "再建群 72 社 2022 → 2024"}
        for key, lt in lab_f.items():
            v = f[key]
            lines.append(f"| {lt} | {v['n']} | {v['q50']:.3g} | {v['q95']:.3g} "
                         f"| {v['pct_med']:.0%} / {v['pct_p90']:.0%} | {v['in95']:.0%} "
                         f"| {v['in99']:.0%} | {v['surv_med']:.0%} |")
        ncr = r["n_cost_recovery"]
        lines += ["", "コストと回復の対応(実データ: 始点で class 1・2 の企業の推移、"
                  "ラベルはステージ38 の遷移、Ψ の四分位ごとの終点で class 0 の割合。"
                  f"件数 1 年 {ncr[1]}・2 年 {ncr[2]}):", "",
                  "| 期間 | Ψ の区間 | 件数 | 回復した割合 |", "|---:|---|---:|---:|"]
        for row in r["cost_recovery"]:
            lines.append(f"| {row['h']} 年 | {row['bin']} | {row['n']} | {row['recovered']:.0%} |")
        dr = r["direction"]
        lines += ["", f"方向: 生成の Δ と、同じ群の実際の回復({dr['n_rec']} 社)の平均 Δ の"
                  f"余弦の中央値 "
                  f"{dr['cos_gen_med']:+.2f}(正 {dr['cos_gen_pos']:.0%})。参考: 実際の回復企業"
                  f"どうし(自分を除く平均との余弦){dr['cos_real_loo_med']:+.2f}(正 "
                  f"{dr['cos_real_loo_pos']:.0%})。", ""]
        dn = r["density"]
        lines += ["### 3. 近傍の実在密度(Ψ(z − y) ≤ r の健全企業の数、自社を除く)", "",
                  f"r は健全企業どうしの最近傍の Ψ の中央値(r50 = {dn['radii']['r50']:.3g})と "
                  f"90% 点(r90 = {dn['radii']['r90']:.3g})。健全企業は 300 社の標本。", "",
                  "| 半径 | 終点 中央値(0 社の割合) | 始点 中央値(0 社) | 健全企業 中央値(0 社) |",
                  "|---|---|---|---|"]
        for rk in ("r50", "r90"):
            v = dn[rk]
            lines.append(f"| {rk} | {v['end_med']:.0f}({v['end_zero']:.0%}) "
                         f"| {v['start_med']:.0f}({v['start_zero']:.0%}) "
                         f"| {v['healthy_med']:.0f}({v['healthy_zero']:.0%}) |")
        lines += ["", f"地図: 古典的 MDS の上位 2 成分の寄与 {r['mds_share']:.0%}"
                  "(正の固有値の和に対する比)。", ""]
    (OUT / "feasibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "feasibility.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                     default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
