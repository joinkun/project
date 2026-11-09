#!/usr/bin/env python
"""ステージ80: A1・SP-F1(単一の生成)とステージ69 の設定の検算(lightgbm・sklearn のプロセス)。

規則は ``results/stage80/criteria.md`` §3・§4。単一の生成の λ̃ は、D1(``results/stage80/d1/
summary.json``)の A1・SP-F1 の λ̃*_f の中央値。分類器・尺度・半径はステージ76〜78 と同じ。

1. 表 1: ゼロの科目数(「ちょうど 0」と |v| < 10⁻⁶)と識別 AUC(候補・終点、終点は丸めた版も)。
2. 苦境の確率(LogReg・LightGBM、企業単位の 5 分割の out-of-fold)とロジットの中央値。
3. 実行可能性・空白地帯率: Ψ60(ステージ60 の c±)と Ψ_A1(57 社の全体の A1、利益剰余金 0)。
4. 回復率の四分位: Ψ_A1(と比較に Ψ60_c2)で、合算・class 2・class 1、1 年・2 年。
5. 地図: Ψ_A1 の古典的 MDS と、LogReg のロジット × MDS 1 の残差(``bs_manifold_mapping_a1spf1``)。
6. BS 表: class 2 の 5 社(ステージ65〜79 と同じ)と、15 項目のチェックで違反した項目。

出力: ``results/stage80/final.md``・``final.json``・``bs_a1spf1.md``・
``bs_manifold_mapping_a1spf1.png``・``.pdf``。

    uv run python scripts/stage80_final.py
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
from evaluate_stage65 import bs_table  # noqa: E402
from stage72_evaluate import ci  # noqa: E402
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
from tables_stage69 import INTERIM, S67, Src, metrics, setup_excl  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import subtotal_members  # noqa: E402
from fsgen.evaluation.classifiers import strict_sklearn_convergence  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402
from fsgen.viz.balance_sheet import load_labels  # noqa: E402

OUT = Path("results/stage80")
D80 = INTERIM / "stage80"
MAIN = "A1・SP-F1(主提案、γ' = 0.1)"
_RE = "jppfs_cor_RetainedEarnings"
ZERO_TOL, K_SMOOTH = 1e-6, 30


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = json.loads((OUT / "d1/summary.json").read_text(encoding="utf-8"))
    stars = [v for v in d1["res"][MAIN]["star"].values() if v is not None]
    if len(stars) != 5:
        raise ValueError("A1・SP-F1 に較正できない fold がある")
    lam = float(np.median(stars))
    src = {"s69": Src(*S67, "s1__l1", "1"),
           "a1sp": Src(D80 / "metrics_full", D80 / "gen_full", f"A1_sp__l{lam:g}", f"{lam:g}")}
    lab_s = {"s69": "ステージ69 の設定(λ̃ = 1)", "a1sp": f"A1・SP-F1(λ̃ = {lam:g})"}
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    setup_excl(names)
    x, lab, docs, _ = load_tilde(names)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes = np.array([code_of.get(d, d) for d in docs])
    healthy = lab == 0
    xh, ch = x[healthy], codes[healthy]
    w79 = json.loads((INTERIM / "stage79/weights.json").read_text(encoding="utf-8"))
    ca1 = (np.array(w79["configs"]["full"]["A1"]["plus"], float),
           np.array(w79["configs"]["full"]["A1"]["minus"], float))
    for c_ in ca1:
        c_[names.index(_RE)] = 0.0
    c60 = {cls: tuple(np.array(json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(
        encoding="utf-8"))[k], float) for k in ("c_plus", "c_minus")) for cls in ("c2", "c1")}

    def gen(key: str, cls: str) -> pd.DataFrame:
        e51._GEN = src["s69"].gen
        g0 = e51.load_gen(cls, src["s69"].setting)
        order = g0[g0["seed"] == 0]["docID"]
        e51._GEN = src[key].gen
        g = e51.load_gen(cls, src[key].setting)
        out: pd.DataFrame = g[g["seed"] == 0].set_index("docID").loc[order]
        return out

    def arrays(key: str, cls: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = gen(key, cls)
        return (g[[f"x0_{n}" for n in names]].to_numpy(float),
                g[[f"x1_{n}" for n in names]].to_numpy(float),
                np.array([code_of.get(str(d), str(d)) for d in g.index]))

    res: dict[str, Any] = {"lambda_tilde": lam, "fold_stars": stars}
    lines = ["# ステージ80: A1・SP-F1(単一の生成)とステージ69 の設定の検算", "",
             f"単一の生成: 57 社の全体の A1 のコスト、γ' = 0.1 のスパース OT、λ̃ = {lam:g}"
             f"(D1 の A1・SP-F1 の λ̃*_f = {', '.join(f'{v:g}' for v in stars)} の中央値)。", ""]
    # 1. 表 1
    rng = np.random.default_rng(0)
    i1 = rng.integers(0, len(xt), 10_000)
    i2 = (i1 + rng.integers(1, len(xt), 10_000)) % len(xt)

    def snap(u: np.ndarray) -> np.ndarray:
        return np.where(np.abs(u) < ZERO_TOL, 0.0, u)

    def auc(v: np.ndarray, real: np.ndarray) -> tuple[float, float]:
        r = identification_auc_repeated(v, real, 3)
        return float(r["lightgbm"].mean()), float(r["logreg"].mean())
    t1_rows: list[tuple[str, np.ndarray, str]] = [
        ("健全企業(class 0、実在)", xt, "none"),
        ("2 社の平均(異なる 2 社、10,000 組)", 0.5 * (xt[i1] + xt[i2]), "raw")]
    for cls in ("c2", "c1"):
        z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        t1_rows.append((f"バランス OT の重心({cls})", np.asarray(z["bal_e0.001__bary"], float),
                        "raw"))
    sp_stats: dict[str, dict[str, float]] = {}
    for cls in ("c2", "c1"):
        pz = np.load(D80 / f"plan_{cls}_full.npz", allow_pickle=True)
        idx, pt = np.asarray(pz["A1sp__idx"]), np.asarray(pz["A1sp__pt"])
        nz = (pt > 0).sum(axis=1)
        sp_stats[cls] = {"s_med": float(np.median(nz)), "s_max": int(nz.max()),
                         "ess": float(np.median(1.0 / (pt ** 2).sum(axis=1)))}
        t1_rows.append((f"A1・SP-F1 の候補 S(i)({cls}、QP の前)", xt[idx[pt > 0]], "raw"))
        for key in ("s69", "a1sp"):
            y = arrays(key, cls)[1]
            t1_rows.append((f"終点: {lab_s[key]}({cls})", y, "raw"))
            t1_rows.append((f"終点: {lab_s[key]}({cls})、両側で丸め", y, "snap"))
    lines += ["## 1. 表 1(命題 2): ゼロの科目数と識別 AUC", "",
              "A1・SP-F1 の候補(57 社の全体): |S(i)| の中央値 / 最大 / 実効サポートは "
              + "、".join(f"{c} {v['s_med']:.1f} / {v['s_max']} / {v['ess']:.2f}"
                         for c, v in sp_stats.items())
              + "。命題 1 の A1 での検算は `verify_a1.md`。",
              "", "| 対象 | 件数 | ちょうど 0 | \\|v\\| < 10⁻⁶ | AUC LightGBM | AUC LogReg |",
              "|---|---:|---:|---:|---:|---:|"]
    t1: dict[str, Any] = {}
    for lt, v, mode in t1_rows:
        z0 = float((v == 0.0).sum(axis=1).mean())
        zt = float((np.abs(v) < ZERO_TOL).sum(axis=1).mean())
        a = (float("nan"), float("nan")) if mode == "none" else (
            auc(snap(v), snap(xt)) if mode == "snap" else auc(v, xt))
        t1[lt] = {"n": len(v), "zero": z0, "zero_tol": zt, "auc": a}
        cell = "— | —" if mode == "none" else f"{a[0]:.3f} | {a[1]:.3f}"
        lines.append(f"| {lt} | {len(v)} | {z0:.2f} | {zt:.2f} | {cell} |")
        print(f"  表 1: {lt}", flush=True)
    res["table1"], res["sparse_full"] = t1, sp_stats
    lines.append("")
    # 2. 苦境の確率
    uc = np.random.default_rng(SEED_CLF).permutation(np.unique(codes))
    fold_of = {c: i for i, part in enumerate(np.array_split(uc, N_FOLD)) for c in part}
    fold = np.array([fold_of[c] for c in codes])
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
                oof[(t, k)][fold == f] = clf.predict_proba(x[fold == f])[:, 1]

    def predict(t: str, k: str, pts: np.ndarray, folds: np.ndarray) -> np.ndarray:
        out = np.empty(len(pts))
        for f in range(N_FOLD):
            s = folds == f
            if s.any():
                out[s] = models[(t, k, f)].predict_proba(pts[s])[:, 1]
        return out

    def logit(p: np.ndarray) -> np.ndarray:
        q = np.clip(p, 1e-12, 1 - 1e-12)
        out: np.ndarray = np.log(q / (1 - q))
        return out

    def corrected(p: np.ndarray, t: str) -> np.ndarray:
        o = p / np.clip(1.0 - p, 1e-12, None) * prior[t]
        out: np.ndarray = o / (1.0 + o)
        return out
    lines += ["## 2. 苦境の確率(始点 → ステージ69 → A1・SP-F1)", "",
              "確率 = クラス重み均衡のスコアを学習データの有病率に戻したもの"
              "(実際の倒産確率ではない)。"
              "ロジット = 均衡スコアの log(p/(1 − p))。", "",
              "| class | 目的変数 | 分類器 | 確率の中央値: 始点 → 69 → A1・SP "
              "| ロジットの中央値: 始点 → 69 "
              "→ A1・SP | 終点 ≤ 健全の中央値(69 / A1・SP) | A1・SP − 69 のスコアの差 [95%] |",
              "|---|---|---|---|---|---|---|"]
    pd_res: list[dict[str, Any]] = []
    for cls in ("c2", "c1"):
        x0, y69, gc = arrays("s69", cls)
        ya = arrays("a1sp", cls)[1]
        gf = np.array([fold_of[c] for c in gc])
        for t in TARGETS:
            for k in CLFS:
                p0, p69, pa = (predict(t, k, v, gf) for v in (x0, y69, ya))
                hz = float(np.median(oof[(t, k)][healthy]))
                row: dict[str, Any] = {"cls": cls, "target": t, "clf": k,
                                       "prob": [float(np.median(corrected(v, t)))
                                                for v in (p0, p69, pa)],
                                       "logit": [float(np.median(logit(v))) for v in (p0, p69, pa)],
                                       "below": [float((v <= hz).mean()) for v in (p69, pa)],
                                       "d": boot_median_diff(pa, p69)}
                pd_res.append(row)
                lines.append(f"| {cls} | {TARGETS[t]} | {k} | "
                             + " → ".join(f"{v:.2%}" for v in row["prob"]) + " | "
                             + " → ".join(f"{v:+.2f}" for v in row["logit"])
                             + f" | {row['below'][0]:.0%} / {row['below'][1]:.0%} "
                             f"| {ci(row['d'])} |")
    res["pd"] = pd_res
    lines.append("")
    # 3. 実行可能性・空白地帯率
    comp, _ = panel_years(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    q_i = np.random.default_rng(1).choice(len(xh), size=2000, replace=False)
    lines += ["## 3. 実行可能性と空白地帯率(Ψ60 と Ψ_A1)", "",
              "空白地帯率 = 健全企業(自社を除く)が Ψ(z − y) ≤ r90 に 1 社もない割合。"
              "r90 は同じ尺度で健全企業 2,000 社の最近傍の 90% 点。"
              "生成の終点は 5 年後とみなし、実在は最長 3 年。", "",
              "| class | 尺度 | 設定 | 生成の Ψ 中央値 | 1 年: 95% / 99% 点以内 "
              "| 3 年: 95% / 99% 点以内 "
              "| 再建群 72 社(2 年): 95% / 99% | 空白地帯率 | 健全企業自身 |",
              "|---|---|---|---:|---|---|---|---:|---:|"]
    feas: dict[str, Any] = {}
    for cls in ("c2", "c1"):
        for mk, (cp, cm) in (("Ψ60", c60[cls]), ("Ψ_A1", ca1)):
            ref = {"h1": np.concatenate([psi(moves(comp, a, b)[0], cp, cm) for a, b in
                                         ((2021, 2022), (2022, 2023), (2023, 2024))]),
                   "h3": psi(moves(comp, 2021, 2024)[0], cp, cm),
                   "rec72": psi(moves(comp, 2022, 2024, pairs["edinetCode"].tolist())[0], cp, cm)}
            nn = np.empty(len(q_i))
            for s in range(0, len(q_i), 100):
                qq = q_i[s:s + 100]
                dd = psi(xh[None, :, :] - xh[qq][:, None, :], cp, cm)
                dd[ch[None, :] == ch[qq][:, None]] = np.inf
                nn[s:s + 100] = dd.min(axis=1)
            r90 = float(np.quantile(nn, 0.9))

            def empty(pts: np.ndarray, own: np.ndarray, cp: np.ndarray = cp,
                      cm: np.ndarray = cm, r90: float = r90) -> float:
                dd = psi(xh[None, :, :] - pts[:, None, :], cp, cm)
                dd[ch[None, :] == own[:, None]] = np.inf
                return float((dd.min(axis=1) > r90).mean())
            hz_e = empty(xh[q_i[:300]], ch[q_i[:300]])
            x0, _, gc = arrays("s69", cls)
            rows_k: list[tuple[str, np.ndarray | None]] = [("始点", None)]
            rows_k += [(lab_s[k], arrays(k, cls)[1]) for k in ("s69", "a1sp")]
            for lk, yy in rows_k:
                if yy is None:
                    lines.append(f"| {cls} | {mk} | 始点 | — | — | — | — "
                                 f"| {empty(x0, gc):.0%} | {hz_e:.0%} |")
                    continue
                cg = psi(yy - x0, cp, cm)
                cells: list[str] = []
                fk: dict[str, Any] = {"gen_q50": float(np.median(cg)), "empty": empty(yy, gc)}
                for rk in ("h1", "h3", "rec72"):
                    q95, q99 = np.quantile(ref[rk], [0.95, 0.99])
                    fk[rk] = (float((cg <= q95).mean()), float((cg <= q99).mean()))
                    cells.append(f"{fk[rk][0]:.0%} / {fk[rk][1]:.0%}")
                feas[f"{cls}|{mk}|{lk}"] = fk
                lines.append(f"| {cls} | {mk} | {lk} | {fk['gen_q50']:.3g} | " + " | ".join(cells)
                             + f" | {fk['empty']:.0%} | {hz_e:.0%} |")
            print(f"  表 4: {cls} {mk}", flush=True)
    res["feasibility"] = feas
    lines.append("")
    # 4. 回復率の四分位
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")[
        ["edinetCode", "t", "label_prev", "label_cur"]].astype({"edinetCode": str})
    lt_ = {t: r38[r38["t"] == t].set_index("edinetCode") for t in (2023, 2024)}
    lab_y = {2022: lt_[2023]["label_prev"], 2023: lt_[2023]["label_cur"],
             2024: lt_[2024]["label_cur"]}
    lines += ["## 4. 実データの移動コストの四分位と回復の割合", "",
              "回復 = 終点で class 0(ラベルはステージ38 の遷移)。"
              "1 年 = 2022 → 2023 と 2023 → 2024 の"
              "合算、2 年 = 2022 → 2024。四分位はそれぞれの母集団の中で切った。", "",
              "| 尺度 | 始点 | 期間 | 件数(四分位ごと) | Q1 | Q2 | Q3 | Q4 | 単調減少 |",
              "|---|---|---|---|---:|---:|---:|---:|---|"]
    quart: dict[str, Any] = {}
    for sk, (cp, cm) in (("Ψ_A1", ca1), ("Ψ60_c2", c60["c2"])):
        frames = []
        for ya_, yb_ in ((2022, 2023), (2023, 2024), (2022, 2024)):
            d, cc = moves(comp, ya_, yb_)
            la, lb = lab_y[ya_].reindex(cc).to_numpy(), lab_y[yb_].reindex(cc).to_numpy()
            ok = (la >= 1) & np.isfinite(lb)
            frames.append(pd.DataFrame({"h": yb_ - ya_, "start": la[ok],
                                        "cost": psi(d[ok], cp, cm),
                                        "rec": (lb[ok] == 0).astype(float)}))
        cr = pd.concat(frames, ignore_index=True)
        for sl, sel in (("class 1・2", cr["start"] >= 1), ("class 2", cr["start"] == 2),
                        ("class 1", cr["start"] == 1)):
            for h in (1, 2):
                s = cr[sel & (cr["h"] == h)]
                qb = pd.qcut(s["cost"], 4, labels=False, duplicates="drop")
                rate = s.groupby(qb)["rec"].mean().to_numpy()
                nq = s.groupby(qb).size().to_numpy()
                mono = bool(np.all(np.diff(rate) < 0))
                quart[f"{sk}|{sl}|{h}"] = {"n": len(s), "rates": rate.tolist(), "mono": mono}
                lines.append(f"| {sk} | {sl} | {h} 年 | {len(s)}({'/'.join(map(str, nq))}) | "
                             + " | ".join(f"{v:.0%}" for v in rate)
                             + f" | {'はい' if mono else 'いいえ'} |")
    res["quartiles"] = quart
    lines.append("")
    # 5. 地図(Ψ_A1)
    axis_clf = make_clf("logreg")
    with strict_sklearn_convergence():
        axis_clf.fit(x, (lab >= 1).astype(int))
    score = oof[("distress", "lightgbm")]
    wsym = 0.5 * (ca1[0] + ca1[1])
    land = xh[np.random.default_rng(0).choice(len(xh), size=N_LAND, replace=False)]
    proj, mean_d2, share = mds_fit(land, wsym)
    e_all = mds_embed(x, land, wsym, proj, mean_d2)
    la_all = axis_clf.decision_function(x)
    beta = np.polyfit(la_all, e_all[:, 0], 1)
    p_all = np.column_stack([la_all, e_all[:, 0] - np.polyval(beta, la_all)])
    fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
    cf = None
    for ci_, cls in enumerate(("c2", "c1")):
        x0, ya, _ = arrays("a1sp", cls)
        rc = [c for c in pairs[pairs["group"] == REAL_GROUP[cls]]["edinetCode"]
              if c in comp[2022].index and c in comp[2024].index]
        xa, xb = comp[2022].loc[rc].to_numpy(), comp[2024].loc[rc].to_numpy()
        emb = {k: mds_embed(v, land, wsym, proj, mean_d2)
               for k, v in (("x0", x0), ("y1", ya), ("xa", xa), ("xb", xb))}

        def plane(pts: np.ndarray, em: np.ndarray, beta: np.ndarray = beta) -> np.ndarray:
            lx = axis_clf.decision_function(pts)
            return np.column_stack([lx, em[:, 0] - np.polyval(beta, lx)])
        pl = {k: plane(v, emb[k]) for k, v in (("x0", x0), ("y1", ya), ("xa", xa), ("xb", xb))}
        res[f"map_{cls}"] = {"logit_start_med": float(np.median(pl["x0"][:, 0])),
                             "logit_end_med": float(np.median(pl["y1"][:, 0])),
                             "logit_real_start_med": float(np.median(pl["xa"][:, 0])),
                             "logit_real_end_med": float(np.median(pl["xb"][:, 0])),
                             "logit_healthy_med": float(np.median(la_all[healthy])),
                             "mds_len_gen_med": float(np.median(np.linalg.norm(
                                 emb["y1"] - emb["x0"], axis=1))),
                             "mds_len_real_med": float(np.median(np.linalg.norm(
                                 emb["xb"] - emb["xa"], axis=1)))}
        for ri, (ea, ee, title) in enumerate((
                (e_all, emb, f"Classical MDS of Ψ_A1 (2-D share {share:.0%})"),
                (p_all, pl, "Distress logit (LogReg) × MDS 1 residual (Ψ_A1)"))):
            ax = axes[ri, ci_]
            pts = np.vstack([ee["x0"], ee["y1"], ee["xa"], ee["xb"], ea[healthy]])
            lo_, hi_ = np.quantile(pts, 0.01, axis=0), np.quantile(pts, 0.99, axis=0)
            pad = 0.08 * (hi_ - lo_)
            lo_, hi_ = lo_ - pad, hi_ + pad
            gx, gy = np.meshgrid(np.linspace(lo_[0], hi_[0], 120),
                                 np.linspace(lo_[1], hi_[1], 120))
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            sc = hi_ - lo_
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
            ax.quiver(*ee["xa"].T, *(ee["xb"] - ee["xa"]).T, angles="xy", scale_units="xy",
                      scale=1.0, color="#1b7837", width=0.0035, alpha=0.9, zorder=2)
            ax.quiver(*ee["x0"].T, *(ee["y1"] - ee["x0"]).T, angles="xy", scale_units="xy",
                      scale=1.0, color="#762a83", width=0.0022, alpha=0.6, zorder=2)
            ax.scatter(*ee["x0"].T, s=14, c="#d6604d", edgecolors="white", linewidths=0.4,
                       zorder=3, label=f"Distressed start x_i (class {cls[1]}, n={len(x0)})")
            ax.scatter(*ee["y1"].T, s=14, c="#762a83", edgecolors="white", linewidths=0.4,
                       zorder=3, label=f"A1-SP-F1 endpoint y_i* (λ̃ = {lam:g})")
            ax.plot([], [], color="#1b7837", lw=1.3, label=f"Real recovery 2022→2024 (n={len(rc)})")
            if ri == 1:
                ax.axvline(float(np.median(la_all[healthy])), color="#1b7837", lw=0.9, ls=":",
                           label="Median logit of healthy firms")
                ax.set_xlabel("LogReg distress logit (higher = riskier)")
                ax.set_ylabel("MDS 1 residual (orthogonal to the logit)")
            else:
                ax.set_xlabel("MDS 1")
                ax.set_ylabel("MDS 2")
            ax.set_xlim(lo_[0], hi_[0])
            ax.set_ylim(lo_[1], hi_[1])
            ax.set_title(f"({'abcd'[2 * ri + ci_]}) Class {cls[1]}: {title}", fontsize=10,
                         loc="left")
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
            ax.grid(True, color="#dddddd", linewidth=0.5)
    fig.suptitle(f"A1-SP-F1 (λ̃ = {lam:g}) on the Ψ_A1 metric; contours = distress score "
                 "(LightGBM, out-of-fold, 30-NN mean in 2-D)", fontsize=11)
    if cf is not None:
        fig.colorbar(cf, ax=axes, shrink=0.6, pad=0.01).set_label("Distress score")
    fig.savefig(OUT / "bs_manifold_mapping_a1spf1.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "bs_manifold_mapping_a1spf1.pdf", bbox_inches="tight")
    res["mds_share"] = share
    lines += ["## 5. 地図(`bs_manifold_mapping_a1spf1.png`)", "",
              f"Ψ_A1 の古典的 MDS の上位 2 成分の寄与 {share:.0%}。"
              "下段の横軸は分類器のロジットなので、"
              "左右の勾配は作り方から生じる。", "",
              "| class | ロジットの中央値: 生成の始点 → 終点 | 実際の回復の 2022 → 2024 | 健全企業 "
              "| MDS 上の矢印の長さの中央値: 生成 / 実際の回復 |", "|---|---|---|---:|---|"]
    for cls in ("c2", "c1"):
        mp = res[f"map_{cls}"]
        lines.append(f"| {cls} | {mp['logit_start_med']:+.2f} → {mp['logit_end_med']:+.2f} "
                     f"| {mp['logit_real_start_med']:+.2f} → {mp['logit_real_end_med']:+.2f} "
                     f"| {mp['logit_healthy_med']:+.2f} | {mp['mds_len_gen_med']:.1f} / "
                     f"{mp['mds_len_real_med']:.1f} |")
    lines.append("")
    # 6. BS 表(class 2 の 5 社)
    g69, ga = gen("s69", "c2"), gen("a1sp", "c2")
    x0c, x1c = [f"x0_{n}" for n in names], [f"x1_{n}" for n in names]
    na0 = g69[x0c].to_numpy() @ members["jppfs_cor_NetAssets"]
    dd_ = np.array(g69.index)
    r_ = np.random.default_rng(0)
    ins = dd_[na0 < 0]
    pick_ins = list(r_.choice(ins, size=min(2, len(ins)), replace=False))
    rest = [d for d in dd_ if d not in pick_ins]
    picks = pick_ins + list(r_.choice(rest, size=3, replace=False))
    viol = {k: pd.read_csv(Path(s_.metrics) / "viol_c2.csv", dtype={"docID": str})
            for k, s_ in src.items()}
    labels = load_labels(Path("conf/accounts/labels_ja.yaml"))
    bs = ["# ステージ80: BS 表(class 2、百万円、生成の終点は 5 年後とみなす)", "",
          f"企業はステージ65〜79 と同じ 5 社。A1・SP-F1 は単一の生成(λ̃ = {lam:g})。", ""]
    for d in picks:
        r0 = g69.loc[d]
        a0 = float(r0["assets_start"])
        cols = [("始点", r0[x0c].to_numpy(float) * a0),
                ("ステージ69", r0[x1c].to_numpy(float) * a0),
                (f"A1・SP-F1 λ̃={lam:g}", ga.loc[d][x1c].to_numpy(float) * a0)]
        tag = "(始点で債務超過)" if d in pick_ins else ""
        bs += bs_table(spec, names, labels, f"{r0['filerName']}({d}){tag}", cols)
        for k, s_ in src.items():
            v = viol[k]
            v = v[(v["setting"] == s_.setting) & (v["docID"] == d)]
            items = sorted(set(v["check"]))
            bs.append(f"- 15 項目のチェック({lab_s[k]}): "
                      + ("違反なし" if not items else "違反 " + "、".join(items)))
        bs.append("")
    (OUT / "bs_a1spf1.md").write_text("\n".join(bs) + "\n", encoding="utf-8")
    m_all = {k: {c: metrics(s_, c, names, members) for c in ("c2", "c1")} for k, s_ in src.items()}
    res["metrics"] = m_all
    lines += ["## 6. 主指標(単一の生成、標準の参照)", "",
              "class 1 の順位相関の標準の参照(T2 の 68 組)は、コストのプール 57 社と 56 社重なる"
              "(リーク)。交差適合の値は `d1/comparison_report.md`。", "",
              "| class | 設定 | 違反 | 純資産の改善 | 解消率 | 順位相関 | 払込資本↓ / ↑ "
              "| 変更科目数 |",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for cls in ("c2", "c1"):
        for k in ("s69", "a1sp"):
            m = m_all[k][cls]
            lines.append(f"| {cls} | {lab_s[k]} | {m['viol']} | {m['nai']:.3f} | {m['res']:.2f} "
                         f"| {m['rho']:+.3f} | {m['paid_dec']:.0%} / {m['paid_inc']:.0%} "
                         f"| {m['n_changed']:.0f} |")
    lines.append("")
    (OUT / "final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                    encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
