#!/usr/bin/env python
"""ステージ78: 論文の表 1・4・5 と図 1 の検算を、提案の R1 の尺度でも計算する(lightgbm・sklearn の
プロセス)。

コード監査の指摘(項目 1-3・2)への対応。ステージ76・77 の出力は変えず、``results/stage78/`` に
並べて出す。

1. 表 1 の下の行: QP の終点(ステージ69 の設定・R1・F1 の案 a)のゼロの科目数(「ちょうど 0」と
   |y| < 10⁻⁶)と、**終点そのもの**の識別 AUC(class 0 と、LightGBM・LogReg、3 回)。比較に、目標
   (候補・対応先)の AUC と、上の行(健全企業・2 社の平均 10,000 組・重心)の AUC も出す。
2. 表 4: 移動の尺度を Ψ60(ステージ60 の c±、class ごと)と Ψ_R1(57 社の全体の R1、利益剰余金 0)
   の両方にして、全上場企業の 1・3 年と再建群 72 社の推移の 95%・99% 点以内の割合と、健全企業から
   r90 以内に 1 社もない終点の割合を出す。r90 はそれぞれの尺度で、健全企業どうしの最近傍の 90% 点。
3. 表 5: 実データの推移(ステージ38 のラベル)を Ψ60_c2・Ψ60_c1・Ψ_R1 の四分位に分けた回復の割合を、
   始点が class 1・2 の合算、class 2 だけ、class 1 だけの 3 通りで出す。
4. 図 1: Ψ_R1 を対称化した距離の古典的 MDS と、LogReg のロジット × MDS 1 の残差の平面に、R1・F1 の
   移動を描く(``bs_manifold_mapping_r1metric.png``・``.pdf``)。

    uv run python scripts/stage78_checks.py
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
from stage76_feasibility_map import (  # noqa: E402
    N_LAND,
    REAL_GROUP,
    SEED_CLF,
    make_clf,
    mds_embed,
    mds_fit,
    moves,
    panel_years,
    psi,
)
from stage77_final_feasibility import SRC  # noqa: E402
from tables_stage69 import INTERIM, setup_excl  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.classifiers import strict_sklearn_convergence  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402

OUT = Path("results/stage78")
_RE = "jppfs_cor_RetainedEarnings"
ZERO_TOL = 1e-6
K_SMOOTH = 30
LAB = {"s69": "ステージ69 の設定(λ̃ = 1)", "r1f1": "R1・F1(案 a、λ̃ = 0.1)"}


def zeros(v: np.ndarray) -> tuple[float, float]:
    return float((v == 0.0).sum(axis=1).mean()), float((np.abs(v) < ZERO_TOL).sum(axis=1).mean())


def auc(v: np.ndarray, xt: np.ndarray) -> tuple[float, float]:
    r = identification_auc_repeated(v, xt, 3)
    return float(r["lightgbm"].mean()), float(r["logreg"].mean())


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    setup_excl(names)
    x, lab, docs, _ = load_tilde(names)
    xt = x[lab == 0]
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes = np.array([code_of.get(d, d) for d in docs])
    healthy = lab == 0
    xh, ch = x[healthy], codes[healthy]
    wj = json.loads((INTERIM / "stage75/weights.json").read_text(encoding="utf-8"))
    cr1 = (np.array(wj["configs"]["full"]["R1"]["plus"], float),
           np.array(wj["configs"]["full"]["R1"]["minus"], float))
    for c_ in cr1:
        c_[names.index(_RE)] = 0.0
    c60 = {cls: tuple(np.array(json.loads(Path(f"data/interim/stage60/meta_{cls}.json").read_text(
        encoding="utf-8"))[k], float) for k in ("c_plus", "c_minus")) for cls in ("c2", "c1")}
    res: dict[str, Any] = {}

    def gen(key: str, cls: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e51._GEN = SRC[key].gen
        g = e51.load_gen(cls, SRC[key].setting)
        g = g[g["seed"] == 0].set_index("docID")
        e51._GEN = SRC["s69"].gen
        g0 = e51.load_gen(cls, SRC["s69"].setting)
        g = g.loc[g0[g0["seed"] == 0]["docID"]]
        gc = np.array([code_of.get(str(d), str(d)) for d in g.index])
        return (g[[f"x0_{n}" for n in names]].to_numpy(float),
                g[[f"x1_{n}" for n in names]].to_numpy(float), gc)

    # 1. 表 1: ゼロと識別 AUC
    t1 = ["## 1. 表 1(命題 2)の検算: ゼロの科目数と識別 AUC", "",
          "ゼロ = 32 科目中の「ちょうど 0」と |y| < 10⁻⁶ の数(平均)。"
          "識別 AUC は class 0 の実在企業と"
          "の 5 分割 CV(class 0 の抽出を 3 回変えた平均、`identification_auc_repeated`)。", "",
          "| 対象 | 件数 | ちょうど 0 | \\|y\\| < 10⁻⁶ | AUC LightGBM | AUC LogReg |",
          "|---|---:|---:|---:|---:|---:|"]
    rng = np.random.default_rng(0)
    i1 = rng.integers(0, len(xt), 10_000)
    i2 = (i1 + rng.integers(1, len(xt), 10_000)) % len(xt)
    rows: list[tuple[str, np.ndarray, bool]] = [("健全企業(class 0、実在)", xt, False),
                                                ("2 社の平均(異なる 2 社、10,000 組)",
                                                 0.5 * (xt[i1] + xt[i2]), True)]
    for cls in ("c2", "c1"):
        z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        rows.append((f"バランス OT の重心({cls})", np.asarray(z["bal_e0.001__bary"], float), True))
    for cls in ("c2", "c1"):
        pz = np.load(INTERIM / f"stage75/plan_{cls}_full.npz", allow_pickle=True)
        idx, pt = np.asarray(pz["R1__idx"]), np.asarray(pz["R1__pt"])
        rows.append((f"R1・F1 の候補 S(i)({cls}、QP の前)", xt[idx[pt > 0]], True))
        z = np.load(f"data/interim/stage64/targets_{cls}.npz", allow_pickle=True)
        rows.append((f"ステージ69 の対応先({cls}、argmax、QP の前)",
                     xt[np.asarray(z["semi_t0.01_e0.001__argmax"]).astype(int)], True))
        for key in ("s69", "r1f1"):
            rows.append((f"終点 y*: {LAB[key]}({cls})", gen(key, cls)[1], True))
            rows.append((f"終点 y*: {LAB[key]}({cls})、両側で \\|v\\| < 10⁻⁶ を 0 に丸めた AUC",
                         gen(key, cls)[1], True))
    for lab_t, v, do_auc in rows:
        z0, zt = zeros(v)
        if "丸めた" in lab_t:
            # 数値誤差(10⁻¹² 程度)で木が分けていないかの確認: 生成・実在の両方を同じ規則で丸める
            def snap(u: np.ndarray) -> np.ndarray:
                return np.where(np.abs(u) < ZERO_TOL, 0.0, u)
            a = auc(snap(v), snap(xt))
        else:
            a = auc(v, xt) if do_auc else (float("nan"), float("nan"))
        res[f"t1|{lab_t}"] = {"n": len(v), "zero_exact": z0, "zero_tol": zt, "auc_lgbm": a[0],
                              "auc_logreg": a[1]}
        cell = f"{a[0]:.3f} | {a[1]:.3f}" if do_auc else "— | —"
        t1.append(f"| {lab_t} | {len(v)} | {z0:.2f} | {zt:.2f} | {cell} |")
        print(f"  {lab_t}: {z0:.2f} / {zt:.2f} / {a}", flush=True)
    t1.append("")
    # 2. 表 4: 実行可能性と空白地帯率(Ψ60・Ψ_R1)
    comp, _ = panel_years(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    q_i = np.random.default_rng(1).choice(len(xh), size=2000, replace=False)
    t4 = ["## 2. 表 4 の検算: 移動の尺度 Ψ60 と Ψ_R1", "",
          "パーセンタイル・95%・99% 点以内は、生成の Ψ(x, y*) を実在の推移の Ψ の分布と比べたもの。"
          "空白地帯率 = 健全企業(自社を除く)が Ψ(z − y) ≤ r90 に 1 社もない割合。r90 は同じ尺度で、"
          "健全企業 2,000 社の最近傍の Ψ の 90% 点。健全企業自身の空白地帯率は 300 社の標本。", "",
          "| class | 尺度 | 設定 | 生成の Ψ 中央値 | 1 年: 95% / 99% 点以内 "
          "| 3 年: 95% / 99% 点以内 "
          "| 再建群 72 社(2 年): 95% / 99% 点以内 | 空白地帯率(始点 → 終点) | 健全企業自身 |",
          "|---|---|---|---:|---|---|---|---|---:|"]
    for cls in ("c2", "c1"):
        for mk, (cp, cm) in (("Ψ60", c60[cls]), ("Ψ_R1", cr1)):
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

            def empty(pts: np.ndarray, own: np.ndarray, cp: np.ndarray = cp, cm: np.ndarray = cm,
                      r90: float = r90) -> float:
                dd = psi(xh[None, :, :] - pts[:, None, :], cp, cm)
                dd[ch[None, :] == own[:, None]] = np.inf
                return float((dd.min(axis=1) > r90).mean())
            hz = empty(xh[q_i[:300]], ch[q_i[:300]])
            for key in ("s69", "r1f1"):
                x0, y1, gc = gen(key, cls)
                cg = psi(y1 - x0, cp, cm)
                cells = []
                out_k: dict[str, Any] = {"r90": r90, "gen_q50": float(np.median(cg))}
                for rk in ("h1", "h3", "rec72"):
                    q95, q99 = np.quantile(ref[rk], [0.95, 0.99])
                    i95, i99 = float((cg <= q95).mean()), float((cg <= q99).mean())
                    out_k[rk] = {"q95": float(q95), "q99": float(q99), "in95": i95, "in99": i99}
                    cells.append(f"{i95:.0%} / {i99:.0%}")
                e0, e1 = empty(x0, gc), empty(y1, gc)
                out_k |= {"empty_start": e0, "empty_end": e1, "empty_healthy": hz}
                res[f"t4|{cls}|{mk}|{key}"] = out_k
                t4.append(f"| {cls} | {mk} | {LAB[key]} | {out_k['gen_q50']:.3g} | "
                          + " | ".join(cells) + f" | {e0:.0%} → {e1:.0%} | {hz:.0%} |")
            print(f"  表 4 {cls} {mk}: r90 = {r90:.3g}", flush=True)
    t4.append("")
    # 3. 表 5: Ψ の四分位と回復の割合
    r38 = pd.read_parquet("data/interim/stage38/real_deltas.parquet")[
        ["edinetCode", "t", "label_prev", "label_cur"]].astype({"edinetCode": str})
    lt_ = {t: r38[r38["t"] == t].set_index("edinetCode") for t in (2023, 2024)}
    lab_y = {2022: lt_[2023]["label_prev"], 2023: lt_[2023]["label_cur"],
             2024: lt_[2024]["label_cur"]}
    t5 = ["## 3. 表 5 の検算: 実データの移動コストの四分位と回復の割合", "",
          "始点のラベルで class 1・2(合算)、class 2 だけ、class 1 だけに分けた。"
          "回復 = 終点で class 0"
          "(ラベルはステージ38 の遷移)。1 年 = 2022 → 2023 と 2023 → 2024 の合算、2 年 = 2022 → "
          "2024。四分位はそれぞれの母集団の中で切った。", "",
          "| 尺度 | 始点 | 期間 | 件数 | Q1 | Q2 | Q3 | Q4 | 単調減少 |",
          "|---|---|---|---:|---:|---:|---:|---:|---|"]
    scales = (("Ψ60_c2", c60["c2"]), ("Ψ60_c1", c60["c1"]), ("Ψ_R1", cr1))
    for sk, (cp, cm) in scales:
        frames = []
        for ya, yb in ((2022, 2023), (2023, 2024), (2022, 2024)):
            d, cc = moves(comp, ya, yb)
            la, lb = lab_y[ya].reindex(cc).to_numpy(), lab_y[yb].reindex(cc).to_numpy()
            ok = (la >= 1) & np.isfinite(lb)
            frames.append(pd.DataFrame({"h": yb - ya, "start": la[ok], "cost": psi(d[ok], cp, cm),
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
                res[f"t5|{sk}|{sl}|{h}"] = {"n": len(s), "rates": rate.tolist(), "n_q": nq.tolist(),
                                            "monotone": mono}
                t5.append(f"| {sk} | {sl} | {h} 年 | {len(s)}({'/'.join(map(str, nq))}) | "
                          + " | ".join(f"{v:.0%}" for v in rate)
                          + f" | {'はい' if mono else 'いいえ'} |")
    t5.append("")
    # 4. 図 1: Ψ_R1 の地図
    axis_clf = make_clf("logreg")
    with strict_sklearn_convergence():
        axis_clf.fit(x, (lab >= 1).astype(int))
    # 等高線用の苦境スコア: ステージ76・77 と同じ企業単位の 5 分割の out-of-fold
    uc = np.random.default_rng(SEED_CLF).permutation(np.unique(codes))
    fold_of = {c: i for i, part in enumerate(np.array_split(uc, 5)) for c in part}
    fold = np.array([fold_of[c] for c in codes])
    score = np.empty(len(lab))
    for f in range(5):
        oof_like = make_clf("lightgbm")
        oof_like.fit(x[fold != f], (lab[fold != f] >= 1).astype(int))
        score[fold == f] = oof_like.predict_proba(x[fold == f])[:, 1]
    wsym = 0.5 * (cr1[0] + cr1[1])
    land = xh[np.random.default_rng(0).choice(len(xh), size=N_LAND, replace=False)]
    proj, mean_d2, share = mds_fit(land, wsym)
    res["mds_share_r1"] = share
    top = np.argsort(wsym)[::-1][:5]
    res["r1_top_weights"] = {names[i]: float(wsym[i]) for i in top}
    e_all = mds_embed(x, land, wsym, proj, mean_d2)
    la_all = axis_clf.decision_function(x)
    beta = np.polyfit(la_all, e_all[:, 0], 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
    cf = None
    for ci_, cls in enumerate(("c2", "c1")):
        x0, y1, _ = gen("r1f1", cls)
        rc = [c for c in pairs[pairs["group"] == REAL_GROUP[cls]]["edinetCode"]
              if c in comp[2022].index and c in comp[2024].index]
        xa, xb = comp[2022].loc[rc].to_numpy(), comp[2024].loc[rc].to_numpy()
        emb = {k: mds_embed(v, land, wsym, proj, mean_d2)
               for k, v in (("x0", x0), ("y1", y1), ("xa", xa), ("xb", xb))}

        def plane(pts: np.ndarray, em: np.ndarray, beta: np.ndarray = beta) -> np.ndarray:
            lx = axis_clf.decision_function(pts)
            return np.column_stack([lx, em[:, 0] - np.polyval(beta, lx)])
        p_all = np.column_stack([la_all, e_all[:, 0] - np.polyval(beta, la_all)])
        pl = {k: plane(v, emb[k]) for k, v in (("x0", x0), ("y1", y1), ("xa", xa), ("xb", xb))}
        for row, (ea, ee, title) in enumerate((
                (e_all, emb, f"Classical MDS of Ψ_R1 (2-D share {share:.0%})"),
                (p_all, pl, "Distress logit (LogReg) × MDS 1 residual (Ψ_R1)"))):
            ax = axes[row, ci_]
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
                       zorder=3, label="R1-F1 endpoint y_i* (λ̃ = 0.1)")
            ax.plot([], [], color="#1b7837", lw=1.3, label=f"Real recovery 2022→2024 (n={len(rc)})")
            if row == 1:
                ax.axvline(float(np.median(la_all[healthy])), color="#1b7837", lw=0.9, ls=":",
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
    fig.suptitle("R1-F1 (λ̃ = 0.1) on the Ψ_R1 metric; contours = distress score "
                 "(LightGBM, out-of-fold, 30-NN mean in 2-D)", fontsize=11)
    if cf is not None:
        fig.colorbar(cf, ax=axes, shrink=0.6, pad=0.01).set_label("Distress score")
    fig.savefig(OUT / "bs_manifold_mapping_r1metric.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "bs_manifold_mapping_r1metric.pdf", bbox_inches="tight")
    lines = ["# ステージ78: 表 1・4・5 と図 1 の検算(R1 の尺度を追加)", "", *t1, *t4, *t5,
             "## 4. 図 1(Ψ_R1 の古典的 MDS)", "",
             f"`bs_manifold_mapping_r1metric.png`。上位 2 成分の寄与 {share:.0%}。対称化した重み "
             "(c⁺ + c⁻)/2 の大きい科目: "
             + "、".join(f"{k} {v:.0f}" for k, v in res["r1_top_weights"].items()) + "。", ""]
    (OUT / "checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "checks.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
