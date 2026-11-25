#!/usr/bin/env python
"""ステージ83 パートD: 図 1・図 2(描画だけ。lightgbm・sklearn のプロセス)。

保存済みの生成(``data/interim/stage83/full``)、Tier C の一覧(``results/stage83/exp5_tierC.csv``)、
バックテストの配列(``data/interim/stage83/backtest_arrays.npz``)を読み、モデルは再計算しない。地図の座標
(古典的 MDS)と背景の苦境スコア(LightGBM の out-of-fold)は、ステージ83 と同じ乱数で決定的に作り直す。

- 図 1 の左列 (a)(c) は、主な点が読める標準の範囲に固定し、枠外の点の数を注記する。Tier C(始点が
  極端な 5 社)の全体は別図 ``bs_tier_c_overview_kde0`` に示す(インセットは目盛りや注記と重なり、
  読みにくかったため)。右列 (b)(d) は従来どおり点の 1〜99% 分位点で範囲を決める。
- 記号は mathtext で表記する。

出力: ``results/stage83/figures/bs_manifold_mapping_kde0_stage1.{png,pdf}``・
``bs_tier_c_overview_kde0.{png,pdf}``・``bs_backtest_overlay_kde0.{png,pdf}``(300 dpi)。

    uv run python scripts/stage83_figures.py
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
    N_FOLD,
    N_LAND,
    SEED_CLF,
    make_clf,
    mds_embed,
    mds_fit,
    panel_years,
)
from stage82_common import Setting  # noqa: E402
from stage83_common import DATA, OUT  # noqa: E402
from tables_stage69 import setup_excl  # noqa: E402
from tilde_stage51 import load_tilde  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.evaluation.classifiers import strict_sklearn_convergence  # noqa: E402

_RE = "jppfs_cor_RetainedEarnings"
K_SMOOTH = 30
# 図 1 の左列(class 2)の標準の表示範囲(Tier C の 5 社はインセットで示す)
MAIN_RANGE = {0: ((-180.0, 30.0), (-160.0, 150.0)), 1: ((-6.5, 12.0), (-95.0, 45.0))}
PSI = r"$\psi_{\mathrm{KDE0}}$"
GEN_LABEL = r"KDE0 + stage-1 penalty endpoint $\hat{\mathbf{y}}_i$"


def draw_figures(lam: float, bs: pd.DataFrame) -> tuple[Path, Path, float]:  # noqa: PLR0915
    """図 1・2 を描き、(図 1 のパス, 図 2 のパス, MDS の 2 次元の寄与) を返す。"""
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    idx = {n: i for i, n in enumerate(names)}
    setup_excl(names)
    x, lab, docs, _ = load_tilde(names)
    healthy = lab == 0
    xh = x[healthy]
    tj = json.loads((DATA / "theta.json").read_text(encoding="utf-8"))
    th = np.array(tj["configs"]["full"]["KDE0"]["theta"], dtype=float)
    th[:, idx[_RE]] = 0.0
    wsym = 0.5 * (th[0] + th[1])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    codes = np.array([code_of.get(d, d) for d in docs])
    clf_axis = make_clf("logreg")
    with strict_sklearn_convergence():
        clf_axis.fit(x, (lab >= 1).astype(int))
    uc = np.random.default_rng(SEED_CLF).permutation(np.unique(codes))
    fold_of = {c: i for i, part in enumerate(np.array_split(uc, N_FOLD)) for c in part}
    fold = np.array([fold_of[c] for c in codes])
    score = np.empty(len(lab))
    for f in range(N_FOLD):
        m_ = make_clf("lightgbm")
        m_.fit(x[fold != f], (lab[fold != f] >= 1).astype(int))
        score[fold == f] = m_.predict_proba(x[fold == f])[:, 1]
    land = xh[np.random.default_rng(0).choice(len(xh), size=N_LAND, replace=False)]
    proj, mean_d2, share = mds_fit(land, wsym)
    e_all = mds_embed(x, land, wsym, proj, mean_d2)
    la_all = clf_axis.decision_function(x)
    beta = np.polyfit(la_all, e_all[:, 0], 1)
    p_all = np.column_stack([la_all, e_all[:, 0] - np.polyval(beta, la_all)])
    logit_h = float(np.median(la_all[healthy]))

    def mds(pts: np.ndarray) -> np.ndarray:
        out: np.ndarray = mds_embed(pts, land, wsym, proj, mean_d2)
        return out

    def plane(pts: np.ndarray) -> np.ndarray:
        lx = clf_axis.decision_function(pts)
        return np.column_stack([lx, mds(pts)[:, 0] - np.polyval(beta, lx)])

    def auto_range(pts: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        lo_, hi_ = np.quantile(pts, 0.01, axis=0), np.quantile(pts, 0.99, axis=0)
        pad = 0.08 * (hi_ - lo_)
        return ((float(lo_[0] - pad[0]), float(hi_[0] + pad[0])),
                (float(lo_[1] - pad[1]), float(hi_[1] + pad[1])))

    def background(ax: Any, ea: np.ndarray, rng_: tuple[tuple[float, float], tuple[float, float]]
                   ) -> Any:
        """表示範囲そのもので等高線を計算する(範囲と等高線がずれて白い切れ目が出ないように)。"""
        (x_lo, x_hi), (y_lo, y_hi) = rng_
        gx, gy = np.meshgrid(np.linspace(x_lo, x_hi, 140), np.linspace(y_lo, y_hi, 140))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        sc = np.array([x_hi - x_lo, y_hi - y_lo])
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
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        return cf

    def arrows(ax: Any, a: np.ndarray, b: np.ndarray, color: str, width: float,
               alpha: float) -> None:
        ax.quiver(*a.T, *(b - a).T, angles="xy", scale_units="xy", scale=1.0, color=color,
                  width=width, alpha=alpha, zorder=2)

    comp, _ = panel_years(names)
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    tierc = pd.read_csv(OUT / "exp5_tierC.csv", dtype={"docID": str})
    tag = Setting("full", "KDE0p", "prop", "base", "t123").tag
    e51._GEN = DATA / "full" / "gen_full"
    fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
    cf = None
    overview: dict[int, dict[str, Any]] = {}
    for ci_, cls in enumerate(("c2", "c1")):
        g = e51.load_gen(cls, f"{tag}__l{lam:g}")
        g = g[g["seed"] == 0]
        x0 = g[[f"x0_{n}" for n in names]].to_numpy(float)
        ya = g[[f"x1_{n}" for n in names]].to_numpy(float)
        dd_ = [str(d) for d in g["docID"]]
        grp = "class 2 → 回復" if cls == "c2" else "class 1 → class 0"
        rc_ = [c for c in pairs[pairs["group"] == grp]["edinetCode"]
               if c in comp[2022].index and c in comp[2024].index]
        xa, xb = comp[2022].loc[rc_].to_numpy(), comp[2024].loc[rc_].to_numpy()
        tc = np.array([d in set(tierc[tierc["cls"] == cls]["docID"]) for d in dd_])
        for ri in (0, 1):
            ax = axes[ri, ci_]
            emb = mds if ri == 0 else plane
            ea = e_all if ri == 0 else p_all
            s0, s1, qa, qb = (emb(v) for v in (x0, ya, xa, xb))
            sm = emb(0.5 * x0[tc] + 0.5 * ya[tc]) if tc.any() else np.empty((0, 2))
            rng_ = (MAIN_RANGE[ri] if cls == "c2"
                    else auto_range(np.vstack([s0, s1, qa, qb, ea[healthy]])))
            cf = background(ax, ea, rng_)
            arrows(ax, qa, qb, "#1b7837", 0.0035, 0.9)
            arrows(ax, s0, s1, "#762a83", 0.0022, 0.6)
            ax.scatter(*s0.T, s=14, c="#d6604d", edgecolors="white", linewidths=0.4, zorder=3,
                       label=rf"Distressed start $\mathbf{{x}}_i$ (class {cls[1]}, n={len(x0)})")
            ax.scatter(*s1.T, s=14, c="#762a83", edgecolors="white", linewidths=0.4, zorder=3,
                       label=rf"{GEN_LABEL} ($\tilde{{\lambda}}$ = {lam:g})")
            if len(sm):
                ax.scatter(*sm.T, s=110, marker="*", c="#f1a340", edgecolors="black",
                           linewidths=0.6, zorder=4,
                           label=(rf"Tier C midpoint $\mathbf{{y}}_i^{{(0.5)}}$ (n={len(sm)}"
                                  "; beyond frame, see overview)"))
            ax.plot([], [], color="#1b7837", lw=1.3,
                    label=f"Real recovery 2022→2024 (n={len(rc_)})")
            if ri == 1:
                ax.axvline(logit_h, color="#1b7837", lw=0.9, ls=":",
                           label="Median logit of healthy firms")
                ax.set_xlabel("LogReg distress logit (higher = riskier)")
                ax.set_ylabel("MDS 1 residual (orthogonal to the logit)")
                title = f"Distress logit (LogReg) × MDS 1 residual ({PSI})"
            else:
                ax.set_xlabel("MDS 1")
                ax.set_ylabel("MDS 2")
                title = f"Classical MDS of {PSI} (2-D share {share:.0%})"
            ax.set_title(f"({'abcd'[2 * ri + ci_]}) Class {cls[1]}: {title}", fontsize=10,
                         loc="left")
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
            if cls != "c2":
                continue
            # 枠外の点の数(Tier C の全体は別図 bs_tier_c_overview_kde0)
            (x_lo, x_hi), (y_lo, y_hi) = rng_
            pts = np.vstack([s0, s1])
            n_out = int((~((pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi) & (pts[:, 1] >= y_lo)
                          & (pts[:, 1] <= y_hi))).sum())
            ax.text(0.99, 0.99, f"{n_out} start/end points beyond frame\n"
                    "(Tier C full range: separate overview figure)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444444",
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bbbbbb",
                          "alpha": 0.9})
            overview[ri] = {"s0": s0, "s1": s1, "sm": sm, "tc": tc, "rng": rng_, "ea": ea}
    fig.suptitle("KDE0 + stage-1 penalty (main model); contours = distress score "
                 "(LightGBM, out-of-fold, 30-NN mean in 2-D)", fontsize=11)
    if cf is not None:
        fig.colorbar(cf, ax=axes, shrink=0.6, pad=0.01).set_label("Distress score")
    f1 = OUT / "figures" / "bs_manifold_mapping_kde0_stage1"
    f1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{f1}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{f1}.pdf", bbox_inches="tight")
    plt.close(fig)
    # 別図: class 2 の Tier C の全体(範囲は class 2 の点と Tier C から決め、健全企業の外れ値には
    # 引っ張られないようにする)
    fig, axo = plt.subplots(1, 2, figsize=(14, 6.2))
    for ri, ax in enumerate(axo):
        o = overview[ri]
        s0, s1, sm, tc = o["s0"], o["s1"], o["sm"], o["tc"]
        (x_lo, x_hi), (y_lo, y_hi) = o["rng"]
        allp = np.vstack([s0, s1, sm, [[x_lo, y_lo], [x_hi, y_hi]]])
        lo_, hi_ = allp.min(axis=0), allp.max(axis=0)
        pad = 0.06 * (hi_ - lo_)
        ax.scatter(*o["ea"][healthy].T, s=2, c="#4a6b8a", alpha=0.2, linewidths=0,
                   label="Healthy firms (class 0)")
        arrows(ax, s0[~tc], s1[~tc], "#762a83", 0.0015, 0.35)
        ax.scatter(*s0[~tc].T, s=8, c="#d6604d", alpha=0.6, linewidths=0,
                   label=r"Distressed start $\mathbf{x}_i$ (Tier A / B)")
        arrows(ax, s0[tc], s1[tc], "#762a83", 0.004, 0.9)
        ax.scatter(*s0[tc].T, s=40, c="#d6604d", edgecolors="black", linewidths=0.5, zorder=3,
                   label=r"Tier C start $\mathbf{x}_i$")
        ax.scatter(*s1[tc].T, s=40, c="#762a83", edgecolors="black", linewidths=0.5, zorder=3,
                   label=r"Tier C endpoint $\hat{\mathbf{y}}_i$")
        ax.scatter(*sm.T, s=150, marker="*", c="#f1a340", edgecolors="black", linewidths=0.6,
                   zorder=4, label=r"Tier C midpoint $\mathbf{y}_i^{(0.5)}$")
        ax.add_patch(mpl.patches.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False,
                                           ec="#333333", lw=1.0, ls="--",
                                           label="Frame of Fig. 1 (a)/(c)"))
        ax.set_xlim(lo_[0] - pad[0], hi_[0] + pad[0])
        ax.set_ylim(lo_[1] - pad[1], hi_[1] + pad[1])
        ax.grid(True, color="#dddddd", linewidth=0.5)
        if ri == 0:
            ax.set_xlabel("MDS 1")
            ax.set_ylabel("MDS 2")
            ax.set_title(f"(a) Class 2: Classical MDS of {PSI}, full range", fontsize=10,
                         loc="left")
        else:
            ax.axvline(logit_h, color="#1b7837", lw=0.9, ls=":",
                       label="Median logit of healthy firms")
            ax.set_xlabel("LogReg distress logit (higher = riskier)")
            ax.set_ylabel("MDS 1 residual (orthogonal to the logit)")
            ax.set_title(f"(b) Class 2: Distress logit × MDS 1 residual ({PSI}), full range",
                         fontsize=10, loc="left")
    # 凡例は図の下に 1 つだけ(パネルの点を隠さないように)
    handles: dict[str, Any] = {}
    for ax in axo:
        for h_, l_ in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(l_, h_)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Tier C firms (n={int(overview[0]['tc'].sum())}): start, midpoint and "
                 "endpoint on the full range (KDE0 + stage-1 penalty)", fontsize=11)
    fo = OUT / "figures" / "bs_tier_c_overview_kde0"
    fig.savefig(f"{fo}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fo}.pdf", bbox_inches="tight")
    plt.close(fig)
    # 図 2(レイアウトは従来どおり、記号だけ mathtext に)
    bt = np.load(DATA / "backtest_arrays.npz", allow_pickle=True)
    fig, axes2 = plt.subplots(1, 2, figsize=(15, 7.2))
    cf = None
    for ci_, cls in enumerate(("c2", "c1")):
        sel = bt["cls"] == cls
        a0, a1, ay = (plane(bt[k][sel]) for k in ("x0", "x1", "y"))
        ax = axes2[ci_]
        cf = background(ax, p_all, auto_range(np.vstack([a0, a1, ay, p_all[healthy]])))
        arrows(ax, a0, a1, "#1b7837", 0.0035, 0.9)
        arrows(ax, a0, ay, "#762a83", 0.0030, 0.75)
        ax.scatter(*a0.T, s=16, c="#d6604d", edgecolors="white", linewidths=0.4, zorder=3,
                   label=rf"Pre-recovery $\mathbf{{x}}_{{k,0}}$ (n={int(sel.sum())})")
        ax.plot([], [], color="#1b7837", lw=1.3,
                label=r"Real recovery $\mathbf{x}_{k,0} \to \mathbf{x}_{k,1}$")
        ax.plot([], [], color="#762a83", lw=1.3,
                label=r"Out-of-fold generation $\mathbf{x}_{k,0} \to \hat{\mathbf{y}}_k$")
        ax.axvline(logit_h, color="#1b7837", lw=0.9, ls=":", label="Median logit of healthy firms")
        r = bs[(bs["cls"] == cls) & (bs["model"] == "KDE0p|main")].iloc[0]
        ax.set_title(rf"({'ab'[ci_]}) Class {cls[1]} recovered firms: median DSA "
                     rf"{r['dsa_0.01_med']:.3f}, Cos {r['cos_med']:.3f}, "
                     rf"$\mathrm{{ERR}}^{{L_1}}$ {r['err_l1_med']:.3f}", fontsize=10, loc="left")
        ax.set_xlabel("LogReg distress logit (higher = riskier)")
        ax.set_ylabel("MDS 1 residual (orthogonal to the logit)")
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Backtest on real recovered firms: real vs out-of-fold generated moves "
                 "(KDE0 + stage-1 penalty)", fontsize=11)
    if cf is not None:
        fig.colorbar(cf, ax=axes2, shrink=0.8, pad=0.01).set_label("Distress score")
    f2 = OUT / "figures" / "bs_backtest_overlay_kde0"
    fig.savefig(f"{f2}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{f2}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f1, f2, share


def main() -> None:
    gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
    bs = pd.read_csv(OUT / "backtest_summary.csv")
    f1, f2, share = draw_figures(float(gate["lambda_tilde"]), bs)
    print(f"{f1}、{f2}(MDS の 2 次元の寄与 {share:.0%})")


if __name__ == "__main__":
    main()
