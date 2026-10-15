#!/usr/bin/env python
"""ステージ65 D.1: 発表用の図 2〜5(PNG、文字は英語。この環境に日本語フォントがないため)。

図 2: 第 1 項の重み m の飽和(ステージ62、λ̃ = 3)
図 3: 科目別の変化量(提案・等重み L2・実データ、年率の中央値)
図 4: 目標点の識別 AUC
図 5: 不均衡 OT の実効サポートと τ1(ε = 0.001)

    uv run python scripts/fig_stage65.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

OUT = Path("results/stage65/figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CLS = {"c2": "class 2 (n = 55)", "c1": "class 1 (n = 128)"}
CLS_SRC = {"c2": "class 2 (n = 55)", "c1": "class 1 (n = 132 sources)"}  # OT・目標は 132 社
EN = {"__OTHER__jppfs_cor_Assets": "other assets", "inventories": "inventories",
      "receivables_trade": "trade receivables", "jppfs_cor_CashAndDeposits": "cash and deposits",
      "jppfs_cor_OtherCA": "other current assets", "allowance_doubtful_current":
      "allowance (current)", "__OTHER__jppfs_cor_CurrentAssets": "current assets, residual",
      "jppfs_cor_PropertyPlantAndEquipment": "property, plant and equipment",
      "jppfs_cor_IntangibleAssets": "intangible assets",
      "jppfs_cor_InvestmentsAndOtherAssets": "investments and other assets",
      "allowance_doubtful_noncurrent": "allowance (non-current)",
      "__OTHER__jppfs_cor_NoncurrentAssets": "non-current assets, residual",
      "short_term_loans": "short-term loans",
      "jppfs_cor_IncomeTaxesPayable": "income taxes payable",
      "jppfs_cor_OtherCL": "other current liabilities", "payables_trade": "trade payables",
      "accounts_payable_other": "accounts payable, other", "bonds_current": "bonds, current",
      "__OTHER__jppfs_cor_CurrentLiabilities": "current liabilities, residual",
      "__OTHER__jppfs_cor_Liabilities": "liabilities, residual",
      "long_term_loans": "long-term loans", "bonds_noncurrent": "bonds",
      "__OTHER__jppfs_cor_NoncurrentLiabilities": "non-current liabilities, residual",
      "non_controlling_interests": "non-controlling interests",
      "__OTHER__jppfs_cor_NetAssets": "net assets, residual",
      "jppfs_cor_RetainedEarnings": "retained earnings", "jppfs_cor_CapitalStock": "capital stock",
      "jppfs_cor_CapitalSurplus": "capital surplus", "jppfs_cor_TreasuryStock": "treasury stock",
      "__OTHER__jppfs_cor_ShareholdersEquity": "shareholders' equity, residual",
      "valuation_difference_securities": "valuation difference, securities",
      "__OTHER__jppfs_cor_ValuationAndTranslationAdjustments": "valuation adjustments, residual"}


def style(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig2() -> None:
    g = pd.read_csv("results/stage62/grid_m.csv")
    g = g[(g["lam_tilde"] == 3.0) & g["design"].isin(["base", "paid3", "paid10", "paid30"])]
    m_of = {"base": 1, "paid3": 3, "paid10": 10, "paid30": 30}
    g = g.assign(m=g["design"].map(m_of))
    real = {"c2": 0.357, "c1": 0.228}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for j, cls in enumerate(("c2", "c1")):
        panels = (("paid_dec", "Share reducing paid-in capital"),
                  ("net_asset_improvement", "Net-asset improvement (median)"))
        for i, (col, title) in enumerate(panels):
            ax = axes[i, j]
            for d, lab, c, mk in (("l2", "equal-weight L2", BLUE, "o"),
                                  ("l1a", "asymmetric L1 (proposed)", AQUA, "^")):
                s = g[(g["cls"] == cls) & (g["dist"] == d)].sort_values("m")
                ax.plot(s["m"], s[col], color=c, marker=mk, markersize=6, linewidth=2, label=lab)
            if col == "paid_dec":
                ax.axhline(real[cls], color=INK, linestyle="--", linewidth=1.2)
                ax.text(1.0, real[cls], " real data", color=INK, fontsize=8, va="bottom")
            ax.set_xscale("log")
            ax.set_xticks([1, 3, 10, 30], labels=["1", "3", "10", "30"])
            style(ax, f"{title}, {CLS[cls]}")
            if i == 1:
                ax.set_xlabel("m for paid-in capital (first-term weight 1/m^2)", color=INK,
                              fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("First-term weight on paid-in capital saturates at m = 10 (lambda tilde = 3)",
                 y=0.995, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "fig2_m_saturation.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig3() -> None:
    star = json.loads(Path("results/stage65/summary.json").read_text(encoding="utf-8"))["star"]
    refs = pd.read_parquet("data/interim/stage65/metrics/refs.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, cls in zip(axes, ("c2", "c1"), strict=True):
        d = pd.read_parquet(f"data/interim/stage65/metrics/deltas_{cls}.parquet")
        d = d[d["space"] == "native"]
        years_ref = 2.0 if cls == "c2" else 1.0
        ref = refs[(refs["cls"] == cls) & (refs["space"] == "native")].set_index("account")[
            "median"] / years_ref
        s5 = f"m5_prop__l{star[cls]['m5_prop']:g}"
        s3 = f"m3_l2__l{star[cls]['m3_l2']:g}"
        gp = d[d["setting"] == s5].groupby("account")["delta"].median() / 5.0
        g3 = d[d["setting"] == s3].groupby("account")["delta"].median() / 5.0
        top = ref.abs().sort_values(ascending=False).index[:10]
        y = np.arange(len(top))
        h = 0.26
        for k, (vals, lab, c) in enumerate(((ref, "real recoveries", INK),
                                            (g3, "equal-weight L2", BLUE),
                                            (gp, "proposed", AQUA))):
            ax.barh(y + (k - 1) * h, vals.reindex(top).to_numpy(), height=h * 0.9, color=c,
                    label=lab)
        names = [EN.get(a, a) for a in top]
        ax.set_yticks(y, labels=names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED, linewidth=0.8)
        style(ax, f"Median annual change / total assets, {CLS[cls]}")
        ax.set_xlabel("change per year (share of starting total assets)", color=INK, fontsize=9)
    h_, lab_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, lab_, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Account-level changes: 10 accounts with the largest real median change",
                 y=0.995, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "fig3_account_changes.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig4() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    rand, avg2 = 0.471, 0.918  # ステージ61 b2.md(class 0 の無作為抽出 55 社、無作為な 2 社の平均)
    for ax, cls in zip(axes, ("c2", "c1"), strict=True):
        a = pd.read_csv(f"results/stage65/auc_{cls}.csv").set_index("method")["auc_lgbm"]
        items = [("random real firms", rand, "real"),
                 ("proposed: unbalanced OT argmax", a["m5_prop"], "real"),
                 ("nearest neighbour, equal L2", a["m3_l2"], "real"),
                 ("balanced OT argmax", a["m6_balarg"], "real"),
                 ("average of 2 random firms", avg2, "synthetic"),
                 ("balanced OT barycentre", a["m7_bary"], "synthetic")]
        y = np.arange(len(items))
        cols = [BLUE if t == "real" else ORANGE for _, _, t in items]
        ax.barh(y, [v for _, v, _ in items], color=cols, height=0.6)
        for yi, (_, v, _) in zip(y, items, strict=True):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, labels=[n for n, _, _ in items], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.5, color=INK, linestyle="--", linewidth=1.0)
        ax.set_xlim(0.3, 1.05)
        style(ax, f"Identification AUC of targets vs real class 0, {CLS_SRC[cls]}")
        ax.set_xlabel("AUC (LightGBM, 5-fold CV; 0.5 = indistinguishable)", color=INK,
                      fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="target is a real firm"),
                        Patch(color=ORANGE, label="target is a synthetic average")],
               loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Averaged targets are separable from real firms", y=0.995, fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig4_auc.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig5() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    bound = {"c2": 3768 / 55, "c1": 3768 / 132}
    for cls, c, mk in (("c2", BLUE, "o"), ("c1", ORANGE, "s")):
        p = pd.read_csv(f"results/stage64/plans_{cls}.csv")
        s = p[(p["method"] == "不均衡") & (p["eps"] == 0.001)].sort_values("tau1")
        n0 = int(p.loc[p["setting"] == "nn", "nn_match"].iloc[0])
        axes[0].plot(s["tau1"], s["support_med"], color=c, marker=mk, linewidth=2,
                     label=CLS_SRC[cls])
        axes[0].axhline(bound[cls], color=c, linestyle="--", linewidth=1.0)
        axes[0].text(0.0011, bound[cls], f" balanced lower bound n1/n0 = {bound[cls]:.1f}",
                     color=INK, fontsize=8, va="bottom")
        axes[1].plot(s["tau1"], 100 * s["nn_match"] / n0, color=c, marker=mk, linewidth=2,
                     label=CLS_SRC[cls])
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(0.01, color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_xlabel("tau1 (KL weight on target marginal, epsilon = 0.001)", color=INK,
                      fontsize=9)
    axes[0].set_yscale("log")
    style(axes[0], "Effective support size (median, 1 / sum p^2)")
    style(axes[1], "Share of maximum-mass targets equal to nearest neighbour (%)")
    axes[1].text(0.0105, 0.04, " chosen tau1 = 0.01", color=INK, fontsize=8,
                 transform=axes[1].get_xaxis_transform())
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Unbalanced OT recovers locality as tau1 falls", y=0.995, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig5_support_tau1.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig2()
    fig3()
    fig4()
    fig5()


if __name__ == "__main__":
    main()
