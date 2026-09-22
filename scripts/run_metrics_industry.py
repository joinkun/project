#!/usr/bin/env python
"""ステージ12: 業種考慮の輸送設計の指標計算・図・BS表(lightgbm/sklearn/matplotlibのみ、torchなし)。

run_generation_industry.py(torch専用)が書いたparquetと、
precompute_industry_transport.py(POT専用)が書いた対応先CSVを読む。
状態ベクトルは総資産で正規化済み(tilde空間)で保存されているので、ここで
Assetsによる再正規化はしない(ステージ10.6の二重正規化バグの再発防止)。

出力(data/interim/industry_12/): metrics.csv, design_summary.csv, identification_auc.csv,
figures/industry_tradeoff.png, samples/*.md
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.identification import identification_auc_repeated  # noqa: E402
from fsgen.models.cost import cost_vectors, load_cost_config  # noqa: E402
from fsgen.viz.balance_sheet import BSLine, build_bs_lines, load_labels  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", *plt.rcParams["font.sans-serif"]]

_CANONICAL = Path("conf/accounts/core.yaml")
_COSTS = Path("conf/accounts/costs.yaml")
_LABELS_JA = Path("conf/accounts/labels_ja.yaml")
_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DIR = Path("data/interim/industry_12")

_SEED = 42
_N_SPLITS = 5
_REPEATS = {0: 20, 1: 10, 2: 10}  # seed0(主)は20回(ステージ11-Aと同じ)、他seedは感度確認で10回
_SPARSITY_THR = 0.01
_SIGN_TOL = 1e-4
_DESIGNS = (
    "current_asis", "current", "soft_l0.5", "soft_l1", "soft_l2", "soft_l5", "soft_l10",
    "hard33", "hard_coarse",
)
_SOFT = (("current", 0.0), ("soft_l0.5", 0.5), ("soft_l1", 1.0), ("soft_l2", 2.0),
         ("soft_l5", 5.0), ("soft_l10", 10.0))

# 図(dataviz): 参照パレットのカテゴリ slot1=青 / slot2=橙、ink・grid・axisはトークン値
_C1, _C2 = "#2a78d6", "#eb6834"
_INK, _INK2, _MUTED, _GRID, _AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
_SURFACE = "#fcfcfb"


def load_real(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    agg = pd.read_csv(_AGGREGATED_CSV, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS_CSV, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float)
    x = x / m["total_assets"].to_numpy(dtype=float)[:, None]
    return x, m["label"].astype(int).to_numpy() == 2


def identification_auc(
    generated: np.ndarray, real0: np.ndarray, repeats: int
) -> dict[str, np.ndarray]:
    """共通実装(``fsgen.evaluation.identification``、収束チェック付き)への薄いラッパー。"""
    return identification_auc_repeated(generated, real0, repeats, n_splits=_N_SPLITS, seed=_SEED)


def render_compare_md(
    lines: list[BSLine], col_labels: list[str], title: str, notes: list[str]
) -> str:
    out = [f"# {title}", "", *notes, "", "単位: 百万円", ""]
    out.append("| 科目 | " + " | ".join(col_labels) + " |")
    out.append("|---|" + "---|" * len(col_labels))
    for line in lines:
        if line.values is None:
            out.append(f"| **{line.label}** |" + " |" * len(col_labels))
            continue
        w = "**" if line.kind in ("subtotal", "total", "diff") else ""
        cells = " | ".join(f"{v / 1e6:,.0f}" for v in line.values)
        out.append(f"| {w}{'&nbsp;&nbsp;' * line.indent}{line.label}{w} | {cells} |")
    return "\n".join(out)


def make_figure(summary: pd.DataFrame, auc: pd.DataFrame, quality: pd.DataFrame) -> Path:
    lam = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    pos = np.arange(len(lam))
    names = [n for n, _ in _SOFT]
    s = summary.set_index("design")
    q = quality.set_index("design")
    a0 = auc[auc["seed"] == 0].set_index(["design", "classifier"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=_SURFACE)
    panels = [
        ("コスト重み付きproximity(中央値)", [s.loc[n, "proximity_cost_weighted"] for n in names],
         "proximity_cost_weighted", "低いほど動かす量が小さい"),
        ("業種一致率(33業種、対応先)", [q.loc[n, "match33"] for n in names],
         "match33", "高いほど同業種へ輸送"),
        ("識別AUC(seed0、20回平均±SD)", None, "auc", "0.5に近いほど実データと見分けがつかない"),
        ("validity(LightGBM健全確率、中央値)", [s.loc[n, "validity_lightgbm"] for n in names],
         "validity_lightgbm", "高いほど健全(class0)らしい"),
    ]
    for ax, (title, ys, key, sub) in zip(axes.ravel(), panels, strict=True):
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, loc="left", fontsize=11, color=_INK, pad=22)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.5, color=_MUTED, va="bottom")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(_AXIS)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.set_xticks(pos, [f"{v:g}" for v in lam])
        ax.set_xlabel("λ(× 基本コストの中央値)", fontsize=9, color=_INK2)
        if key == "auc":
            for clf, col, lab in (("lightgbm", _C1, "LightGBM"), ("logreg", _C2, "LogReg")):
                m = np.array([a0.loc[(n, clf), "auc_mean"] for n in names])
                e = np.array([a0.loc[(n, clf), "auc_std"] for n in names])
                ax.errorbar(pos, m, yerr=e, color=col, linewidth=2, marker="o", markersize=6,
                            markeredgecolor=_SURFACE, markeredgewidth=1.5, capsize=3, label=lab)
                for d in ("hard33", "hard_coarse"):
                    ax.axhline(a0.loc[(d, clf), "auc_mean"], color=col, linewidth=1,
                               linestyle=(0, (4, 3)), alpha=0.7)
            ax.set_ylim(0.5, 1.02)
            ax.axhline(0.5, color=_AXIS, linewidth=1)
            ax.text(len(lam) - 0.85, 0.51, "0.5=見分け不能", fontsize=8, color=_MUTED, va="bottom")
            ax.legend(frameon=False, fontsize=9, labelcolor=_INK2, loc="center left",
                      title="点線=ハード(33/粗)", title_fontsize=8)
        else:
            assert ys is not None
            ax.plot(pos, ys, color=_C1, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=_SURFACE, markeredgewidth=1.5)
            for d, cc in (("hard33", "ハード(33)"), ("hard_coarse", "ハード(粗)")):
                v = q.loc[d, "match33"] if key == "match33" else s.loc[d, key]
                ax.axhline(v, color=_INK2, linewidth=1, linestyle=(0, (4, 3)), alpha=0.8)
                ax.text(len(lam) - 0.85, v, cc, fontsize=8, color=_INK2, va="bottom", ha="left")
        if key == "match33":
            ax.set_ylim(0, 1.05)
        elif key == "validity_lightgbm":
            ax.set_ylim(0, 1.0)
        elif key == "proximity_cost_weighted":
            ax.set_ylim(0, 35)
        ax.set_xlim(-0.4, len(lam) + 0.6)
    fig.suptitle("ソフトペナルティ λ のトレードオフ(点線=ハード制約)", x=0.06, ha="left",
                 fontsize=13, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = _DIR / "figures" / "industry_tradeoff.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return out


def main() -> None:  # noqa: PLR0915
    spec = load_canonical_spec(_CANONICAL)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    cost_config = load_cost_config(_COSTS)
    w_plus, w_minus = cost_vectors(cost_config, names, non_actionable_mode="keep")
    signs = {b.name: b.sign for b in spec.basis}
    asset_idx = [i for i, b in enumerate(spec.basis) if b.side == "assets"]
    net_idx = [i for i, b in enumerate(spec.basis) if b.side == "net_assets"]

    gen = pd.read_parquet(_DIR / "generated_endpoints.parquet")
    quality = pd.read_csv(_DIR / "transport_quality.csv")
    x_all, is2 = load_real(names)
    print(f"読み込み: {len(gen)}行、{gen['design'].nunique()}設計")

    print("\n=== validity分類器の学習 ===")
    clfs = train_classifiers(x_all, is2, seed=_SEED)

    x0 = gen[[f"x0_{n}" for n in names]].to_numpy()
    x1 = gen[[f"x1_{n}" for n in names]].to_numpy()
    delta = x1 - x0
    m = gen[["design", "seed", "fold", "docID", "filerName"]].copy()
    m["proximity_l2"] = np.linalg.norm(delta, axis=1)
    m["proximity_cost_weighted"] = np.sum(np.where(delta >= 0, w_plus, w_minus) * np.abs(delta), 1)
    m["sparsity"] = np.sum(np.abs(delta) > _SPARSITY_THR, axis=1)
    m["balance_residual"] = np.max(np.abs(x1 @ c), axis=1)
    viol = np.zeros(len(gen), dtype=int)
    for j, n in enumerate(names):
        if signs[n] == "nonneg":
            viol += (x1[:, j] < -_SIGN_TOL).astype(int)
        elif signs[n] == "nonpos":
            viol += (x1[:, j] > _SIGN_TOL).astype(int)
    m["sign_violations"] = viol
    a_start = x0[:, asset_idx].sum(1)
    m["asset_change"] = (x1[:, asset_idx].sum(1) - a_start) / a_start
    for clf_name, model in clfs.items():
        m[f"validity_{clf_name}"] = 1.0 - model.predict_proba_class2(x1)
    m.to_csv(_DIR / "metrics.csv", index=False)

    cols = ["proximity_l2", "proximity_cost_weighted", "sparsity", "balance_residual",
            "sign_violations", "asset_change", "validity_lightgbm", "validity_logreg",
            "validity_random_forest"]
    summary = m.groupby("design")[cols].median().reindex(list(_DESIGNS)).reset_index()

    # --- 識別AUC(全設計×3 seed) ---
    print("\n=== 識別AUC(生成68点 vs class0無作為抽出68点、5-fold CV) ===")
    real0 = x_all[~is2]
    auc_rows = []
    per_rep: dict[tuple[str, int, str], np.ndarray] = {}
    for design in _DESIGNS:
        for seed, reps in _REPEATS.items():
            sub = gen[(gen["design"] == design) & (gen["seed"] == seed)]
            res = identification_auc(sub[[f"x1_{n}" for n in names]].to_numpy(), real0, reps)
            for clf_name, arr in res.items():
                per_rep[(design, seed, clf_name)] = arr
                auc_rows.append({
                    "design": design, "seed": seed, "classifier": clf_name, "n_points": len(sub),
                    "repeats": reps, "auc_mean": float(arr.mean()), "auc_std": float(arr.std()),
                })
        print(f"  {design:<13} 完了", flush=True)
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(_DIR / "identification_auc.csv", index=False)

    a0 = auc[auc["seed"] == 0].pivot(index="design", columns="classifier", values="auc_mean")
    a0s = auc[auc["seed"] == 0].pivot(index="design", columns="classifier", values="auc_std")
    a_all = auc.groupby(["design", "classifier"])["auc_mean"].mean().unstack()
    summary["auc_lightgbm_seed0"] = summary["design"].map(a0["lightgbm"])
    summary["auc_logreg_seed0"] = summary["design"].map(a0["logreg"])
    summary["auc_lightgbm_3seed"] = summary["design"].map(a_all["lightgbm"])
    summary["auc_logreg_3seed"] = summary["design"].map(a_all["logreg"])
    summary.to_csv(_DIR / "design_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== §3.2 指標(設計別の中央値、3seed×5fold×68社=204行) ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== 識別AUC(seed0、平均±SD) と 3seed平均 ===")
    for design in _DESIGNS:
        lg, lgs = a0.loc[design, "lightgbm"], a0s.loc[design, "lightgbm"]
        lr, lrs = a0.loc[design, "logreg"], a0s.loc[design, "logreg"]
        lg3, lr3 = a_all.loc[design, "lightgbm"], a_all.loc[design, "logreg"]
        print(
            f"  {design:<13} LGBM {lg:.4f}±{lgs:.4f}  LogReg {lr:.4f}±{lrs:.4f}"
            f"  | 3seed平均 LGBM {lg3:.4f} LogReg {lr3:.4f}"
        )
    print("\n=== 識別AUC: current との対応のある差(seed0、同じclass0抽出) ===")
    for design in _DESIGNS:
        for clf_name in ("lightgbm", "logreg"):
            diff = per_rep[(design, 0, clf_name)] - per_rep[("current", 0, clf_name)]
            print(f"  {design:<13} {clf_name:<9} 差={diff.mean():+.4f} ± {diff.std():.4f}")

    fig = make_figure(summary, auc, quality)
    print(f"\n図: {fig}")

    # --- §3.4 BS表 ---
    tg = pd.read_csv(_DIR / "full_targets.csv")
    seed0 = gen[gen["seed"] == 0]
    names_by_doc = seed0.drop_duplicates("docID").set_index("docID")["filerName"]
    named = {
        "株式会社アマガサ": None, "株式会社ｅｎｉｓｈ": None,
        "アーキテクツ・スタジオ・ジャパン株式会社": None,
    }
    chosen: list[str] = []
    for nm in named:
        hit = names_by_doc[names_by_doc == nm]
        chosen.append(str(hit.index[0]))
    cur0 = seed0[seed0["design"] == "current"].set_index("docID")
    x0_cur = cur0[[f"x0_{n}" for n in names]].to_numpy()
    neg = [d for d, v in zip(cur0.index, x0_cur[:, net_idx].sum(1), strict=True) if v < 0]
    cand = sorted(d for d in neg if d not in chosen and d != "S100TWQJ")
    pick2 = list(np.random.default_rng(0).choice(cand, size=2, replace=False))
    chosen += [str(p) for p in pick2]
    print(f"\n=== §3.4 BS表対象(債務超過{len(neg)}社から2社を無作為抽出、東証亜鉛は除外): {chosen}")
    labels_map = load_labels(_LABELS_JA)
    samples = _DIR / "samples"
    samples.mkdir(exist_ok=True)
    key_items = ("jppfs_cor_CashAndDeposits", "jppfs_cor_RetainedEarnings",
                 "short_term_loans", "long_term_loans")
    for doc in chosen:
        rows = {dz: seed0[(seed0["design"] == dz) & (seed0["docID"] == doc)].iloc[0]
                for dz in ("current_asis", "current", "hard33")}
        asset = float(rows["current"]["assets_start"])
        xs0 = rows["current"][[f"x0_{n}" for n in names]].to_numpy(dtype=float) * asset
        xc = rows["current"][[f"x1_{n}" for n in names]].to_numpy(dtype=float) * asset
        xh = rows["hard33"][[f"x1_{n}" for n in names]].to_numpy(dtype=float) * asset
        lines = build_bs_lines(spec, names, [xs0, xc, xh], labels_map)
        tinfo = {dz: tg[(tg["design"] == dz) & (tg["source_docID"] == doc)].iloc[0]
                 for dz in ("current_asis", "current", "hard33")}
        src_ind = tinfo["current"]["source_ind33"]
        notes = [f"始点の業種: {src_ind}"]
        labs = (("current_asis", "現行(as-run)"), ("current", "現行(収束版)"),
                ("hard33", "ハード(33)"))
        for dz, dlab in labs:
            t = tinfo[dz]
            notes.append(
                f"- {dlab}の対応先: {t['target_name']}({t['target_ind33']}、"
                f"RE/資産={t['target_RE_over_assets']:.2f})"
            )
        title = f"{names_by_doc[doc]}({doc})"
        md = render_compare_md(
            lines, ["始点", "現行(収束版)の終点", "ハード(33)の終点"], title, notes
        )
        (samples / f"{doc}.md").write_text(md, encoding="utf-8")
        print(f"\n--- {title}  [{src_ind}] 総資産{asset / 1e6:,.0f}百万円 ---")
        for dz, dlab in labs:
            t = tinfo[dz]
            print(f"  {dlab:<12} 対応先: {t['target_name']}[{t['target_ind33']}] "
                  f"RE/資産={t['target_RE_over_assets']:.2f}")
        for k in key_items:
            j = names.index(k)
            print(f"  {k:<28} 始点{xs0[j] / 1e6:>9,.0f}  現行{xc[j] / 1e6:>9,.0f}  "
                  f"ハード{xh[j] / 1e6:>9,.0f}")
    print(f"\n出力先: {_DIR}")


if __name__ == "__main__":
    main()
