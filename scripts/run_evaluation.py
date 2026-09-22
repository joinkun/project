#!/usr/bin/env python
"""ステージ8 §B: 評価基盤を実行し、手法比較(§C)の元データを作る。

4つのベースライン(prior_work / convex_qp / naive_ot / gaussian)と提案手法
(proposed)を leave-one-out で比較する。

使い方:
    uv run python scripts/run_evaluation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", *plt.rcParams["font.sans-serif"]]

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.projection_qp import load_direction_constraints  # noqa: E402
from fsgen.evaluation import metrics as m  # noqa: E402
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402
from fsgen.evaluation.loo import LooDataset, run_loo  # noqa: E402

_DEFAULT_CANONICAL = Path("conf/accounts/core.yaml")
_DEFAULT_AGGREGATED_CSV = Path("data/interim/values/aggregated_core.csv")
_DEFAULT_LABELS_CSV = Path("data/interim/labels/labels.csv")
_DEFAULT_OUTPUT_DIR = Path("data/interim/evaluation")
_K = 13
_EPSILON = 0.05
_TS = (0.0, 0.25, 0.5, 0.75, 1.0)
_SEED = 42
_METHOD_ORDER = ("prior_work", "convex_qp", "naive_ot", "gaussian", "proposed")
_METHOD_LABELS_JA = {
    "prior_work": "先行研究方式",
    "convex_qp": "直接QP",
    "naive_ot": "OT+線形補間(射影なし)",
    "gaussian": "ガウス近似(Bures-Wasserstein)",
    "proposed": "提案手法(OT+射影)",
}


def load_dataset(
    aggregated_csv: Path, labels_csv: Path, names: tuple[str, ...]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[int]]:
    agg = pd.read_csv(aggregated_csv, dtype={"docID": str})
    is_financial = agg["is_financial"].astype(str)
    agg = agg[(agg["parse_status"] == "ok") & (is_financial != "True")]
    labels_df = pd.read_csv(labels_csv, dtype={"docID": str})
    labels_df = labels_df[pd.to_numeric(labels_df["label"], errors="coerce").notna()]
    merged = agg.merge(labels_df[["docID", "filerName", "label"]], on="docID", how="inner")
    merged = merged[merged["total_assets"].notna() & (merged["total_assets"] != 0)]
    x = merged[list(names)].fillna(0.0).to_numpy(dtype=float)
    assets = merged["total_assets"].to_numpy(dtype=float)
    labels = merged["label"].astype(int).tolist()
    return merged["docID"].tolist(), merged["filerName"].tolist(), x, assets, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=_DEFAULT_CANONICAL)
    parser.add_argument("--aggregated-csv", type=Path, default=_DEFAULT_AGGREGATED_CSV)
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    spec = load_canonical_spec(args.canonical)
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    direction_constraints = load_direction_constraints(args.canonical)

    doc_ids, filer_names, x, assets, labels = load_dataset(
        args.aggregated_csv, args.labels_csv, names
    )
    ds = LooDataset(
        doc_ids=doc_ids,
        filer_names=filer_names,
        x=x,
        assets=assets,
        labels=labels,
        names=names,
        spec=spec,
        c=c,
        direction_constraints=direction_constraints,
    )
    print(f"class0: {len(ds.idx0)}社, class2: {len(ds.idx2)}社")

    def _progress(done: int, total: int) -> None:
        print(f"  LOO進捗: {done}/{total}")

    result = run_loo(ds, _TS, k=_K, epsilon=_EPSILON, on_progress=_progress)
    metrics_df = result.metrics_df

    # --- validity: LOOの外側で1回だけ学習した3分類器でP(class2|x(1))を測る ---
    #
    # ステージ8.5 §A: classifiers.predict_proba_class2 はその名の通り
    # P(class2|x)(不健全確率、低いほど良い)を返す。しかし報告・図・表では
    # 「validity=高いほど良い(健全確率)」という反実仮想説明の標準的な
    # 慣習で統一する(ステージ8の報告はこの反転を怠っており、結論の一部が
    # 逆向きになっていた——詳細はステージ8.5の報告参照)。
    # ここで health_probability = 1 - P(class2|x) = P(class0|x) に変換し、
    # 以降 "validity" 列は常にこの「健全確率」を指す。
    print("\n=== validity分類器の学習 ===")
    x_all_tilde = ds.x / ds.assets[:, None]
    is_class2 = np.array([lab == 2 for lab in ds.labels])
    classifiers = train_classifiers(x_all_tilde, is_class2, seed=_SEED)
    print(f"学習に使った企業数: {len(ds.x)}社(class2={int(is_class2.sum())}社)")

    for clf_name, clf in classifiers.items():
        col = f"validity_{clf_name}"
        metrics_df[col] = np.nan
        for method in _METHOD_ORDER:
            p_class2 = clf.predict_proba_class2(result.endpoints[method])
            health_probability = 1.0 - p_class2
            mask = metrics_df["method"] == method
            metrics_df.loc[mask, col] = health_probability

    metrics_df["validity"] = metrics_df["validity_lightgbm"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"\nmetrics.csv を書き出しました: {args.output_dir / 'metrics.csv'}")

    # --- §B.2 plausibility(手法ごとにプールレベルで計算) ---
    print("\n=== plausibility(手法ごと) ===")
    x0_tilde = ds.x[ds.idx0] / ds.assets[ds.idx0][:, None]
    real_nn_dist = m.pairwise_nearest_neighbor_distances(x0_tilde)
    rng = np.random.default_rng(_SEED)
    sample_idx = rng.choice(len(x0_tilde), size=min(len(ds.idx2), len(x0_tilde)), replace=False)
    real_sample = x0_tilde[sample_idx]

    plausibility_rows = []
    for method in _METHOD_ORDER:
        gen = result.endpoints[method]
        nn_percentiles = [
            m.percentile_rank(m.nearest_neighbor_distance(g, x0_tilde), real_nn_dist) for g in gen
        ]
        auc = m.discriminative_auc(gen, real_sample, seed=_SEED)
        corr_dist = m.correlation_frobenius_distance(gen, x0_tilde)
        plausibility_rows.append(
            {
                "method": method,
                "nn_percentile_median": float(np.median(nn_percentiles)),
                "discriminative_auc": auc,
                "correlation_frobenius_distance": corr_dist,
            }
        )
        print(
            f"  {method:<12} NN percentile中央値={np.median(nn_percentiles):6.1f}, "
            f"識別AUC={auc:.3f}, 相関Frobenius距離={corr_dist:.3f}"
        )
    plausibility_df = pd.DataFrame(plausibility_rows)
    plausibility_df.to_csv(args.output_dir / "plausibility.csv", index=False)

    # --- §B.4 敵対的操作チェック: 分類器間の手法順位相関 ---
    print("\n=== §B.4 分類器間の手法順位相関(validity平均でランク付け) ===")
    mean_validity = metrics_df.groupby("method")[
        ["validity_lightgbm", "validity_logreg", "validity_random_forest"]
    ].mean()
    mean_validity = mean_validity.loc[list(_METHOD_ORDER)]
    print(mean_validity)
    rank_lgb = mean_validity["validity_lightgbm"].rank(ascending=False)
    rank_lr = mean_validity["validity_logreg"].rank(ascending=False)
    rank_rf = mean_validity["validity_random_forest"].rank(ascending=False)
    rho_lgb_lr, _ = spearmanr(rank_lgb, rank_lr)
    rho_lgb_rf, _ = spearmanr(rank_lgb, rank_rf)
    rho_lr_rf, _ = spearmanr(rank_lr, rank_rf)
    print(f"Spearman順位相関: lightgbm-logreg={rho_lgb_lr:.3f}, "
          f"lightgbm-rf={rho_lgb_rf:.3f}, logreg-rf={rho_lr_rf:.3f}")

    # --- §C.2 手法×指標の集計表 ---
    print("\n=== 手法×指標 集計表(中央値) ===")
    summary_cols = [
        "proximity_l2",
        "proximity_l1",
        "proximity_l2_weighted",
        "proximity_l1_weighted",
        "sparsity",
        "balance_residual_ratio",
        "sign_violations",
        "intermediate_violations",
        "direction_violations",
        "monotonicity_reversals",
        "validity_lightgbm",
        "validity_logreg",
        "validity_random_forest",
    ]
    summary_median = metrics_df.groupby("method")[summary_cols].median().loc[list(_METHOD_ORDER)]
    summary_q1 = metrics_df.groupby("method")[summary_cols].quantile(0.25).loc[list(_METHOD_ORDER)]
    summary_q3 = metrics_df.groupby("method")[summary_cols].quantile(0.75).loc[list(_METHOD_ORDER)]
    print(summary_median)

    # --- §C.4 proposed vs convex_qp ---
    print("\n=== proposed vs convex_qp(直接比較) ===")
    prop = metrics_df[metrics_df["method"] == "proposed"].set_index("docID")
    cqp = metrics_df[metrics_df["method"] == "convex_qp"].set_index("docID")
    prox_diff = prop["proximity_l2"] - cqp["proximity_l2"]
    valid_diff = prop["validity_lightgbm"] - cqp["validity_lightgbm"]
    prop_wins_proximity = int(np.sum(prox_diff < 0))
    prop_wins_validity = int(np.sum(valid_diff > 0))
    print(f"proximityでproposedが勝った(小さい)社数: {prop_wins_proximity}/{len(prox_diff)}")
    print(f"validityでproposedが勝った(大きい)社数: {prop_wins_validity}/{len(valid_diff)}")
    print(f"proximity中央値: proposed={prop['proximity_l2'].median():.4e}, "
          f"convex_qp={cqp['proximity_l2'].median():.4e}")
    print(f"validity(lightgbm)中央値: proposed={prop['validity_lightgbm'].median():.4f}, "
          f"convex_qp={cqp['validity_lightgbm'].median():.4f}")

    # --- 図: tradeoff.png ---
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(_METHOD_ORDER)))
    for method, color in zip(_METHOD_ORDER, colors, strict=True):
        sub = metrics_df[metrics_df["method"] == method]
        ax.scatter(
            sub["proximity_l2"], sub["validity_lightgbm"], label=_METHOD_LABELS_JA[method],
            color=color, alpha=0.7, s=25,
        )
    ax.set_xlabel("proximity (‖x(1)-x(0)‖ / Assets_0、小さいほど近い)")
    ax.set_ylabel("validity (LightGBM分類器による健全確率 1-P(class2|x(1)))")
    ax.set_title("proximity vs validity(左上が理想)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "tradeoff.png", dpi=150)
    plt.close(fig)

    # --- 図: feasibility.png ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    box_data_residual = [
        metrics_df[metrics_df["method"] == m_]["balance_residual_ratio"] for m_ in _METHOD_ORDER
    ]
    axes[0].boxplot(box_data_residual, tick_labels=[_METHOD_LABELS_JA[m_] for m_ in _METHOD_ORDER])
    axes[0].set_yscale("symlog", linthresh=1e-12)
    axes[0].set_ylabel("恒等式残差 / Assets_0(最大)")
    axes[0].tick_params(axis="x", rotation=30)
    box_data_signs = [
        metrics_df[metrics_df["method"] == m_]["sign_violations"] for m_ in _METHOD_ORDER
    ]
    axes[1].boxplot(box_data_signs, tick_labels=[_METHOD_LABELS_JA[m_] for m_ in _METHOD_ORDER])
    axes[1].set_ylabel("符号違反件数(全t合計)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("手法別の実行可能性")
    fig.tight_layout()
    fig.savefig(figures_dir / "feasibility.png", dpi=150)
    plt.close(fig)
    print(f"\n図を出力しました: {figures_dir}")

    # --- summary.md ---
    lines = ["# ステージ8 評価基盤: 手法比較サマリ\n"]
    lines.append("## 手法×指標(中央値 [Q1, Q3])\n")
    lines.append("| 指標 | " + " | ".join(_METHOD_LABELS_JA[m_] for m_ in _METHOD_ORDER) + " |")
    lines.append("|---|" + "---|" * len(_METHOD_ORDER))
    for col in summary_cols:
        cells = []
        for m_ in _METHOD_ORDER:
            med = summary_median.loc[m_, col]
            q1 = summary_q1.loc[m_, col]
            q3 = summary_q3.loc[m_, col]
            cells.append(f"{med:.3g} [{q1:.3g}, {q3:.3g}]")
        lines.append(f"| {col} | " + " | ".join(cells) + " |")
    lines.append("\n## plausibility(プールレベル)\n")
    lines.append(
        "| 手法 | NN percentile中央値 | 識別AUC | 相関Frobenius距離 |\n|---|---|---|---|"
    )
    for _, row_p in plausibility_df.iterrows():
        lines.append(
            f"| {_METHOD_LABELS_JA[row_p['method']]} | {row_p['nn_percentile_median']:.1f} | "
            f"{row_p['discriminative_auc']:.3f} | {row_p['correlation_frobenius_distance']:.3f} |"
        )
    lines.append("\n## §B.4 分類器間の手法順位相関(Spearman)\n")
    lines.append(
        f"- lightgbm vs logreg: {rho_lgb_lr:.3f}\n"
        f"- lightgbm vs random_forest: {rho_lgb_rf:.3f}\n"
        f"- logreg vs random_forest: {rho_lr_rf:.3f}\n"
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"summary.md を書き出しました: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
