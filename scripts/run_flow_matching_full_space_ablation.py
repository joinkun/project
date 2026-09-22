#!/usr/bin/env python
"""ステージ10 §2.1: 低次元(z, 13次元) vs 元空間(x̃, 30次元)でのFM学習比較。

低次元版は標本複雑度で有利(class2は67社しかない)が、ステージ9で判明した
通り基底Bが密なため科目ごとの制約が効きにくい、というトレードオフを
検証する。1 fold・1 seedの縮小規模で、学習・検証損失の推移(過学習の
度合い)とサンプリング結果の指標を比較する。

使い方:
    uv run python scripts/run_flow_matching_full_space_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", *plt.rcParams["font.sans-serif"]]

from fsgen.accounting.constraints import (  # noqa: E402
    build_constraint_matrix,
    load_canonical_spec,
    state_vector_names,
)
from fsgen.accounting.subspace import build_reduced_basis  # noqa: E402
from fsgen.evaluation import metrics as m  # noqa: E402
from fsgen.models.cost import (  # noqa: E402
    cost_vectors,
    load_cost_config,
    weighted_asymmetric_cost_matrix,
)
from fsgen.models.flow_matching import VelocityField, euler_sample_batch, train_fm  # noqa: E402
from fsgen.models.transport import solve_ot_from_cost  # noqa: E402

_K = 13
_EPSILON = 0.05
_TS = (0.0, 0.25, 0.5, 0.75, 1.0)
_EPOCHS = 50  # run_flow_matching_evaluation.pyと同じ理由(検証損失は早期に頭打ち)
_LR = 1e-3
_HIDDEN = 128
_N_LAYERS = 3
_SEED = 0


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


def make_folds(n2: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """class2企業(0..n2-1)をn_folds分割する(run_flow_matching_evaluation.pyと同一ロジック)。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n2)
    chunks = np.array_split(order, n_folds)
    folds = []
    for i in range(n_folds):
        test_pos = chunks[i]
        train_pos = np.concatenate([chunks[j] for j in range(n_folds) if j != i])
        folds.append((train_pos, test_pos))
    return folds


def main() -> None:
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    c = build_constraint_matrix(spec, include_subtotals=False)
    cost_config = load_cost_config(Path("conf/accounts/costs.yaml"))
    w_plus_ot, w_minus_ot = cost_vectors(cost_config, names, non_actionable_mode="ignore")

    doc_ids, filer_names, x, assets, labels = load_dataset(
        Path("data/interim/values/aggregated_core.csv"),
        Path("data/interim/labels/labels.csv"),
        names,
    )
    x_tilde = x / assets[:, None]
    idx0 = [i for i, lab in enumerate(labels) if lab == 0]
    idx2 = [i for i, lab in enumerate(labels) if lab == 2]
    x0_tilde = x_tilde[idx0]
    x2_tilde = x_tilde[idx2]
    n2 = len(idx2)
    print(f"class0: {len(idx0)}社, class2: {n2}社")

    folds = make_folds(n2, n_folds=5, seed=0)
    train_pos, test_pos = folds[0]
    print(f"fold0: train={len(train_pos)}社, test={len(test_pos)}社(seed={_SEED}のみ)")

    cost_train = weighted_asymmetric_cost_matrix(
        x2_tilde[train_pos], x0_tilde, w_plus_ot, w_minus_ot
    )
    plan_train = solve_ot_from_cost(cost_train, _EPSILON)
    cost_test = weighted_asymmetric_cost_matrix(x2_tilde[test_pos], x0_tilde, w_plus_ot, w_minus_ot)
    plan_test = solve_ot_from_cost(cost_test, _EPSILON)

    results = {}

    # --- 低次元版(z, 13次元) ---
    pool_idx = idx0 + [idx2[p] for p in train_pos]
    b = build_reduced_basis(c, x_tilde[pool_idx], k=_K)
    z0 = x0_tilde @ b
    z_train = x2_tilde[train_pos] @ b
    z_test = x2_tilde[test_pos] @ b
    net_low = VelocityField(dim=_K, hidden=_HIDDEN, n_layers=_N_LAYERS).double()
    result_low = train_fm(
        z_train, z0, plan_train, net_low, epochs=_EPOCHS, lr=_LR, seed=_SEED,
        z_val=z_test, plan_val=plan_test,
    )
    results["low_dim(z,13)"] = result_low

    z_start_test = x2_tilde[test_pos] @ b
    z_path_low = euler_sample_batch(net_low, z_start_test, _TS)
    disp = z_path_low - z_start_test[:, None, :]
    x_path_low_tilde = x2_tilde[test_pos][:, None, :] + np.einsum("dk,ntk->ntd", b, disp)

    # --- 元空間版(x̃, 30次元、Bを経由しない) ---
    net_full = VelocityField(dim=len(names), hidden=_HIDDEN, n_layers=_N_LAYERS).double()
    result_full = train_fm(
        x2_tilde[train_pos], x0_tilde, plan_train, net_full, epochs=_EPOCHS, lr=_LR, seed=_SEED,
        z_val=x2_tilde[test_pos], plan_val=plan_test,
    )
    results["full_space(x,30)"] = result_full
    x_path_full_tilde = euler_sample_batch(net_full, x2_tilde[test_pos], _TS)

    print("\n=== 学習/検証損失の最終値、過学習の度合い(val/train比) ===")
    for label, res in results.items():
        train_final = res.train_losses[-1]
        val_final = res.val_losses[-1] if res.val_losses else float("nan")
        ratio = val_final / train_final if train_final > 0 else float("nan")
        print(
            f"  {label:<18} train={train_final:.4f}, val={val_final:.4f}, "
            f"val/train比={ratio:.2f}"
        )

    print("\n=== サンプリング結果の指標(射影なし、テスト企業のみ) ===")
    n_test = len(test_pos)
    assets_test = assets[[idx2[p] for p in test_pos]]
    path_variants = (("low_dim(z,13)", x_path_low_tilde), ("full_space(x,30)", x_path_full_tilde))
    for label, x_path_tilde in path_variants:
        x_path_raw = x_path_tilde * assets_test[:, None, None]
        prox = np.median(
            [m.proximity_l2(x_path_raw[i], float(assets_test[i])) for i in range(n_test)]
        )
        sparsity = np.median(
            [m.sparsity(x_path_raw[i], float(assets_test[i])) for i in range(n_test)]
        )
        resid = np.median(
            [
                m.endpoint_balance_residual(x_path_raw[i], c, float(assets_test[i]))
                for i in range(n_test)
            ]
        )
        print(
            f"  {label:<18} proximity中央値={prox:.4f}, sparsity中央値={sparsity:.1f}, "
            f"終点恒等式残差中央値={resid:.4e}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for label, res in results.items():
        axes[0].plot(res.train_losses, label=f"{label} (train)")
        axes[1].plot(res.val_losses, label=f"{label} (val)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation loss")
    axes[1].legend(fontsize=8)
    fig.suptitle("低次元 vs 元空間: 学習・検証損失(fold0, seed=0)")
    fig.tight_layout()
    out_dir = Path("data/interim/flow_matching/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "low_dim_vs_full_space.png", dpi=150)
    plt.close(fig)
    print(f"\n図を出力: {out_dir / 'low_dim_vs_full_space.png'}")


if __name__ == "__main__":
    main()
